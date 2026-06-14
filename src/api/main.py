"""
Tamkeen Adaptive Skill Assessment API

Serves the trained skill predictor model via FastAPI.
Uses adaptive question selection: harder after correct answers, easier after wrong.

Endpoints:
    POST /start_assessment  - Begin a session, get first question
    POST /submit_answer     - Submit answer, get next question or finish
    POST /get_result        - Aggregate answers, predict skill level
"""

import os
import uuid
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tamkeen Skill Assessment API",
    description="Adaptive quiz engine that predicts student skill levels using ML.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

model_data = None                       # loaded .pkl dict
sessions: Dict[str, dict] = {}          # session_id -> session state

MAX_QUESTIONS = 10

# disability_type -> time multiplier (matches generate_data.py)
DISABILITY_MULTIPLIERS = {
    "visual":    3.5,
    "motor":     2.5,
    "cognitive": 1.8,
    "hearing":   1.3,
    "none":      1.0,
}


# ---------------------------------------------------------------------------
# Question bank  (difficulty 1-5, matching the training data)
# In production this would come from the database.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Question bank loader
# Loads from data/raw/questions.csv at startup
# ---------------------------------------------------------------------------

QUESTIONS: Dict[str, Dict[int, List[dict]]] = {}  # assessment_id -> difficulty -> [questions]

def load_questions_from_csv():
    global QUESTIONS
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, "data", "raw", "questions_v2.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "raw", "questions.csv")
    
    if not os.path.exists(csv_path):
        logger.warning(f"Questions CSV not found at {csv_path}. Using fallback questions.")
        # Minimal fallback
        QUESTIONS = {"default": {1: [{"id": "fallback", "text": "What is 1+1?", "options": ["1","2","3","4"], "answer": 1, "difficulty": 1}]}}
        return

    try:
        df = pd.read_csv(csv_path)
        count = 0
        for _, row in df.iterrows():
            a_id = str(row['assessment_id'])
            diff = int(row['difficulty_level'])
            
            if a_id not in QUESTIONS:
                QUESTIONS[a_id] = {}
            if diff not in QUESTIONS[a_id]:
                QUESTIONS[a_id][diff] = []
            
            # Options in CSV are piped: "Opt A|Opt B|Opt C|Opt D"
            options = str(row['options']).split('|')
            # Correct answer in CSV is 'A' (mapped to index 0)
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'True': 0, 'False': 1}
            ans_idx = answer_map.get(str(row['correct_answer']), 0)

            QUESTIONS[a_id][diff].append({
                "id": str(row['question_id']),
                "text": str(row['question_text']),
                "options": options,
                "answer": ans_idx,
                "difficulty": diff
            })
            count += 1
        logger.info(f"Loaded {count} questions for {len(QUESTIONS)} assessments from CSV.")
    except Exception as e:
        logger.error(f"Error loading questions: {e}")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    user_id: str = Field(..., example="user_42")
    course_id: str = Field(..., example="course_7")
    disability_type: str = Field("none", example="visual")


class QuestionOut(BaseModel):
    question_id: str
    text: str
    options: List[str]
    difficulty: int


class StartResponse(BaseModel):
    session_id: str
    message: str
    first_question: QuestionOut
    total_questions: int


class SubmitRequest(BaseModel):
    session_id: str
    question_id: str
    selected_option: int = Field(..., ge=0, le=3)
    response_time: float = Field(..., gt=0, description="seconds")


class SubmitResponse(BaseModel):
    status: str                     # "continue" or "finished"
    is_correct: bool
    questions_answered: int
    questions_remaining: int
    next_question: Optional[QuestionOut] = None


class ResultRequest(BaseModel):
    session_id: str


class AnswerDetail(BaseModel):
    question_id: str
    difficulty: int
    is_correct: bool
    response_time: float


class ResultResponse(BaseModel):
    session_id: str
    user_id: str
    course_id: str
    disability_type: str
    skill_level: str
    confidence_score: float
    total_correct: int
    total_questions: int
    accuracy: float
    avg_response_time: float
    answers: List[AnswerDetail]


# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global model_data
    
    # 1. Load model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "skill_predictor.pkl")

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
    else:
        model_data = joblib.load(model_path)
        logger.info(f"Model loaded: {model_data['model_name']} "
                    f"(test acc {model_data['test_accuracy']:.4f})")
    
    # 2. Load questions
    load_questions_from_csv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_question(difficulty: int, used_ids: set, assessment_id: str = "default") -> Optional[dict]:
    """Pick a random unused question at the target difficulty for the given assessment.
    Falls back to adjacent difficulties if none available."""
    difficulty = max(1, min(5, difficulty))
    
    # Map course_id (e.g. course_7) to assessment_id (e.g. assessment_7) if needed
    if assessment_id.startswith("course_"):
        assessment_id = assessment_id.replace("course_", "assessment_")

    # If the specific assessment doesn't exist, try 'assessment_0' or 'default' or any first available
    if assessment_id not in QUESTIONS:
        logger.warning(f"Assessment {assessment_id} not found in bank. Falling back.")
        if "assessment_0" in QUESTIONS:
            assessment_id = "assessment_0"
        elif QUESTIONS:
            assessment_id = list(QUESTIONS.keys())[0]
        else:
            return None

    bank = QUESTIONS[assessment_id]

    for d in [difficulty, difficulty - 1, difficulty + 1, difficulty - 2, difficulty + 2]:
        if d < 1 or d > 5:
            continue
        available = [q for q in bank.get(d, []) if q["id"] not in used_ids]
        if available:
            return random.choice(available)
    return None


def format_question(q: dict) -> QuestionOut:
    return QuestionOut(
        question_id=q["id"],
        text=q["text"],
        options=q["options"],
        difficulty=q.get("difficulty", 3),
    )


def next_difficulty(current: int, is_correct: bool) -> int:
    """Adaptive logic: correct -> harder, wrong -> easier."""
    if is_correct:
        return min(5, current + 1)
    else:
        return max(1, current - 1)


# ---------------------------------------------------------------------------
# Inject difficulty into question dicts (so we don't repeat it everywhere)
# ---------------------------------------------------------------------------
for _diff, _qlist in QUESTIONS.items():
    for _q in _qlist:
        _q["difficulty"] = _diff


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_data is not None,
        "active_sessions": len(sessions),
    }


@app.post("/start_assessment", response_model=StartResponse)
async def start_assessment(req: StartRequest):
    """Start a new assessment session. Returns the first question at medium difficulty."""

    disability = req.disability_type.lower().strip()
    if disability not in DISABILITY_MULTIPLIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid disability_type. Valid: {list(DISABILITY_MULTIPLIERS.keys())}",
        )

    session_id = f"sess_{uuid.uuid4().hex[:12]}"

    # Start with medium difficulty (3)
    first_q = pick_question(difficulty=3, used_ids=set(), assessment_id=req.course_id)
    if first_q is None:
        raise HTTPException(status_code=500, detail="No questions available")

    sessions[session_id] = {
        "user_id": req.user_id,
        "course_id": req.course_id,
        "disability_type": disability,
        "current_difficulty": first_q["difficulty"],
        "current_question": first_q,
        "answers": [],
        "used_ids": {first_q["id"]},
        "finished": False,
        "started_at": datetime.utcnow().isoformat(),
    }

    logger.info(f"Session started: {session_id} user={req.user_id} disability={disability}")

    return StartResponse(
        session_id=session_id,
        message="Assessment started",
        first_question=format_question(first_q),
        total_questions=MAX_QUESTIONS,
    )


@app.post("/submit_answer", response_model=SubmitResponse)
async def submit_answer(req: SubmitRequest):
    """Submit an answer. Returns next question (adaptive) or finished status."""

    session = sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["finished"]:
        raise HTTPException(status_code=400, detail="Session already finished. Call /get_result.")

    current_q = session["current_question"]
    if current_q["id"] != req.question_id:
        raise HTTPException(
            status_code=400,
            detail=f"Expected question '{current_q['id']}', got '{req.question_id}'",
        )

    # Check answer
    is_correct = req.selected_option == current_q["answer"]

    session["answers"].append({
        "question_id": current_q["id"],
        "difficulty": current_q["difficulty"],
        "is_correct": is_correct,
        "response_time": req.response_time,
    })

    answered = len(session["answers"])
    remaining = MAX_QUESTIONS - answered

    # Finished?
    if answered >= MAX_QUESTIONS:
        session["finished"] = True
        session["current_question"] = None
        return SubmitResponse(
            status="finished",
            is_correct=is_correct,
            questions_answered=answered,
            questions_remaining=0,
            next_question=None,
        )

    # Adaptive: pick next question
    new_diff = next_difficulty(current_q["difficulty"], is_correct)
    session["current_difficulty"] = new_diff

    nq = pick_question(
        difficulty=new_diff, 
        used_ids=session["used_ids"], 
        assessment_id=session["course_id"]
    )
    if nq is None:
        session["finished"] = True
        session["current_question"] = None
        return SubmitResponse(
            status="finished",
            is_correct=is_correct,
            questions_answered=answered,
            questions_remaining=0,
            next_question=None,
        )

    session["current_question"] = nq
    session["used_ids"].add(nq["id"])

    return SubmitResponse(
        status="continue",
        is_correct=is_correct,
        questions_answered=answered,
        questions_remaining=remaining,
        next_question=format_question(nq),
    )


@app.post("/get_result", response_model=ResultResponse)
async def get_result(req: ResultRequest):
    """Aggregate answers, build features, run model.predict(), return skill level."""

    session = sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session["finished"]:
        raise HTTPException(
            status_code=400,
            detail=f"Not finished yet. {len(session['answers'])}/{MAX_QUESTIONS} answered.",
        )
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    answers = session["answers"]
    disability = session["disability_type"]

    # ---- Build the 12 features the model expects ----
    # The model was trained on per-user-course aggregated features.
    # We replicate that same aggregation here from the session answers.

    difficulties = [a["difficulty"] for a in answers]
    correct_flags = [1 if a["is_correct"] else 0 for a in answers]
    times = [a["response_time"] for a in answers]

    disability_map = model_data["disability_map"]       # {'visual':0, 'motor':1, ...}
    label_map = model_data["label_map"]                 # {'Beginner':0, ...}
    features = model_data["features"]                   # list of 12 feature names
    reverse_label = {v: k for k, v in label_map.items()}

    num_questions = len(answers)
    avg_difficulty = np.mean(difficulties)
    max_difficulty = max(difficulties)
    accuracy = np.mean(correct_flags)
    total_correct = sum(correct_flags)
    avg_response_time = np.mean(times)
    std_response_time = np.std(times, ddof=1) if len(times) > 1 else 0.0
    max_response_time = max(times)
    min_response_time = min(times)
    time_range = max_response_time - min_response_time
    time_per_difficulty = avg_response_time / avg_difficulty if avg_difficulty > 0 else 0.0
    disability_encoded = disability_map.get(disability, disability_map.get("none", 0))

    row = pd.DataFrame([{
        "disability_encoded": disability_encoded,
        "num_questions": num_questions,
        "avg_difficulty": avg_difficulty,
        "max_difficulty": max_difficulty,
        "accuracy": accuracy,
        "total_correct": total_correct,
        "avg_response_time": avg_response_time,
        "std_response_time": std_response_time,
        "max_response_time": max_response_time,
        "min_response_time": min_response_time,
        "time_range": time_range,
        "time_per_difficulty": time_per_difficulty,
    }])

    # Reorder columns to match training order
    row = row[features]

    model = model_data["model"]
    prediction = int(model.predict(row)[0])
    skill_level = reverse_label.get(prediction, "Unknown")

    # Confidence from predict_proba
    probas = model.predict_proba(row)[0]
    confidence = float(probas[prediction])

    logger.info(
        f"Result for {req.session_id}: {skill_level} "
        f"(confidence={confidence:.2f}, accuracy={accuracy:.2f})"
    )

    return ResultResponse(
        session_id=req.session_id,
        user_id=session["user_id"],
        course_id=session["course_id"],
        disability_type=disability,
        skill_level=skill_level,
        confidence_score=round(confidence, 4),
        total_correct=total_correct,
        total_questions=num_questions,
        accuracy=round(accuracy, 4),
        avg_response_time=round(avg_response_time, 2),
        answers=[AnswerDetail(**a) for a in answers],
    )
