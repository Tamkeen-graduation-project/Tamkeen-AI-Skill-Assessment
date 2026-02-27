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

QUESTIONS: Dict[int, List[dict]] = {
    1: [
        {"id": "q1_1", "text": "What does HTML stand for?",
         "options": ["Hyper Text Markup Language", "High Tech Modern Language",
                     "Hyper Transfer Markup Language", "Home Tool Markup Language"],
         "answer": 0},
        {"id": "q1_2", "text": "Which symbol starts a comment in Python?",
         "options": ["//", "#", "/*", "--"], "answer": 1},
        {"id": "q1_3", "text": "What data type stores True or False?",
         "options": ["String", "Integer", "Boolean", "Float"], "answer": 2},
        {"id": "q1_4", "text": "What does CPU stand for?",
         "options": ["Central Processing Unit", "Central Program Utility",
                     "Computer Personal Unit", "Central Processor Unifier"],
         "answer": 0},
        {"id": "q1_5", "text": "Which tag makes text bold in HTML?",
         "options": ["<i>", "<b>", "<u>", "<p>"], "answer": 1},
    ],
    2: [
        {"id": "q2_1", "text": "What is the output of print(type([])) in Python?",
         "options": ["<class 'tuple'>", "<class 'list'>",
                     "<class 'dict'>", "<class 'set'>"], "answer": 1},
        {"id": "q2_2", "text": "Which HTTP method is used to update a resource?",
         "options": ["GET", "POST", "PUT", "DELETE"], "answer": 2},
        {"id": "q2_3", "text": "What does CSS stand for?",
         "options": ["Cascading Style Sheets", "Computer Style Sheets",
                     "Creative Style System", "Cascading System Sheets"], "answer": 0},
        {"id": "q2_4", "text": "Which keyword defines a function in Python?",
         "options": ["func", "define", "def", "function"], "answer": 2},
        {"id": "q2_5", "text": "What is a primary key in a database?",
         "options": ["A password", "A unique identifier for each row",
                     "A foreign table", "A column name"], "answer": 1},
    ],
    3: [
        {"id": "q3_1", "text": "What is the time complexity of binary search?",
         "options": ["O(n)", "O(log n)", "O(n^2)", "O(1)"], "answer": 1},
        {"id": "q3_2", "text": "Which SQL clause filters grouped results?",
         "options": ["WHERE", "HAVING", "GROUP BY", "ORDER BY"], "answer": 1},
        {"id": "q3_3", "text": "What design pattern ensures only one instance of a class?",
         "options": ["Factory", "Observer", "Singleton", "Strategy"], "answer": 2},
        {"id": "q3_4", "text": "Which data structure uses FIFO?",
         "options": ["Stack", "Queue", "Tree", "Graph"], "answer": 1},
        {"id": "q3_5", "text": "What does REST stand for?",
         "options": ["Representational State Transfer",
                     "Remote Execution Standard Technology",
                     "Resource State Transformation",
                     "Reliable Efficient System Transfer"], "answer": 0},
    ],
    4: [
        {"id": "q4_1", "text": "What is the space complexity of merge sort?",
         "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "answer": 2},
        {"id": "q4_2", "text": "Which protocol operates at the transport layer?",
         "options": ["HTTP", "TCP", "DNS", "FTP"], "answer": 1},
        {"id": "q4_3", "text": "What is a deadlock in operating systems?",
         "options": ["A crashed process", "Two processes waiting for each other",
                     "A memory leak", "A full disk"], "answer": 1},
        {"id": "q4_4", "text": "What is the purpose of an index in a database?",
         "options": ["Encrypt data", "Speed up queries",
                     "Compress tables", "Validate input"], "answer": 1},
        {"id": "q4_5", "text": "Which sorting algorithm is not comparison-based?",
         "options": ["Quick Sort", "Merge Sort", "Counting Sort", "Heap Sort"],
         "answer": 2},
    ],
    5: [
        {"id": "q5_1", "text": "What theorem says a distributed system cannot have C, A, and P simultaneously?",
         "options": ["CAP Theorem", "ACID Theorem",
                     "Byzantine Theorem", "Raft Theorem"], "answer": 0},
        {"id": "q5_2", "text": "What does the vanishing gradient problem affect in neural networks?",
         "options": ["Output layer only", "Deep layers during backpropagation",
                     "Data preprocessing", "Batch normalization"], "answer": 1},
        {"id": "q5_3", "text": "What is the purpose of a semaphore?",
         "options": ["Memory allocation", "Thread synchronization",
                     "Garbage collection", "Compilation"], "answer": 1},
        {"id": "q5_4", "text": "Which consistency model does DynamoDB use by default?",
         "options": ["Strong", "Eventual", "Causal", "Linearizable"], "answer": 1},
        {"id": "q5_5", "text": "What is the amortized time complexity of inserting into a dynamic array?",
         "options": ["O(n)", "O(1)", "O(log n)", "O(n^2)"], "answer": 1},
    ],
}


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
async def load_model():
    global model_data

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "skill_predictor.pkl")

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    model_data = joblib.load(model_path)
    logger.info(f"Model loaded: {model_data['model_name']} "
                f"(test acc {model_data['test_accuracy']:.4f})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_question(difficulty: int, used_ids: set) -> Optional[dict]:
    """Pick a random unused question at the target difficulty.
    Falls back to adjacent difficulties if none available."""
    difficulty = max(1, min(5, difficulty))

    for d in [difficulty, difficulty - 1, difficulty + 1, difficulty - 2, difficulty + 2]:
        if d < 1 or d > 5:
            continue
        available = [q for q in QUESTIONS.get(d, []) if q["id"] not in used_ids]
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
    first_q = pick_question(difficulty=3, used_ids=set())
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

    nq = pick_question(new_diff, session["used_ids"])
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
