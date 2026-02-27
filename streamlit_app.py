"""
Tamkeen Adaptive Skill Assessment - Streamlit App

Interactive adaptive quiz that adjusts difficulty based on answers,
then predicts skill level using a trained ML model.
"""

import os
import time
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tamkeen Skill Assessment",
    page_icon="",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "src", "models", "skill_predictor.pkl")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return joblib.load(model_path)


model_data = load_model()

# ---------------------------------------------------------------------------
# Question bank (same as the FastAPI backend)
# ---------------------------------------------------------------------------
QUESTIONS = {
    1: [
        {"id": "q1_1", "text": "What does HTML stand for?",
         "options": ["Hyper Text Markup Language", "High Tech Modern Language",
                     "Hyper Transfer Markup Language", "Home Tool Markup Language"],
         "answer": 0, "difficulty": 1},
        {"id": "q1_2", "text": "Which symbol starts a comment in Python?",
         "options": ["//", "#", "/*", "--"], "answer": 1, "difficulty": 1},
        {"id": "q1_3", "text": "What data type stores True or False?",
         "options": ["String", "Integer", "Boolean", "Float"], "answer": 2, "difficulty": 1},
        {"id": "q1_4", "text": "What does CPU stand for?",
         "options": ["Central Processing Unit", "Central Program Utility",
                     "Computer Personal Unit", "Central Processor Unifier"],
         "answer": 0, "difficulty": 1},
        {"id": "q1_5", "text": "Which tag makes text bold in HTML?",
         "options": ["<i>", "<b>", "<u>", "<p>"], "answer": 1, "difficulty": 1},
    ],
    2: [
        {"id": "q2_1", "text": "What is the output of print(type([])) in Python?",
         "options": ["<class 'tuple'>", "<class 'list'>",
                     "<class 'dict'>", "<class 'set'>"], "answer": 1, "difficulty": 2},
        {"id": "q2_2", "text": "Which HTTP method is used to update a resource?",
         "options": ["GET", "POST", "PUT", "DELETE"], "answer": 2, "difficulty": 2},
        {"id": "q2_3", "text": "What does CSS stand for?",
         "options": ["Cascading Style Sheets", "Computer Style Sheets",
                     "Creative Style System", "Cascading System Sheets"],
         "answer": 0, "difficulty": 2},
        {"id": "q2_4", "text": "Which keyword defines a function in Python?",
         "options": ["func", "define", "def", "function"], "answer": 2, "difficulty": 2},
        {"id": "q2_5", "text": "What is a primary key in a database?",
         "options": ["A password", "A unique identifier for each row",
                     "A foreign table", "A column name"], "answer": 1, "difficulty": 2},
    ],
    3: [
        {"id": "q3_1", "text": "What is the time complexity of binary search?",
         "options": ["O(n)", "O(log n)", "O(n^2)", "O(1)"], "answer": 1, "difficulty": 3},
        {"id": "q3_2", "text": "Which SQL clause filters grouped results?",
         "options": ["WHERE", "HAVING", "GROUP BY", "ORDER BY"], "answer": 1, "difficulty": 3},
        {"id": "q3_3", "text": "What design pattern ensures only one instance of a class?",
         "options": ["Factory", "Observer", "Singleton", "Strategy"],
         "answer": 2, "difficulty": 3},
        {"id": "q3_4", "text": "Which data structure uses FIFO?",
         "options": ["Stack", "Queue", "Tree", "Graph"], "answer": 1, "difficulty": 3},
        {"id": "q3_5", "text": "What does REST stand for?",
         "options": ["Representational State Transfer",
                     "Remote Execution Standard Technology",
                     "Resource State Transformation",
                     "Reliable Efficient System Transfer"],
         "answer": 0, "difficulty": 3},
    ],
    4: [
        {"id": "q4_1", "text": "What is the space complexity of merge sort?",
         "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "answer": 2, "difficulty": 4},
        {"id": "q4_2", "text": "Which protocol operates at the transport layer?",
         "options": ["HTTP", "TCP", "DNS", "FTP"], "answer": 1, "difficulty": 4},
        {"id": "q4_3", "text": "What is a deadlock in operating systems?",
         "options": ["A crashed process", "Two processes waiting for each other",
                     "A memory leak", "A full disk"], "answer": 1, "difficulty": 4},
        {"id": "q4_4", "text": "What is the purpose of an index in a database?",
         "options": ["Encrypt data", "Speed up queries",
                     "Compress tables", "Validate input"], "answer": 1, "difficulty": 4},
        {"id": "q4_5", "text": "Which sorting algorithm is not comparison-based?",
         "options": ["Quick Sort", "Merge Sort", "Counting Sort", "Heap Sort"],
         "answer": 2, "difficulty": 4},
    ],
    5: [
        {"id": "q5_1", "text": "What theorem says a distributed system cannot have C, A, and P simultaneously?",
         "options": ["CAP Theorem", "ACID Theorem",
                     "Byzantine Theorem", "Raft Theorem"], "answer": 0, "difficulty": 5},
        {"id": "q5_2", "text": "What does the vanishing gradient problem affect in neural networks?",
         "options": ["Output layer only", "Deep layers during backpropagation",
                     "Data preprocessing", "Batch normalization"], "answer": 1, "difficulty": 5},
        {"id": "q5_3", "text": "What is the purpose of a semaphore?",
         "options": ["Memory allocation", "Thread synchronization",
                     "Garbage collection", "Compilation"], "answer": 1, "difficulty": 5},
        {"id": "q5_4", "text": "Which consistency model does DynamoDB use by default?",
         "options": ["Strong", "Eventual", "Causal", "Linearizable"], "answer": 1, "difficulty": 5},
        {"id": "q5_5", "text": "What is the amortized time complexity of inserting into a dynamic array?",
         "options": ["O(n)", "O(1)", "O(log n)", "O(n^2)"], "answer": 1, "difficulty": 5},
    ],
}

MAX_QUESTIONS = 10
DIFFICULTY_LABELS = {1: "Easy", 2: "Below Average", 3: "Medium", 4: "Hard", 5: "Expert"}


def pick_question(difficulty, used_ids):
    difficulty = max(1, min(5, difficulty))
    for d in [difficulty, difficulty - 1, difficulty + 1, difficulty - 2, difficulty + 2]:
        if d < 1 or d > 5:
            continue
        available = [q for q in QUESTIONS.get(d, []) if q["id"] not in used_ids]
        if available:
            return random.choice(available)
    return None


def next_difficulty(current, is_correct):
    if is_correct:
        return min(5, current + 1)
    return max(1, current - 1)


def predict_skill(answers, disability_type):
    """Build 12 features and run model prediction."""
    difficulties = [a["difficulty"] for a in answers]
    correct_flags = [1 if a["is_correct"] else 0 for a in answers]
    times = [a["response_time"] for a in answers]

    disability_map = model_data["disability_map"]
    label_map = model_data["label_map"]
    features = model_data["features"]
    reverse_label = {v: k for k, v in label_map.items()}

    num_questions = len(answers)
    avg_difficulty = float(np.mean(difficulties))
    max_difficulty = max(difficulties)
    accuracy = float(np.mean(correct_flags))
    total_correct = sum(correct_flags)
    avg_response_time = float(np.mean(times))
    std_response_time = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    max_response_time = float(max(times))
    min_response_time = float(min(times))
    time_range = max_response_time - min_response_time
    time_per_difficulty = avg_response_time / avg_difficulty if avg_difficulty > 0 else 0.0
    disability_encoded = disability_map.get(disability_type, disability_map.get("none", 0))

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
    row = row[features]

    model = model_data["model"]
    prediction = int(model.predict(row)[0])
    probas = model.predict_proba(row)[0]
    skill_level = reverse_label.get(prediction, "Unknown")
    confidence = float(probas[prediction])

    return {
        "skill_level": skill_level,
        "confidence": confidence,
        "probabilities": {reverse_label[i]: float(p) for i, p in enumerate(probas)},
        "accuracy": accuracy,
        "total_correct": total_correct,
        "avg_difficulty": avg_difficulty,
        "max_difficulty": max_difficulty,
        "avg_response_time": avg_response_time,
    }



def init_session():
    if "stage" not in st.session_state:
        st.session_state.stage = "setup"        # setup -> quiz -> result
        st.session_state.answers = []
        st.session_state.difficulty_path = []
        st.session_state.current_difficulty = 3
        st.session_state.current_question = None
        st.session_state.used_ids = set()
        st.session_state.disability = "none"
        st.session_state.question_start_time = None
        st.session_state.question_number = 0


init_session()



# CSS

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .difficulty-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .diff-1 { background: #d4edda; color: #155724; }
    .diff-2 { background: #cce5ff; color: #004085; }
    .diff-3 { background: #fff3cd; color: #856404; }
    .diff-4 { background: #f8d7da; color: #721c24; }
    .diff-5 { background: #d6d8db; color: #1b1e21; }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card h1 { color: white; margin: 0; font-size: 2.5rem; }
    .result-card p { color: rgba(255,255,255,0.9); font-size: 1.1rem; }
    .metric-row {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)



# STAGE 1: Setup

if st.session_state.stage == "setup":

    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Tamkeen Skill Assessment")
    st.markdown("**Adaptive quiz that adjusts to your level using AI**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        disability = st.selectbox(
            "Accessibility Profile",
            options=["none", "visual", "motor", "cognitive", "hearing"],
            format_func=lambda x: {
                "none": "No disability",
                "visual": "Visual impairment",
                "motor": "Motor impairment",
                "cognitive": "Cognitive impairment",
                "hearing": "Hearing impairment",
            }[x],
        )
    with col2:
        st.markdown("")
        st.markdown("")
        st.info(f"Questions: **{MAX_QUESTIONS}** | Start difficulty: **Medium (3)**")

    st.divider()

    st.markdown("**How it works:**")
    st.markdown(
        "1. Answer 10 multiple-choice questions\n"
        "2. Difficulty adapts: correct answers raise it, wrong answers lower it\n"
        "3. A trained ML model predicts your skill level based on your performance"
    )

    if st.button("Start Assessment", type="primary", use_container_width=True):
        st.session_state.disability = disability
        st.session_state.current_difficulty = 3
        st.session_state.difficulty_path = [3]
        st.session_state.answers = []
        st.session_state.used_ids = set()
        st.session_state.question_number = 0

        # Pick first question
        q = pick_question(3, set())
        st.session_state.current_question = q
        st.session_state.used_ids.add(q["id"])
        st.session_state.question_start_time = time.time()
        st.session_state.question_number = 1
        st.session_state.stage = "quiz"
        st.rerun()



# STAGE 2: Quiz

elif st.session_state.stage == "quiz":
    q = st.session_state.current_question
    qnum = st.session_state.question_number
    diff = q["difficulty"]

    # Header
    col_a, col_b, col_c = st.columns([2, 3, 2])
    with col_a:
        st.markdown(f"**Question {qnum}/{MAX_QUESTIONS}**")
    with col_b:
        diff_label = DIFFICULTY_LABELS.get(diff, "?")
        st.markdown(
            f'<span class="difficulty-badge diff-{diff}">Difficulty: {diff} - {diff_label}</span>',
            unsafe_allow_html=True,
        )
    with col_c:
        st.progress(qnum / MAX_QUESTIONS)

    st.divider()

    # Question text
    st.subheader(q["text"])

    # Options as radio buttons
    selected = st.radio(
        "Select your answer:",
        options=list(range(len(q["options"]))),
        format_func=lambda i: q["options"][i],
        key=f"answer_{q['id']}",
        label_visibility="collapsed",
    )

    if st.button("Submit Answer", type="primary", use_container_width=True):
        response_time = round(time.time() - st.session_state.question_start_time, 2)
        response_time = max(0.5, response_time)  # floor at 0.5s

        is_correct = selected == q["answer"]

        st.session_state.answers.append({
            "question_id": q["id"],
            "difficulty": diff,
            "is_correct": is_correct,
            "response_time": response_time,
        })

        answered = len(st.session_state.answers)

        if answered >= MAX_QUESTIONS:
            # Done - go to results
            st.session_state.stage = "result"
            st.rerun()
        else:
            # Adaptive next question
            new_diff = next_difficulty(diff, is_correct)
            st.session_state.current_difficulty = new_diff
            st.session_state.difficulty_path.append(new_diff)

            nq = pick_question(new_diff, st.session_state.used_ids)
            if nq is None:
                st.session_state.stage = "result"
                st.rerun()
            else:
                st.session_state.current_question = nq
                st.session_state.used_ids.add(nq["id"])
                st.session_state.question_start_time = time.time()
                st.session_state.question_number = answered + 1
                st.rerun()


# STAGE 3: Results

elif st.session_state.stage == "result":
    answers = st.session_state.answers
    disability = st.session_state.disability
    result = predict_skill(answers, disability)

    # Skill level header
    skill = result["skill_level"]
    confidence = result["confidence"]

    level_emoji = {"Beginner": "🌱", "Intermediate": "⚡", "Advanced": "🏆"}.get(skill, "📊")
    level_color = {"Beginner": "#e74c3c", "Intermediate": "#f39c12", "Advanced": "#27ae60"}.get(skill, "#3498db")

    st.markdown(
        f"""
        <div class="result-card">
            <p style="font-size:2.5rem; margin:0;">{level_emoji}</p>
            <h1>{skill}</h1>
            <p>Model confidence: {confidence:.1%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Correct", f"{result['total_correct']}/{len(answers)}")
    col2.metric("Accuracy", f"{result['accuracy']:.0%}")
    col3.metric("Avg Difficulty", f"{result['avg_difficulty']:.1f}/5")
    col4.metric("Avg Time", f"{result['avg_response_time']:.1f}s")

    st.divider()

    # Probability breakdown
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Skill Level": list(result["probabilities"].keys()),
        "Probability": list(result["probabilities"].values()),
    })
    prob_df = prob_df.sort_values("Skill Level")

    colors = {"Beginner": "#e74c3c", "Intermediate": "#f39c12", "Advanced": "#27ae60"}
    fig_prob, ax_prob = plt.subplots(figsize=(6, 2.5))
    bars = ax_prob.barh(
        prob_df["Skill Level"],
        prob_df["Probability"],
        color=[colors.get(s, "#3498db") for s in prob_df["Skill Level"]],
        height=0.5,
    )
    ax_prob.set_xlim(0, 1)
    ax_prob.set_xlabel("Probability")
    for bar, val in zip(bars, prob_df["Probability"]):
        ax_prob.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1%}", va="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_prob)
    plt.close()

    # Difficulty path chart
    st.subheader("Adaptive Difficulty Path")
    diff_path = st.session_state.difficulty_path
    fig_path, ax_path = plt.subplots(figsize=(8, 3))
    xs = list(range(1, len(diff_path) + 1))
    ax_path.plot(xs, diff_path, marker="o", color="#667eea", linewidth=2, markersize=8)
    ax_path.fill_between(xs, diff_path, alpha=0.15, color="#667eea")
    ax_path.set_xlabel("Question Number")
    ax_path.set_ylabel("Difficulty Level")
    ax_path.set_yticks([1, 2, 3, 4, 5])
    ax_path.set_yticklabels(["1 (Easy)", "2", "3 (Medium)", "4", "5 (Expert)"])
    ax_path.set_xticks(xs)
    ax_path.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_path)
    plt.close()

    # Answer details
    st.subheader("Answer Details")
    detail_data = []
    for i, a in enumerate(answers, 1):
        detail_data.append({
            "Q#": i,
            "Difficulty": a["difficulty"],
            "Correct": "Yes" if a["is_correct"] else "No",
            "Time (s)": round(a["response_time"], 1),
        })
    st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

    st.divider()

    # Restart button
    if st.button("Take Assessment Again", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Sidebar info
    with st.sidebar:
        st.markdown("### Model Info")
        st.markdown(f"**Model:** {model_data['model_name']}")
        st.markdown(f"**Test Accuracy:** {model_data['test_accuracy']:.2%}")
        st.markdown(f"**CV Score:** {model_data['cv_score']:.4f}")
        st.markdown(f"**Features:** {len(model_data['features'])}")
        st.markdown(f"**Disability:** {disability}")
