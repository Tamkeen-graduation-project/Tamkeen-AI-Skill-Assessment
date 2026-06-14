"""
Tamkeen Adaptive Skill Assessment - Streamlit App

Interactive adaptive quiz that adjusts difficulty based on answers,
then predicts skill level using a trained ML model.
Enhanced for premium UI/UX demo with accessibility accommodations.
"""

import os
import time
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tamkeen Skill Assessment",
    page_icon="🤖",
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
# Question bank loader
# Loads from data/raw/questions_v2.csv
# ---------------------------------------------------------------------------
@st.cache_data
def load_questions_from_csv():
    csv_path = "data/raw/questions_v2.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/raw/questions.csv"
    
    if not os.path.exists(csv_path):
        return {"default": {1: [{"id": "fallback", "text": "What is 1+1?", "options": ["1","2","3","4"], "answer": 1, "difficulty": 1}]}}

    try:
        df = pd.read_csv(csv_path)
        questions_bank = {}
        for _, row in df.iterrows():
            a_id = str(row['assessment_id'])
            diff = int(row['difficulty_level'])
            
            if a_id not in questions_bank:
                questions_bank[a_id] = {}
            if diff not in questions_bank[a_id]:
                questions_bank[a_id][diff] = []
            
            options = str(row['options']).split('|')
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'True': 0, 'False': 1}
            ans_idx = answer_map.get(str(row['correct_answer']), 0)

            questions_bank[a_id][diff].append({
                "id": str(row['question_id']),
                "text": str(row['question_text']),
                "options": options,
                "answer": ans_idx,
                "difficulty": diff
            })
        return questions_bank
    except Exception as e:
        st.error(f"Error loading questions: {e}")
        return {}


QUESTIONS_BANK = load_questions_from_csv()

MAX_QUESTIONS = 10
DIFFICULTY_LABELS = {1: "Easy", 2: "Below Average", 3: "Medium", 4: "Hard", 5: "Expert"}
DISABILITY_MULTIPLIERS = {
    "visual": 3.5,
    "motor": 2.5,
    "cognitive": 1.8,
    "hearing": 1.3,
    "none": 1.0,
}

# ---------------------------------------------------------------------------
# Cognitive plain language mapping
# ---------------------------------------------------------------------------
COGNITIVE_SIMPLIFIED_QUESTIONS = {
    "When does the marketing process traditionally begin?": "When does marketing start?",
    "What is the core definition of Digital Marketing according to Tamkeen Content?": "What is the main idea of digital marketing?",
    "Which stage of marketing focuses on turning the audience into customers?": "How do we get visitors to buy?",
    "Digital Marketing proficiency requirements vary based on what?": "What changes the skills marketing people need?",
    "What is the primary characteristic of a Scalar object in Python?": "What is a basic single value in Python?",
    "What does the int() function do when given a float like 3.9?": "What does int(3.9) do in Python?",
    "Which of these is a Non-Scalar object in Python?": "Which object can hold multiple values?",
    "What is returned by type(None) in Python?": "What is the type of None in Python?",
    "What is the primary role of a 'device driver' in an embedded system?": "What does a device driver do?",
    "Why can't an application interact with hardware directly?": "Why do programs need a driver to talk to hardware?",
    "Which of the following was the first category of devices connected to the Internet?": "What was connected to the internet first?",
    "In the context of IoT, what is a 'server'?": "What is a server in IoT?",
    "Which category of IoT devices includes water and electricity meters?": "What category are water and electricity meters in?",
    "What is the main goal of Information Architecture in UX?": "What is the main goal of information structure in UX design?"
}

def get_simplified_question(question_text: str) -> str:
    for k, v in COGNITIVE_SIMPLIFIED_QUESTIONS.items():
        if k.lower() in question_text.lower() or question_text.lower() in k.lower():
            return v
    return question_text

# ---------------------------------------------------------------------------
# Adaptive engine helper functions
# ---------------------------------------------------------------------------
def pick_question(difficulty: int, used_ids: set, assessment_id: str = "default") -> Optional[dict]:
    """Pick a random unused question at the target difficulty for the given assessment."""
    difficulty = max(1, min(5, difficulty))
    
    # Bug fix: course names are digital_marketing, etc. questions_v2.csv assessment ids are assessment_digital_marketing, etc.
    if not assessment_id.startswith("assessment_"):
        assessment_id = f"assessment_{assessment_id}"

    if assessment_id not in QUESTIONS_BANK:
        if "assessment_0" in QUESTIONS_BANK:
            assessment_id = "assessment_0"
        elif QUESTIONS_BANK:
            assessment_id = list(QUESTIONS_BANK.keys())[0]
        else:
            return None

    bank = QUESTIONS_BANK[assessment_id]

    for d in [difficulty, difficulty - 1, difficulty + 1, difficulty - 2, difficulty + 2]:
        if d < 1 or d > 5:
            continue
        available = [q for q in bank.get(d, []) if q["id"] not in used_ids]
        if available:
            return random.choice(available)
    return None


def next_difficulty(current: int, is_correct: bool) -> int:
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
        st.session_state.student_name = "Demo User"
        st.session_state.selected_course = None
        st.session_state.selected_accessibility = "none"
        st.session_state.session_id = f"sess_{random.randint(100000, 999999)}"


init_session()

# ---------------------------------------------------------------------------
# Dynamic CSS Injection based on Accessibility profile
# ---------------------------------------------------------------------------
def inject_custom_css(disability: str):
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        /* Hide Streamlit header, deploy button, menu, and footer for stand-alone UI recording */
        div[data-testid="stHeader"] {
            display: none !important;
        }
        #MainMenu {
            visibility: hidden !important;
        }
        footer {
            visibility: hidden !important;
        }

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #0b0f19;
            color: #f1f5f9;
        }
        
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .main-header {
            text-align: center;
            padding: 1.5rem 0;
            background: linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(168,85,247,0.06) 100%);
            border-radius: 16px;
            margin-bottom: 2rem;
            border: 1px solid rgba(99, 102, 241, 0.15);
        }
        
        .glass-card {
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.12);
        }
        
        .active-card {
            background: rgba(99, 102, 241, 0.15) !important;
            border: 2px solid #6366f1 !important;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.25) !important;
        }
        
        .course-title {
            font-weight: 700;
            font-size: 1.1rem;
            color: #ffffff;
            margin-top: 0.25rem;
        }
        
        .course-desc {
            font-size: 0.825rem;
            color: #94a3b8;
            margin-top: 0.5rem;
            line-height: 1.4;
        }
        
        .access-card {
            border-left: 5px solid #6366f1;
        }
        .access-none { border-left-color: #6366f1; }
        .access-visual { border-left-color: #fbbf24; }
        .access-motor { border-left-color: #ec4899; }
        .access-cognitive { border-left-color: #3b82f6; }
        .access-hearing { border-left-color: #10b981; }
        
        .hud-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1.25rem;
            background: rgba(30, 41, 59, 0.4);
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.9rem;
        }
        
        .difficulty-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .diff-1 { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
        .diff-2 { background: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
        .diff-3 { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
        .diff-4 { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
        .diff-5 { background: rgba(168, 85, 247, 0.15); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); }
        
        .stButton>button {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.8rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2) !important;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35) !important;
        }
        
        .stButton>button:active {
            transform: translateY(0px) !important;
        }
        
        div[data-testid="stRadio"] label {
            background: rgba(17, 24, 39, 0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            padding: 0.9rem 1.2rem !important;
            margin-bottom: 0.6rem !important;
            width: 100%;
            transition: all 0.2s ease;
            cursor: pointer;
            color: #f1f5f9 !important;
            font-size: 0.95rem;
        }
        
        div[data-testid="stRadio"] label:hover {
            background: rgba(99, 102, 241, 0.06) !important;
            border-color: rgba(99, 102, 241, 0.25) !important;
        }
        
        div[data-testid="stRadio"] [data-checked="true"] {
            border-color: #6366f1 !important;
            background: rgba(99, 102, 241, 0.12) !important;
            font-weight: 600;
        }
        
        .result-card {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #c084fc 100%);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .result-card::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 80%);
            pointer-events: none;
        }
        
        .result-card h1 {
            color: white !important;
            margin: 0.5rem 0 0 0 !important;
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            text-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }
        
        .result-card p {
            color: rgba(255,255,255,0.9) !important;
            font-size: 1.1rem;
        }

        .metric-card {
            background: rgba(17, 24, 39, 0.6);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
    </style>
    """
    
    if disability == "visual":
        css += """
        <style>
            html, body, [data-testid="stAppViewContainer"], .stApp {
                background-color: #000000 !important;
                color: #ffffff !important;
                font-size: 1.2rem !important;
            }
            .glass-card, .hud-container, .metric-card, div[data-testid="stRadio"] label {
                background-color: #000000 !important;
                border: 3px solid #fbbf24 !important;
                color: #ffffff !important;
                border-radius: 0px !important;
                box-shadow: none !important;
            }
            div[data-testid="stRadio"] label:hover {
                background-color: #1c1917 !important;
            }
            .active-card, div[data-testid="stRadio"] [data-checked="true"] {
                background-color: #fbbf24 !important;
                color: #000000 !important;
                border: 3px solid #ffffff !important;
            }
            div[data-testid="stRadio"] [data-checked="true"] * {
                color: #000000 !important;
            }
            .course-title, .course-desc {
                color: #ffffff !important;
            }
            .stButton>button {
                background: #fbbf24 !important;
                color: #000000 !important;
                border: 3px solid #ffffff !important;
                font-weight: 800 !important;
                font-size: 1.25rem !important;
                border-radius: 0px !important;
                box-shadow: none !important;
            }
            .stButton>button:hover {
                background: #fef08a !important;
                transform: none !important;
            }
            .difficulty-badge {
                border: 2px solid #ffffff !important;
                background-color: #000000 !important;
                color: #ffffff !important;
                border-radius: 0px !important;
            }
            .stProgress > div > div > div > div {
                background-image: none !important;
                background-color: #fbbf24 !important;
            }
        </style>
        """
    elif disability == "motor":
        css += """
        <style>
            div[data-testid="stRadio"] label {
                padding: 1.5rem 2rem !important;
                margin-bottom: 0.85rem !important;
                font-size: 1.1rem !important;
            }
            .stButton>button {
                padding: 1.1rem 2.2rem !important;
                font-size: 1.1rem !important;
            }
            .glass-card {
                padding: 2.2rem !important;
                margin-bottom: 2rem !important;
            }
        </style>
        """
    elif disability == "cognitive":
        css += """
        <style>
            .glass-card {
                border-left-width: 8px !important;
            }
            body {
                line-height: 1.7;
            }
        </style>
        """
        
    st.markdown(css, unsafe_allow_html=True)


inject_custom_css(st.session_state.get("selected_accessibility", "none"))

# ---------------------------------------------------------------------------
# STAGE 1: Setup Assessment
# ---------------------------------------------------------------------------
if st.session_state.stage == "setup":
    # Graphic Header Banner
    if os.path.exists("tamkeen_banner.png"):
        st.image("tamkeen_banner.png", use_container_width=True)

    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Tamkeen Skill Assessment")
    st.markdown("**Adaptive AI quiz that maps skills and accounts for accessibility**")
    st.markdown('</div>', unsafe_allow_html=True)

    # Student name input
    st.markdown("### 1. Tell us about yourself")
    student_name = st.text_input(
        "Enter Student Name", 
        value=st.session_state.student_name, 
        placeholder="e.g. Alex Johnson",
        label_visibility="collapsed"
    )
    st.session_state.student_name = student_name if student_name.strip() else "Demo User"

    st.divider()

    # Load courses for description metadata
    try:
        courses_df = pd.read_csv('data/raw/courses.csv')
        course_options = {row['title']: row['course_id'] for _, row in courses_df.iterrows()}
    except:
        course_options = {
            "Digital Marketing": "digital_marketing",
            "Python Programming": "python",
            "Embedded Systems C": "embedded",
            "Internet of Things": "iot",
            "User Experience Design": "ux",
            "Software Testing": "testing",
            "Freelancing with Khamsat": "freelancing_khamsat",
        }

    course_details = {
        "digital_marketing": {"icon": "📣", "desc": "Digital marketing science, strategies, and search engine optimizations."},
        "python": {"icon": "🐍", "desc": "Scalar and non-scalar variables, syntax, lists, functions, and control flows."},
        "embedded": {"icon": "🔌", "desc": "Hardware registers, assembly mappings, interrupt handlers, and C code translations."},
        "iot": {"icon": "🌐", "desc": "Actuators, internet network connections, sensors, and hardware topologies."},
        "ux": {"icon": "🎨", "desc": "User architectures, wireframes, styling metrics, and interface structures."},
        "testing": {"icon": "🧪", "desc": "Manual validation, coverage metrics, automated tests, and bug verification processes."},
        "freelancing_khamsat": {"icon": "💼", "desc": "Commercial services, marketplace catalogs, profile setup, and freelance operations."}
    }

    # Course selection cards
    st.markdown("### 2. Select Course Assessment")
    
    course_list = list(course_options.items())
    cols = st.columns(3)
    for idx, (title, c_id) in enumerate(course_list):
        with cols[idx % 3]:
            details = course_details.get(c_id, {"icon": "📚", "desc": f"Official Tamkeen assessment on {title}."})
            is_selected = st.session_state.selected_course == c_id
            card_class = "glass-card active-card" if is_selected else "glass-card"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">{details['icon']}</div>
                <div class="course-title">{title}</div>
                <div class="course-desc">{details['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Choose {title}", key=f"select_course_{c_id}", use_container_width=True):
                st.session_state.selected_course = c_id
                st.session_state.course_id = c_id
                st.session_state.course_title = title
                st.rerun()

    st.divider()

    # Accessibility Selection cards
    st.markdown("### 3. Choose Accessibility Profile")
    
    profiles = [
        {"id": "none", "title": "No Disability", "icon": "👤", "desc": "Standard sizing, layout themes, and time measurements.", "class": "access-none"},
        {"id": "visual", "title": "Visual Impairment", "icon": "👁️", "desc": "High-contrast layout, bold yellow components, and larger fonts.", "class": "access-visual"},
        {"id": "motor", "title": "Motor Impairment", "icon": "🖐️", "desc": "Enlarged click options and buttons to facilitate coordination.", "class": "access-motor"},
        {"id": "cognitive", "title": "Cognitive Support", "icon": "🧠", "desc": "Simplification toggle for question phrasing and helper text.", "class": "access-cognitive"},
        {"id": "hearing", "title": "Hearing Impairment", "icon": "👂", "desc": "Entirely text-based signals, logs, and notification boxes.", "class": "access-hearing"}
    ]

    cols_acc = st.columns(5)
    for idx, prof in enumerate(profiles):
        with cols_acc[idx % 5]:
            is_selected = st.session_state.selected_accessibility == prof["id"]
            card_class = f"glass-card access-card {prof['class']} active-card" if is_selected else f"glass-card access-card {prof['class']}"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size: 1.8rem; margin-bottom: 0.4rem;">{prof['icon']}</div>
                <div class="course-title" style="font-size: 0.95rem;">{prof['title']}</div>
                <p style="font-size: 0.725rem; color: #94a3b8; line-height: 1.3; margin-top: 0.4rem;">{prof['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Apply {prof['title']}", key=f"select_acc_{prof['id']}", use_container_width=True):
                st.session_state.selected_accessibility = prof["id"]
                st.session_state.disability = prof["id"]
                st.rerun()

    st.divider()

    # Setup warnings or starts
    if st.session_state.selected_course is None:
        st.warning("⚠️ Please select a course assessment from the cards above to proceed.")
    else:
        if st.button("🚀 Start Assessment Session", type="primary", use_container_width=True):
            st.session_state.current_difficulty = 3
            st.session_state.difficulty_path = [3]
            st.session_state.answers = []
            st.session_state.used_ids = set()
            st.session_state.question_number = 0

            # Pick first question
            q = pick_question(3, set(), st.session_state.course_id)
            if q:
                st.session_state.current_question = q
                st.session_state.used_ids.add(q["id"])
                st.session_state.question_start_time = time.time()
                st.session_state.question_number = 1
                st.session_state.stage = "quiz"
                st.rerun()
            else:
                st.error("No questions found for this course!")

# ---------------------------------------------------------------------------
# STAGE 2: Quiz Assessment
# ---------------------------------------------------------------------------
elif st.session_state.stage == "quiz":
    q = st.session_state.current_question
    qnum = st.session_state.question_number
    diff = q["difficulty"]

    # HUD row: Student & course data, dynamic Javascript response timer
    student = st.session_state.student_name
    course = st.session_state.course_title
    
    # HTML + JS live response timer (updates independently on screen without rerunning Streamlit)
    timer_html = """
    <div style="font-family: 'Plus Jakarta Sans', sans-serif; background: rgba(30, 41, 59, 0.4); border-radius: 12px; padding: 0.75rem 1.25rem; border: 1px solid rgba(255,255,255,0.05); color: #f1f5f9; display: flex; justify-content: space-between; align-items: center; font-size: 0.95rem;">
        <div>👤 <b>Student:</b> <span style="color:#6366f1;">""" + student + """</span></div>
        <div style="margin: 0 15px;">📚 <b>Assessment:</b> <span style="color:#a855f7;">""" + course + """</span></div>
        <div>⏱️ <b>Live Timer:</b> <span id="timer" style="font-weight: 700; color: #fbbf24; font-size:1.1rem;">0.0s</span></div>
    </div>
    <script>
        var start = Date.now();
        setInterval(function() {
            var elapsed = ((Date.now() - start) / 1000).toFixed(1);
            document.getElementById("timer").innerText = elapsed + "s";
        }, 100);
    </script>
    """
    components.html(timer_html, height=48)

    # Progress bar and difficulties
    col_a, col_b = st.columns([7, 3])
    with col_a:
        diff_label = DIFFICULTY_LABELS.get(diff, "?")
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
            <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">Question {qnum} of {MAX_QUESTIONS}</span>
            <span class="difficulty-badge diff-{diff}">Level {diff}: {diff_label}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.progress(qnum / MAX_QUESTIONS)

    st.divider()

    # Cognitive accommodation - simplify question helper
    question_text = q["text"]
    if st.session_state.disability == "cognitive":
        st.markdown("""
        <div style="background: rgba(59, 130, 246, 0.12); padding: 0.75rem 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 1rem; font-size: 0.85rem; color:#93c5fd;">
            🧠 <b>Cognitive Accommodation:</b> Plain language option is available to simplify terms.
        </div>
        """, unsafe_allow_html=True)
        plain_lang = st.toggle("Enable Plain Language (Simplified wording)", value=True)
        if plain_lang:
            question_text = get_simplified_question(q["text"])

    # Highlight Question
    st.markdown(f"""
    <div class="glass-card" style="border-left: 5px solid #6366f1; margin-top: 1rem;">
        <h3 style="margin: 0; font-weight: 600; line-height: 1.45; color: #ffffff; font-size: 1.25rem;">{question_text}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Styled Radio choices
    selected = st.radio(
        "Select your answer:",
        options=list(range(len(q["options"]))),
        format_func=lambda i: q["options"][i],
        key=f"answer_{q['id']}",
        label_visibility="collapsed",
    )

    st.divider()

    # Submit button
    if st.button("Submit Answer ➔", type="primary", use_container_width=True):
        response_time = round(time.time() - st.session_state.question_start_time, 2)
        response_time = max(0.5, response_time)  # floor at 0.5s

        is_correct = selected == q["answer"]

        st.session_state.answers.append({
            "question_id": q["id"],
            "question_text": q["text"],
            "difficulty": diff,
            "is_correct": is_correct,
            "response_time": response_time,
            "selected_option": q["options"][selected],
            "correct_option": q["options"][q["answer"]],
        })

        answered = len(st.session_state.answers)

        if answered >= MAX_QUESTIONS:
            st.session_state.stage = "result"
            st.rerun()
        else:
            new_diff = next_difficulty(diff, is_correct)
            st.session_state.current_difficulty = new_diff
            st.session_state.difficulty_path.append(new_diff)

            nq = pick_question(new_diff, st.session_state.used_ids, st.session_state.course_id)
            if nq is None:
                st.session_state.stage = "result"
                st.rerun()
            else:
                st.session_state.current_question = nq
                st.session_state.used_ids.add(nq["id"])
                st.session_state.question_start_time = time.time()
                st.session_state.question_number = answered + 1
                st.rerun()

# ---------------------------------------------------------------------------
# STAGE 3: Assessment Results & Model Prediction
# ---------------------------------------------------------------------------
elif st.session_state.stage == "result":
    answers = st.session_state.answers
    disability = st.session_state.disability
    result = predict_skill(answers, disability)

    skill = result["skill_level"]
    confidence = result["confidence"]

    level_emoji = {"Beginner": "🌱", "Intermediate": "⚡", "Advanced": "🏆"}.get(skill, "📊")
    
    # Success card
    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size:3.5rem; margin:0;">{level_emoji}</div>
            <p style="margin: 5px 0 0 0; text-transform: uppercase; letter-spacing:0.15em; font-weight:700; color:rgba(255,255,255,0.75);">Predicted Skill Level</p>
            <h1>{skill}</h1>
            <p style="margin-top:10px; font-weight:500;">Model Confidence: <span style="font-weight:700; color:#fbbf24;">{confidence:.1%}</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#94a3b8; font-size:0.75rem; margin:0; text-transform:uppercase;">CORRECT</p>
            <span style="font-size:1.6rem; font-weight:700; color:#10b981;">{result['total_correct']}/{len(answers)}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#94a3b8; font-size:0.75rem; margin:0; text-transform:uppercase;">ACCURACY</p>
            <span style="font-size:1.6rem; font-weight:700; color:#3b82f6;">{result['accuracy']:.0%}</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#94a3b8; font-size:0.75rem; margin:0; text-transform:uppercase;">AVG DIFFICULTY</p>
            <span style="font-size:1.6rem; font-weight:700; color:#f59e0b;">{result['avg_difficulty']:.1f} / 5</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#94a3b8; font-size:0.75rem; margin:0; text-transform:uppercase;">AVG TIMING</p>
            <span style="font-size:1.6rem; font-weight:700; color:#ec4899;">{result['avg_response_time']:.1f}s</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Charts section
    st.subheader("📊 Performance Analytics")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**Prediction Probabilities**")
        # Build prob df
        prob_df = pd.DataFrame({
            "Skill Level": list(result["probabilities"].keys()),
            "Probability": list(result["probabilities"].values()),
        })
        prob_df = prob_df.sort_values("Skill Level")

        # Styling variables based on accessibility/theme
        is_visual = disability == "visual"
        bg_color = "#000000" if is_visual else "#111827"
        text_color = "#ffffff" if is_visual else "#f1f5f9"
        grid_color = "#374151" if is_visual else "#ffffff"
        grid_alpha = 0.3 if is_visual else 0.08

        colors = {
            "Beginner": "#ffff00" if is_visual else "#ef4444",
            "Intermediate": "#ffffff" if is_visual else "#f59e0b",
            "Advanced": "#00ff00" if is_visual else "#10b981"
        }

        fig_prob, ax_prob = plt.subplots(figsize=(6, 3), dpi=150)
        fig_prob.patch.set_facecolor(bg_color)
        ax_prob.set_facecolor(bg_color)

        bar_colors = [colors.get(s, "#6366f1") for s in prob_df["Skill Level"]]
        bars = ax_prob.barh(
            prob_df["Skill Level"],
            prob_df["Probability"],
            color=bar_colors,
            height=0.45,
            edgecolor=text_color if is_visual else "none",
            linewidth=1.5 if is_visual else 0
        )

        ax_prob.set_xlim(0, 1.05)
        ax_prob.set_xlabel("Probability", color=text_color, fontsize=9, labelpad=5)
        ax_prob.tick_params(colors=text_color, labelsize=8)

        for spine in ["top", "right", "left", "bottom"]:
            ax_prob.spines[spine].set_color("none")

        ax_prob.grid(True, axis="x", color=grid_color, linestyle="--", linewidth=0.5, alpha=grid_alpha)

        for bar, val in zip(bars, prob_df["Probability"]):
            ax_prob.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center",
                ha="left",
                color=text_color,
                fontsize=8,
                fontweight="bold"
            )

        plt.tight_layout()
        st.pyplot(fig_prob)
        plt.close()

    with col_chart2:
        st.markdown("**Adaptive Difficulty Path**")
        diff_path = st.session_state.difficulty_path
        
        fig_path, ax_path = plt.subplots(figsize=(6, 3), dpi=150)
        fig_path.patch.set_facecolor(bg_color)
        ax_path.set_facecolor(bg_color)

        xs = list(range(1, len(diff_path) + 1))
        line_color = "#facc15" if is_visual else "#6366f1"

        ax_path.plot(xs, diff_path, marker="o", color=line_color, linewidth=2, markersize=6, markerfacecolor=line_color)
        
        if not is_visual:
            ax_path.fill_between(xs, diff_path, alpha=0.15, color="#6366f1")

        ax_path.set_xlabel("Question Number", color=text_color, fontsize=9, labelpad=5)
        ax_path.set_ylabel("Difficulty Level", color=text_color, fontsize=9, labelpad=5)
        
        ax_path.set_yticks([1, 2, 3, 4, 5])
        ax_path.set_yticklabels(["1", "2", "3", "4", "5"])
        ax_path.set_xticks(xs)
        ax_path.tick_params(colors=text_color, labelsize=8)

        if is_visual:
            for spine in ["top", "right", "left", "bottom"]:
                ax_path.spines[spine].set_color("#ffffff")
                ax_path.spines[spine].set_linewidth(1)
        else:
            for spine in ["top", "right", "left", "bottom"]:
                ax_path.spines[spine].set_color("none")

        ax_path.grid(True, color=grid_color, linestyle="--", linewidth=0.5, alpha=grid_alpha)
        plt.tight_layout()
        st.pyplot(fig_path)
        plt.close()

    st.divider()

    # AI Explainability Section
    st.subheader("🤖 Behind the AI: Feature Vector Analysis")
    st.markdown("""
    The Machine Learning classifier (trained model) processes **12 individual interaction indicators** calculated from your assessment session to classify your ultimate skill profile.
    """)

    times = [a["response_time"] for a in answers]
    avg_diff = result['avg_difficulty']

    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem; border-color: rgba(99,102,241,0.2); margin-bottom: 0.8rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Average Difficulty served</span><br>
            <span style="font-size:1.15rem; font-weight:700; color:#6366f1;">{avg_diff:.2f} / 5.0</span>
        </div>
        <div class="glass-card" style="padding: 1rem; border-color: rgba(59,130,246,0.2); margin-bottom: 0.8rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Max Difficulty reached</span><br>
            <span style="font-size:1.15rem; font-weight:700; color:#3b82f6;">{result['max_difficulty']} / 5</span>
        </div>
        <div class="glass-card" style="padding: 1rem; border-color: rgba(236,72,153,0.2); margin-bottom: 0.8rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Response Time Std Dev</span><br>
            <span style="font-size:1.15rem; font-weight:700; color:#ec4899;">{np.std(times, ddof=1):.2f}s</span>
        </div>
        <div class="glass-card" style="padding: 1rem; border-color: rgba(16,185,129,0.2); margin-bottom: 0.8rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Accessibility timings modifier</span><br>
            <span style="font-size:1.15rem; font-weight:700; color:#10b981;">{DISABILITY_MULTIPLIERS.get(disability, 1.0)}x</span>
        </div>
        """, unsafe_allow_html=True)

    with col_feat2:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem; border-color: rgba(99,102,241,0.2); margin-bottom: 0.8rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Raw Accuracy</span><br>
            <span style="font-size:1.15rem; font-weight:700; color:#6366f1;">{result['accuracy']:.1%}</span>
        </div>
        <div class="glass-card" style="padding: 1rem; border-color: rgba(59,130,246,0.2); margin-bottom: 0.8rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Response time range</span><br>
            <span style="font-size:1.15rem; font-weight:700; color:#3b82f6;">{max(times) - min(times):.2f}s</span>
        </div>
        <div class="glass-card" style="padding: 1rem; border-color: rgba(236,72,153,0.2); margin-bottom: 0.8rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Timing adjusted by difficulty</span><br>
            <span style="font-size:1.15rem; font-weight:700; color:#ec4899;">{result['avg_response_time'] / avg_diff if avg_diff > 0 else 0:.2f}s/unit</span>
        </div>
        <div class="glass-card" style="padding: 1rem; border-color: rgba(16,185,129,0.2); margin-bottom: 0.8rem;">
            <span style="color:#94a3b8; font-size:0.8rem;">Evaluation Session ID</span><br>
            <span style="font-size:0.95rem; font-weight:700; color:#10b981; font-family:monospace;">{st.session_state.session_id}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Step-by-step History details
    st.subheader("📝 Session Timeline History")
    detail_data = []
    for i, a in enumerate(answers, 1):
        detail_data.append({
            "Q#": i,
            "Question text snippet": a["question_text"][:50] + "...",
            "Difficulty served": a["difficulty"],
            "Your Selected Answer": a["selected_option"],
            "Correct Answer": a["correct_option"],
            "Is Correct": "✅ Correct" if a["is_correct"] else "❌ Incorrect",
            "Response Duration": f"{a['response_time']:.1f}s",
        })
    st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

    st.divider()

    # Restart button
    if st.button("🔄 Restart Assessment Session", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------------------------------------------------------------------------
# Sidebar information
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
        <div style="font-size: 3.5rem;">🤖</div>
        <h2 style="margin: 0; color: #ffffff; font-weight: 700;">Tamkeen AI</h2>
        <p style="color: #94a3b8; font-size: 0.825rem; margin-top: 0.25rem;">Skill Assessment Platform</p>
    </div>
    <hr style="border-color: rgba(255,255,255,0.06); margin: 1rem 0;" />
    """, unsafe_allow_html=True)

    if st.session_state.stage == "result":
        st.markdown(f"""
        ### Classifier Details
        <div class="glass-card" style="padding: 1.1rem; border-color: rgba(99,102,241,0.25);">
            <p style="margin:0 0 5px 0; font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">MODEL TYPE</p>
            <b style="color:#ffffff;">{model_data['model_name']}</b>
            <hr style="border-color: rgba(255,255,255,0.05); margin: 10px 0;" />
            <p style="margin:0 0 5px 0; font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">TEST SET ACCURACY</p>
            <b style="color:#10b981; font-size:1.15rem;">{model_data['test_accuracy']:.2%}</b>
            <hr style="border-color: rgba(255,255,255,0.05); margin: 10px 0;" />
            <p style="margin:0 0 5px 0; font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">CROSS-VALIDATION</p>
            <b style="color:#3b82f6;">{model_data['cv_score']:.4f}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        ### Assessment Overview
        Welcome to the adaptive assessment dashboard, **{st.session_state.student_name}**!
        
        This evaluation uses:
        1. **Adaptive Difficulty Scaling**: Starts at medium difficulty (3) and dynamically scales up or down based on response accuracy.
        2. **Accessibility Multipliers**: Calibrates timing features dynamically per disability profile.
        3. **ML Classifier**: Runs model inference on aggregated data to identify final skill proficiency level.
        """)
