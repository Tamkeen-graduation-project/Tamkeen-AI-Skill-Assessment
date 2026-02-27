"""
generate_data.py
================
Generates synthetic training data for the Accessibility-Aware Skill Assessment Model.

Reads REAL user and course data from the Tamkeen Recommendation System project
(../tamkeen-recommendation-system/out_small/) and simulates quiz interactions.

The key innovation: response time is inflated by disability-specific multipliers so the
model can learn to distinguish "slow because unskilled" from "slow because of
a disability (e.g., Screen Reader overhead)."

Input (from recommendation system)
-----------------------------------
- users.csv     : user_id, disability_type, pref_screen_reader, etc.
- courses.csv   : course_id, category, difficulty_level, etc.
- questions.csv : question_id, assessment_id, difficulty_level, question_type

Output
------
- data/processed/assessment_training_data.csv
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Read from the recommendation system's data
RECOMMENDATION_DATA = os.path.join(
    PROJECT_ROOT, "..", "tamkeen-recommendation-system", "out_small"
)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "assessment_training_data.csv")

SKILL_LABELS = ["Beginner", "Intermediate", "Advanced"]

# Probability of assigning each skill level (roughly balanced)
SKILL_WEIGHTS = [0.35, 0.35, 0.30]

# Number of courses each user takes quizzes on
MIN_COURSES_PER_USER = 3
MAX_COURSES_PER_USER = 8

# Number of questions each user answers per quiz session
MIN_QUESTIONS = 10
MAX_QUESTIONS = 15

# -----------------------------------------------------------
# Disability Time Multipliers
# -----------------------------------------------------------
# These multipliers model the extra time a user needs to interact
# with the quiz interface due to their disability.  They do NOT
# reflect lower ability -- only interaction overhead.
#
#   visual     x3.5  -- Screen reader reads every element aloud;
#                       the user must listen, navigate, and confirm.
#   motor      x2.5  -- Switch / eye-tracking input is slower than
#                       a mouse click.
#   cognitive  x1.8  -- Extra processing time for reading and
#                       understanding the question.
#   hearing    x1.3  -- Minimal overhead; mainly affects audio-based
#                       questions (captions / sign-language lookup).
#   none       x1.0  -- Baseline, no additional overhead.
# -----------------------------------------------------------
DISABILITY_TIME_MULTIPLIERS = {
    "visual":    3.5,
    "motor":     2.5,
    "cognitive": 1.8,
    "hearing":   1.3,
    "none":      1.0,
}

# -----------------------------------------------------------
# Base response time per question difficulty (seconds)
# -----------------------------------------------------------
# Questions in the recommendation system have difficulty 1-5.
# Harder questions naturally take longer.
BASE_TIME_BY_DIFFICULTY = {
    1: 15,   # Very Easy
    2: 25,   # Easy
    3: 40,   # Medium
    4: 55,   # Hard
    5: 70,   # Very Hard
}

# -----------------------------------------------------------
# Correctness probabilities  P(correct | skill, difficulty)
# -----------------------------------------------------------
# Rows = skill level, Columns = question difficulty (1-5)
# A Beginner rarely answers Hard/Very Hard questions correctly.
# An Advanced learner almost always answers Easy ones correctly.
CORRECTNESS_PROB = {
    "Beginner": {
        1: 0.80, 2: 0.60, 3: 0.35, 4: 0.15, 5: 0.05,
    },
    "Intermediate": {
        1: 0.95, 2: 0.80, 3: 0.60, 4: 0.40, 5: 0.20,
    },
    "Advanced": {
        1: 0.98, 2: 0.92, 3: 0.80, 4: 0.65, 5: 0.45,
    },
}

# -----------------------------------------------------------
# Skill-based time adjustment factor
# -----------------------------------------------------------
# Skilled users answer faster; beginners hesitate longer.
# Multiplied with base_time BEFORE the disability multiplier.
SKILL_TIME_FACTOR = {
    "Beginner":     1.3,   # 30% slower than baseline
    "Intermediate": 1.0,   # baseline
    "Advanced":     0.75,  # 25% faster than baseline
}


# ============================================================
# Helper functions
# ============================================================
def load_inputs():
    """Load users, courses, and questions from the recommendation system."""
    users_path = os.path.join(RECOMMENDATION_DATA, "users.csv")
    courses_path = os.path.join(RECOMMENDATION_DATA, "courses.csv")
    questions_path = os.path.join(RECOMMENDATION_DATA, "questions.csv")

    for p in [users_path, courses_path, questions_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                f"Make sure the recommendation system data is at: {RECOMMENDATION_DATA}"
            )

    users = pd.read_csv(users_path)
    courses = pd.read_csv(courses_path)
    questions = pd.read_csv(questions_path)

    # Normalize disability_type to lowercase
    users["disability_type"] = users["disability_type"].fillna("none").str.lower()

    print(f"Loaded from recommendation system:")
    print(f"  Users:     {len(users)} ({dict(users['disability_type'].value_counts())})")
    print(f"  Courses:   {len(courses)} (difficulty 1-{courses['difficulty_level'].max()})")
    print(f"  Questions: {len(questions)} (difficulty 1-{questions['difficulty_level'].max()})")

    return users, courses, questions


def assign_skill(n_users):
    """Randomly assign a true skill label to each user."""
    return np.random.choice(SKILL_LABELS, size=n_users, p=SKILL_WEIGHTS)


def simulate_question(skill, difficulty, disability_type):
    """
    Simulate a single question interaction.

    Returns
    -------
    is_correct : int (0 or 1)
    response_time_seconds : float
    """
    # Clamp difficulty to 1-5
    difficulty = max(1, min(5, difficulty))

    # --- Correctness ---
    p_correct = CORRECTNESS_PROB[skill][difficulty]
    is_correct = int(np.random.random() < p_correct)

    # --- Response time ---
    # 1. Start with base time for this difficulty
    base = BASE_TIME_BY_DIFFICULTY[difficulty]

    # 2. Adjust for skill level (beginners slower, advanced faster)
    base *= SKILL_TIME_FACTOR[skill]

    # 3. If the answer is wrong, add "hesitation" time (they thought
    #    longer but still got it wrong)
    if not is_correct:
        base *= np.random.uniform(1.1, 1.5)

    # 4. Apply disability multiplier -- this is the KEY step.
    #    A visually impaired user with a Screen Reader needs ~3.5x
    #    the time regardless of their actual knowledge level.
    multiplier = DISABILITY_TIME_MULTIPLIERS.get(disability_type, 1.0)
    base *= multiplier

    # 5. Add random noise (+/- 20%) to avoid perfectly uniform data
    noise = np.random.uniform(0.80, 1.20)
    response_time = round(base * noise, 1)

    return is_correct, response_time


# ============================================================
# Main generation loop
# ============================================================
def generate_dataset(users, courses, questions):
    """
    For each user, simulate quiz sessions across MULTIPLE courses.
    Each user takes between MIN_COURSES_PER_USER and MAX_COURSES_PER_USER courses.
    """
    records = []

    # Assign a true skill label to every user
    skills = assign_skill(len(users))

    for idx, (_, user_row) in enumerate(users.iterrows()):
        user_id = user_row["user_id"]
        disability = user_row["disability_type"]
        skill = skills[idx]

        # Pick multiple random courses for this user
        n_courses = np.random.randint(MIN_COURSES_PER_USER, MAX_COURSES_PER_USER + 1)
        user_courses = courses.sample(min(n_courses, len(courses)))

        for _, course_row in user_courses.iterrows():
            course_id = course_row["course_id"]
            course_difficulty = course_row["difficulty_level"]

            # Number of questions for this session
            n_questions = np.random.randint(MIN_QUESTIONS, MAX_QUESTIONS + 1)

            for q_num in range(n_questions):
                # Use course difficulty as base, add some variation (+/- 1)
                q_difficulty = course_difficulty + np.random.randint(-1, 2)
                q_difficulty = max(1, min(5, q_difficulty))

                is_correct, resp_time = simulate_question(
                    skill, q_difficulty, disability
                )

                records.append({
                    "user_id":               user_id,
                    "course_id":             course_id,
                    "disability_type":       disability,
                    "question_number":       q_num + 1,
                    "question_difficulty":   q_difficulty,
                    "is_correct":            is_correct,
                    "response_time_seconds": resp_time,
                    "true_skill_label":      skill,
                })

    df = pd.DataFrame(records)
    return df


def main():
    print("=" * 60)
    print("  Accessibility-Aware Assessment Data Generator")
    print("=" * 60)

    # 1. Load from recommendation system
    users, courses, questions = load_inputs()

    # 2. Generate
    print("\nGenerating quiz interactions ...")
    df = generate_dataset(users, courses, questions)

    # 3. Print summary
    print(f"\nGenerated {len(df):,} rows for {df['user_id'].nunique()} users.")

    print(f"\nSkill distribution:")
    for label in SKILL_LABELS:
        n_users = df[df["true_skill_label"] == label]["user_id"].nunique()
        n_rows = (df["true_skill_label"] == label).sum()
        print(f"  {label:15s}: {n_users:>4} users, {n_rows:>5,} rows")

    print(f"\nDisability distribution:")
    for dis in sorted(DISABILITY_TIME_MULTIPLIERS.keys()):
        sub = df[df["disability_type"] == dis]
        n = sub["user_id"].nunique()
        if n > 0:
            avg_time = sub["response_time_seconds"].mean()
            print(f"  {dis:12s}: {n:>4} users, avg response time = {avg_time:.1f}s")

    print(f"\nMean response time by disability x skill:")
    pivot = df.groupby(["disability_type", "true_skill_label"])[
        "response_time_seconds"
    ].mean().unstack()
    print(pivot.round(1).to_string())

    # 4. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
