
import os
import json
import pandas as pd
import random
from typing import List, Dict

# Paths
CONTENT_BASE = r"f:\Cs\Projects\Tamkeen\Tamkeen Content\project_data\courses"
RAW_DATA_DIR = "data/raw"

def get_real_courses():
    courses = []
    if not os.path.exists(CONTENT_BASE):
        print(f"Error: {CONTENT_BASE} not found.")
        return []

    for course_slug in os.listdir(CONTENT_BASE):
        course_path = os.path.join(CONTENT_BASE, course_slug, "meta", "course.json")
        if os.path.exists(course_path):
            with open(course_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                lessons = [l['title'] for l in data.get('lessons', [])]
                courses.append({
                    "course_id": course_slug,
                    "title": data.get('title', course_slug.replace('_', ' ').title()),
                    "category": course_slug,
                    "lessons": lessons,
                    "description": f"Official Tamkeen course on {course_slug.replace('_', ' ')}."
                })
    return courses

def generate_questions(courses):
    questions = []
    
    # Generic templates for question generation based on lesson titles
    templates = [
        ("What is the main focus of '{lesson}'?", ["Basic concepts", "Advanced optimization", "Historical context", "Legal regulations"], 0),
        ("In the context of {course}, what does '{lesson}' typically cover?", ["Fundamental principles", "Hardware maintenance", "Social media trends", "Cooking techniques"], 0),
        ("Which of the following is a key takeaway from '{lesson}'?", ["Practical implementation", "Random guessing", "Ignoring errors", "Using outdated tools"], 0),
        ("Why is '{lesson}' important for {course}?", ["It builds core foundations", "It is an optional trivia", "It is only for experts", "It has no practical use"], 0)
    ]

    for course in courses:
        course_id = course['course_id']
        assessment_id = f"assessment_{course_id}"
        lessons = course['lessons']
        
        # Ensure we have questions for all 5 difficulties
        for diff in range(1, 6):
            # Generate 4-6 questions per difficulty level
            num_q = random.randint(4, 6)
            for i in range(num_q):
                lesson = random.choice(lessons) if lessons else "General Concepts"
                template, options, ans_idx = random.choice(templates)
                
                q_text = template.format(lesson=lesson, course=course['title'])
                if diff >= 4:
                    q_text = f"Advanced: {q_text}"
                elif diff <= 2:
                    q_text = f"Intro: {q_text}"
                
                questions.append({
                    "question_id": f"q_{assessment_id}_{diff}_{i}",
                    "assessment_id": assessment_id,
                    "question_text": q_text,
                    "question_type": "multiple_choice",
                    "correct_answer": "ABCD"[ans_idx],
                    "options": "|".join(options),
                    "difficulty_level": diff
                })
    
    return pd.DataFrame(questions)

def main():
    print("Extracting real courses from Tamkeen Content...")
    courses = get_real_courses()
    if not courses:
        return

    # 1. Generate courses.csv
    courses_df = pd.DataFrame([
        {
            "course_id": c["course_id"],
            "category": c["category"],
            "difficulty_level": random.randint(1, 3),
            "duration_hours": round(random.uniform(2, 15), 2),
            "rating": round(random.uniform(3.5, 5.0), 2),
            "completion_rate": round(random.uniform(0.5, 0.9), 3),
            "is_mock": False,
            "tags": "[]",
            "related_to": "[]",
            "skills": "[]",
            "title": c["title"],
            "description": c["description"]
        } for c in courses
    ])
    
    # Add some mock variations to maintain dataset size if needed for training
    # But for isolation, we prefer the real ones.
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    courses_df.to_csv(os.path.join(RAW_DATA_DIR, "courses.csv"), index=False)
    print(f"Saved {len(courses_df)} real courses to {RAW_DATA_DIR}/courses.csv")

    # 2. Generate questions.csv
    questions_df = generate_questions(courses)
    questions_df.to_csv(os.path.join(RAW_DATA_DIR, "questions.csv"), index=False)
    print(f"Saved {len(questions_df)} questions to {RAW_DATA_DIR}/questions.csv")

if __name__ == "__main__":
    main()
