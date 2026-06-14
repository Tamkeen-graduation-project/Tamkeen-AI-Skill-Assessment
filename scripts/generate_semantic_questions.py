
import os
import pandas as pd
import random

RAW_DATA_DIR = "data/raw"

SEMANTIC_DATA = {
    "digital_marketing": [
        {
            "q": "When does the marketing process traditionally begin?",
            "options": ["After the product is manufactured", "Even before the product exists", "Only when sales decline", "After the first customer complaint"],
            "ans": 1,
            "diff": 1
        },
        {
            "q": "What is the core definition of Digital Marketing according to Tamkeen Content?",
            "options": ["Posting on social media every day", "The intersection of marketing science and information technology", "Buying ads on Google", "Designing a website"],
            "ans": 1,
            "diff": 2
        },
        {
            "q": "Which stage of marketing focuses on turning the audience into customers?",
            "options": ["Market Research", "Branding", "Conversion", "Loyalty Group"],
            "ans": 2,
            "diff": 3
        },
        {
            "q": "Digital Marketing proficiency requirements vary based on what?",
            "options": ["The budget of the company", "The specific specialization", "The age of the marketer", "The size of the laptop used"],
            "ans": 1,
            "diff": 4
        }
    ],
    "python": [
        {
            "q": "What is the primary characteristic of a Scalar object in Python?",
            "options": ["It can be converted to a list", "You cannot derive a new object from it", "It always has a decimal point", "It can only be used in loops"],
            "ans": 1,
            "diff": 1
        },
        {
            "q": "What does the int() function do when given a float like 3.9?",
            "options": ["Rounds it to 4", "Returns 3 (removes the fraction)", "Returns 3.0", "Errors"],
            "ans": 1,
            "diff": 2
        },
        {
            "q": "Which of these is a Non-Scalar object in Python?",
            "options": ["int", "float", "list", "bool"],
            "ans": 2,
            "diff": 3
        },
        {
            "q": "What is returned by type(None) in Python?",
            "options": ["None", "class 'NoneType'", "Null", "0"],
            "ans": 1,
            "diff": 4
        }
    ],
    "embedded": [
        {
            "q": "What is the primary role of a 'device driver' in an embedded system?",
            "options": ["To power the hardware", "To act as a translator between application and hardware", "To store user data", "To increase the CPU speed"],
            "ans": 1,
            "diff": 1
        },
        {
            "q": "Why can't an application interact with hardware directly?",
            "options": ["Hardware is too fast", "Someone needs to communicate in the hardware's specific language", "Security protocols always prevent it", "Applications are only for software"],
            "ans": 1,
            "diff": 2
        }
    ],
    "iot": [
        {
            "q": "Which of the following was the first category of devices connected to the Internet?",
            "options": ["Smart watches", "Computers", "Mobile phones", "Cars"],
            "ans": 1,
            "diff": 1
        },
        {
            "q": "In the context of IoT, what is a 'server'?",
            "options": ["A person who helps users", "A main computer that offers services to other devices", "A type of sensor", "A wireless router"],
            "ans": 1,
            "diff": 2
        },
        {
            "q": "Which category of IoT devices includes water and electricity meters?",
            "options": ["Personal devices", "Utility meters", "Home appliances", "Carrier networks"],
            "ans": 1,
            "diff": 3
        }
    ],
    "ux": [
        {
            "q": "What is the main goal of Information Architecture in UX?",
            "options": ["Choosing the right colors", "Structuring and organizing content logically", "Writing code for the frontend", "Testing the server speed"],
            "ans": 1,
            "diff": 2
        }
    ]
}

def generate_questions():
    questions = []
    
    # Process categories with real semantic data
    for category, items in SEMANTIC_DATA.items():
        assessment_id = f"assessment_{category}"
        for i, item in enumerate(items):
            questions.append({
                "question_id": f"q_{assessment_id}_{item['diff']}_{i}",
                "assessment_id": assessment_id,
                "question_text": item['q'],
                "question_type": "multiple_choice",
                "correct_answer": "ABCD"[item['ans']],
                "options": "|".join(item['options']),
                "difficulty_level": item['diff']
            })
            
    # Add some generic but better questions for other categories or to fill the gaps
    # For a production system, we'd have semantic data for all. 
    # For now, we'll fill to ensure 5 difficulties per course.
    categories = ["digital_marketing", "embedded", "freelancing_khamsat", "iot", "python", "testing", "ux"]
    
    for cat in categories:
        a_id = f"assessment_{cat}"
        existing_diffs = [q['difficulty_level'] for q in questions if q['assessment_id'] == a_id]
        
        for diff in range(1, 6):
            if existing_diffs.count(diff) < 3:
                for i in range(3 - existing_diffs.count(diff)):
                    questions.append({
                        "question_id": f"q_{a_id}_{diff}_gen_{i}",
                        "assessment_id": a_id,
                        "question_text": f"Semantic question about {cat.replace('_', ' ')} (Difficulty {diff})",
                        "question_type": "multiple_choice",
                        "correct_answer": "A",
                        "options": "Correct semantic answer|Wrong answer 1|Wrong answer 2|Wrong answer 3",
                        "difficulty_level": diff
                    })

    return pd.DataFrame(questions)

def main():
    print("Generating semantic questions based on Tamkeen transcripts...")
    df = generate_questions()
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(RAW_DATA_DIR, "questions_v2.csv"), index=False)
    print(f"Saved {len(df)} semantic questions to {RAW_DATA_DIR}/questions_v2.csv")

if __name__ == "__main__":
    main()
