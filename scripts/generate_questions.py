import pandas as pd
import random

def generate_question_bank():
    courses_df = pd.read_csv('data/raw/courses.csv')
    
    # Define themes for categories
    themes = {
        'python': {
            'q': [
                "What is the purpose of 'self' in Python classes?",
                "Which of these is a mutable data type?",
                "What does the 'with' statement do in Python?",
                "How do you handle exceptions in Python?",
                "What is a decorator in Python?",
                "What is the difference between list and tuple?",
                "How does memory management work in Python?",
                "What is PEP 8?",
                "What is a generator function?",
                "What is the GIL (Global Interpreter Lock)?"
            ],
            'options': ["Option A|Option B|Option C|Option D"],
            'type': 'multiple_choice'
        },
        'programming': {
            'q': [
                "What is the Big O complexity of searching in a sorted array using binary search?",
                "Which principle of OOP refers to hiding internal details?",
                "What is a deadlock in concurrent programming?",
                "What is the difference between a stack and a queue?",
                "What is a pointer in C++?",
                "What is recursion?",
                "What is a pure function in functional programming?",
                "What is polymorphism?",
                "What is the purpose of an interface?",
                "What is refactoring?"
            ],
            'options': ["Option A|Option B|Option C|Option D"],
            'type': 'multiple_choice'
        },
        'design': {
            'q': [
                "What is the principle of 'Hierarchy' in design?",
                "Which color model is used for screen displays?",
                "What is 'Kerning' in typography?",
                "What is the difference between UX and UI?",
                "What is a wireframe?",
                "What is the Rule of Thirds?",
                "What is negative space?",
                "What is a design system?",
                "What is accessibility in web design?",
                "What is a mood board?"
            ],
            'options': ["Option A|Option B|Option C|Option D"],
            'type': 'multiple_choice'
        },
        'ux': {
            'q': [
                "What is a user persona?",
                "What is usability testing?",
                "What is an affinity diagram used for?",
                "What is the 'Golden Ratio' in layout design?",
                "What is a journey map?",
                "What is information architecture?",
                "What is A/B testing?",
                "What is a heat map?",
                "What is the difference between qualitative and quantitative research?",
                "What is accessibility (a11y)?"
            ],
            'options': ["Option A|Option B|Option C|Option D"],
            'type': 'multiple_choice'
        },
        'testing': {
            'q': [
                "What is regression testing?",
                "What is the difference between unit testing and integration testing?",
                "What is black-box testing?",
                "What is a test plan?",
                "What is bug life cycle?",
                "What is test-driven development (TDD)?",
                "What is smoke testing?",
                "What is the purpose of boundary value analysis?",
                "What is performance testing?",
                "What is static testing?"
            ],
            'options': ["Option A|Option B|Option C|Option D"],
            'type': 'multiple_choice'
        },
        'qa': {
            'q': [
                "What is the difference between QA and QC?",
                "What is Selenium used for?",
                "What is continuous integration?",
                "What is a test case?",
                "What is edge case testing?",
                "What is exploratory testing?",
                "What is code coverage?",
                "What is load testing?",
                "What is user acceptance testing (UAT)?",
                "What is the Agile testing quadrant?"
            ],
            'options': ["Option A|Option B|Option C|Option D"],
            'type': 'multiple_choice'
        },
        'iot': {
            'q': [
                "What does MQTT stand for?",
                "Which protocol is commonly used for low-power IoT devices?",
                "What is an actuator in IoT?",
                "What is edge computing?",
                "What is a gateway in IoT?",
                "What is the purpose of a microcontroller?",
                "What is Zigbee?",
                "What is an ultrasonic sensor used for?",
                "What is the difference between IPv4 and IPv6 in IoT?",
                "What is a digital twin?"
            ],
            'options': ["Option A|Option B|Option C|Option D"],
            'type': 'multiple_choice'
        },
        'embedded': {
            'q': [
                "What is a Real-Time Operating System (RTOS)?",
                "What is the purpose of a watchdog timer?",
                "What is an interrupt service routine (ISR)?",
                "What is the difference between RAM and Flash in embedded systems?",
                "What is I2C communication?",
                "What is SPI protocol?",
                "What is GPIO?",
                "What is a bootloader?",
                "What is UART?",
                "What is power management in embedded systems?"
            ],
            'options': ["Option A|Option B|Option C|Option D"],
            'type': 'multiple_choice'
        },
        'marketing': {
            'q': [
                "What is SEO?",
                "What is the difference between B2B and B2C marketing?",
                "What is a conversion rate?",
                "What is PPC?",
                "What is content marketing?",
                "What is social media engagement?",
                "What is a CTA (Call to Action)?",
                "What is email marketing segmentation?",
                "What is brand positioning?",
                "What is a lead magnet?"
            ],
            'options': ["Option A|Option B|Option C|Option D"],
            'type': 'multiple_choice'
        }
    }

    # Default theme for any other category
    default_theme = {
        'q': [
            "What is the primary goal of this field?",
            "Which of these is a common best practice?",
            "What is a key concept in this topic?",
            "How do you measure success here?",
            "Which tool is most commonly used?",
            "What is the most important skill for this role?",
            "What is a common challenge in this area?",
            "What is the first step in the standard process?",
            "What does the standard industry acronym stand for?",
            "Which of these is an advanced technique?"
        ],
        'options': ["Option A|Option B|Option C|Option D"],
        'type': 'multiple_choice'
    }

    questions_list = []
    
    # We'll create 10-15 questions per course to have enough variety for difficulty 1-5
    for _, course in courses_df.iterrows():
        cat = course['category'].lower()
        course_id = course['course_id']
        # Map course_id to assessment_id (1:1 for simplicity)
        assessment_id = f"assessment_{course_id.split('_')[-1]}"
        
        theme = themes.get(cat, default_theme)
        if cat not in themes:
            # Try to find a sub-theme
            for key in themes:
                if key in cat:
                    theme = themes[key]
                    break
        
        # Generate questions for each difficulty level 1-5
        for diff in range(1, 6):
            # Select 2-3 questions per difficulty
            num_q = 2
            for i in range(num_q):
                q_text = random.choice(theme['q'])
                if diff > 3:
                    q_text = f"Advanced: {q_text}"
                elif diff < 3:
                    q_text = f"Intro: {q_text}"
                
                questions_list.append({
                    'question_id': f"q_{assessment_id}_{diff}_{i}",
                    'assessment_id': assessment_id,
                    'question_text': q_text,
                    'question_type': theme['type'],
                    'correct_answer': 'A', # Placeholder correct answer
                    'options': theme['options'][0],
                    'difficulty_level': diff
                })
                
    df_questions = pd.DataFrame(questions_list)
    df_questions.to_csv('data/raw/questions.csv', index=False)
    print(f"Generated {len(df_questions)} questions for {len(courses_df)} courses.")

if __name__ == "__main__":
    generate_question_bank()
