# Tamkeen - Accessibility-Aware Skill Assessment

An AI-powered skill prediction system that accurately assesses learner skill levels while accounting for disability-related interaction differences.

## The Problem

Traditional skill assessment systems penalize users with disabilities. A visually impaired learner using a Screen Reader takes 3-4x longer to answer -- but that does **not** mean they are less skilled. This model learns to separate "slow because unskilled" from "slow because of disability."

## Project Structure

```
tamkeen-skill-assessment/
|-- data/
|   |-- processed/               # Generated training data
|       |-- assessment_training_data.csv
|-- scripts/
|   |-- generate_data.py         # Synthetic data generator
|-- src/
|   |-- models/                  # Trained model artifacts (.pkl)
|-- notebooks/
|   |-- 01_data_analysis.ipynb   # EDA & exploration
|   |-- 02_model_training.ipynb  # Training & evaluation
|-- docs/                        # Documentation
|-- README.md
|-- requirements.txt
|-- .gitignore
```

**Data Source:** User and course data is read from the sibling project
`tamkeen-recommendation-system/out_small/` (100 users, 50 courses, 861 questions).

## Disability Time Multipliers

| Disability     | Multiplier | Reason                          |
|----------------|------------|----------------------------------|
| Visual         | x3.5       | Screen Reader overhead           |
| Motor          | x2.5       | Switch / eye-tracking input      |
| Cognitive      | x1.8       | Extra processing time            |
| Hearing        | x1.3       | Caption / sign-language lookup   |
| No Disability  | x1.0       | Baseline                         |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (reads from ../tamkeen-recommendation-system/out_small/)
python scripts/generate_data.py

# 3. Run the notebooks
# Open notebooks/skill_assessment_model.ipynb for training
```

## Dataset

- **100 real users** from the recommendation system (4 disability types)
- **50 real courses** with difficulty levels 1-5
- **1,242 simulated quiz interactions** (10-15 questions per user)
- **3 skill levels**: Beginner (35%), Intermediate (35%), Advanced (30%)
- **8 columns**: user_id, course_id, disability_type, question_number, question_difficulty, is_correct, response_time_seconds, true_skill_label

## Tech Stack

- Python 3.9
- pandas, numpy (data processing)
- scikit-learn (ML models)
- matplotlib, seaborn (visualization)

## Team

Tamkeen Graduation Project - AI Team
