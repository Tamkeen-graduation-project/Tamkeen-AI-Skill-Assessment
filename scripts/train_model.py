
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = "data/processed/assessment_training_data.csv"
MODEL_DIR = "src/models"
MODEL_PATH = os.path.join(MODEL_DIR, "skill_predictor.pkl")

# Feature definition (MUST match API expectation)
FEATURES = [
    "disability_type",
    "num_questions",
    "avg_difficulty",
    "max_difficulty",
    "accuracy",
    "total_correct",
    "avg_response_time",
    "std_response_time",
    "max_response_time",
    "min_response_time",
    "time_range",
    "time_per_difficulty"
]

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run generate_data.py first.")
        return

    print(f"Loading training data from {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)

    # Aggregate by (user_id, course_id) to create session features
    print("Aggregating interactions into session features...")
    
    # We need to preserve disability_type and true_skill_label (they are the same for all rows in a session)
    agg_funcs = {
        'question_difficulty': ['count', 'mean', 'max'],
        'is_correct': ['mean', 'sum'],
        'response_time_seconds': ['mean', 'std', 'max', 'min'],
        'disability_type': 'first',
        'true_skill_label': 'first'
    }
    
    df = df_raw.groupby(['user_id', 'course_id']).agg(agg_funcs).reset_index()
    
    # Flatten multi-index columns
    df.columns = [
        'user_id', 'course_id', 
        'num_questions', 'avg_difficulty', 'max_difficulty',
        'accuracy', 'total_correct',
        'avg_response_time', 'std_response_time', 'max_response_time', 'min_response_time',
        'disability_type', 'true_skill_label'
    ]
    
    # Fill NaN for std if only one question (unlikely but safe)
    df['std_response_time'] = df['std_response_time'].fillna(0)
    
    # Add derived features
    df['time_range'] = df['max_response_time'] - df['min_response_time']
    df['time_per_difficulty'] = df['avg_response_time'] / df['avg_difficulty']

    # 1. Encode disability_type
    disability_order = ["none", "visual", "motor", "cognitive", "hearing"]
    disability_map = {d: i for i, d in enumerate(disability_order)}
    df['disability_type_encoded'] = df['disability_type'].map(disability_map)
    
    # Map back to original name for feature selection
    df['disability_type'] = df['disability_type_encoded']

    X = df[FEATURES]
    
    # 2. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['true_skill_label'])
    label_map = {label: int(code) for code, label in enumerate(le.classes_)}

    print(f"Training XGBClassifier on {len(X)} samples...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X, y)

    # 3. Save model bundle
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "disability_map": disability_map,
        "label_map": label_map,
        "description": "Accessibility-aware skill predictor trained on real course categories"
    }
    
    joblib.dump(bundle, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")
    print(f"Labels: {label_map}")

if __name__ == "__main__":
    train()
