# train_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import random

np.random.seed(42)
random.seed(42)

def generate_synthetic_students(n=2000):
    # Basic student meta
    reg_nos = [f"REG{1000+i}" for i in range(n)]
    names = [f"Student_{i+1}" for i in range(n)]
    depts = np.random.choice(["CS", "IT", "BCA"], size=n)
    years = np.random.choice(["I", "II", "III"], size=n)
    sections = np.random.choice(["A", "B", "C"], size=n)
    semesters = np.random.choice([1,2,3,4,5,6], size=n)

    # Academic features
    attendance_pct = np.clip(np.random.normal(80, 10, n), 30, 100)      # %
    avg_internal_marks = np.clip(np.random.normal(65, 15, n), 0, 100)   # out of 100
    prev_cgpa = np.clip(np.random.normal(7, 1.2, n), 4, 10)             # 4-10
    num_backlogs = np.random.poisson(0.7, n)
    num_backlogs = np.clip(num_backlogs, 0, 6)
    assignment_score = np.clip(np.random.normal(70, 15, n), 0, 100)

    # Risk score (higher means more at-risk)
    # If attendance & marks low, CGPA low, backlogs high => high risk
    raw_risk = (
        0.06 * (60 - attendance_pct) +        # low attendance increases risk
        0.05 * (60 - avg_internal_marks) +    # low internal marks
        0.35 * (7 - prev_cgpa) +              # low CGPA
        0.6 * num_backlogs +                  # more backlogs
        0.03 * (60 - assignment_score) +
        np.random.normal(0, 1.5, n)           # noise
    )

    # Convert to probability 0-1
    prob_risk = 1 / (1 + np.exp(-raw_risk))

    # Label as 1 = at risk, 0 = not at risk (threshold 0.5)
    at_risk = (prob_risk > 0.5).astype(int)

    df = pd.DataFrame({
        "reg_no": reg_nos,
        "name": names,
        "dept": depts,
        "year": years,
        "section": sections,
        "semester": semesters,
        "attendance_pct": attendance_pct,
        "avg_internal_marks": avg_internal_marks,
        "prev_cgpa": prev_cgpa,
        "num_backlogs": num_backlogs,
        "assignment_score": assignment_score,
        "at_risk": at_risk,
        "risk_probability_true": prob_risk
    })
    return df

def train_model():
    df = generate_synthetic_students(3000)

    feature_cols = [
        "attendance_pct",
        "avg_internal_marks",
        "prev_cgpa",
        "num_backlogs",
        "assignment_score"
    ]
    X = df[feature_cols]
    y = df["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "student_risk_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save a small sample for demo
    sample = df.sample(30, random_state=42)
    sample.to_csv("sample_students.csv", index=False)
    print("Sample data saved to sample_students.csv")

if __name__ == "__main__":
    train_model()
