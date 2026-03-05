from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, send_file
)
import pandas as pd
import joblib
import os
import reportlab
import sqlite3
import csv
from datetime import datetime
from io import BytesIO

app = Flask(__name__)
app.secret_key = "some_secret_key_for_flash"

MODEL_PATH = os.path.join("model", "student_risk_model.joblib")
model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    "attendance_pct",
    "avg_internal_marks",
    "prev_cgpa",
    "num_backlogs",
    "assignment_score"
]

DB_PATH = "student_risk.db"

# ---------- DB helpers ----------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            user TEXT,
            source TEXT,
            total INTEGER,
            safe INTEGER,
            warning INTEGER,
            critical INTEGER
        )
        """
    )
    conn.commit()
    conn.close()

app = Flask(__name__)
app.secret_key = "some_secret_key_for_flash"

init_db()  # <-- RUN DB SETUP ON START


def save_analysis_summary(username, source, counts, total):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO analyses (created_at, user, source, total, safe, warning, critical)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            username,
            source,
            total,
            counts.get("Safe", 0),
            counts.get("Warning", 0),
            counts.get("Critical", 0),
        ),
    )
    conn.commit()
    conn.close()

# store last result in memory for CSV download
LAST_RESULT_DF = None

# ---------- Auth helpers ----------

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "mentor": {"password": "mentor123", "role": "mentor"},
}

def current_user():
    if "username" in session:
        return {
            "username": session["username"],
            "role": session.get("role", "mentor"),
        }
    return None

def login_required(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user():
            flash("Please log in first.", "error")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapper

# ---------- Core ML logic ----------

def compute_risk_levels(df):
    # Ensure required columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]  # probability of being 'at risk'

    df_result = df.copy()
    df_result["risk_probability"] = probs

    levels = []
    actions = []
    top_factors = []

    for idx, row in df_result.iterrows():
        p = row["risk_probability"]
        # Risk level
        if p < 0.33:
            levels.append("Safe")
            actions.append("Maintain performance and consistency.")
        elif p < 0.66:
            levels.append("Warning")
            actions.append("Attend extra practice sessions; monitor closely.")
        else:
            levels.append("Critical")
            actions.append("Immediate counselling and parent meeting required.")

        # Simple rule-based explainability (top factors)
        reasons = []
        try:
            if float(row.get("attendance_pct", 100)) < 75:
                reasons.append("Low attendance")
        except Exception:
            pass
        try:
            if float(row.get("avg_internal_marks", 100)) < 50:
                reasons.append("Low internal marks")
        except Exception:
            pass
        try:
            if float(row.get("prev_cgpa", 10)) < 5.0:
                reasons.append("Low CGPA")
        except Exception:
            pass
        try:
            if int(row.get("num_backlogs", 0)) > 0:
                reasons.append(f"{int(row.get('num_backlogs'))} backlog(s)")
        except Exception:
            pass
        try:
            if float(row.get("assignment_score", 100)) < 50:
                reasons.append("Weak assignment score")
        except Exception:
            pass

        if not reasons:
            reasons = ["Balanced performance"]
        top_factors.append(", ".join(reasons))

    df_result["risk_level"] = levels
    df_result["suggested_action"] = actions
    df_result["top_factors"] = top_factors

    return df_result


# ---------- Routes ----------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = USERS.get(username)
        if user and user["password"] == password:
            session["username"] = username
            session["role"] = user["role"]
            flash(f"Logged in as {username}", "info")
            return redirect(url_for("index"))
        flash("Invalid username or password", "error")
    return render_template("login.html", user=current_user())

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

@app.route("/download_report")
def download_report():
    from flask import send_file
    df = LAST_RESULT_DF.copy()

    file_path = "student_risk_report.pdf"
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 40, "Student Early Warning System — Risk Analysis Report")
    c.setFont("Helvetica", 10)

    y = height - 80
    for index, row in df.iterrows():
        text = f"{row['reg_no']}  |  {row['name']}  |  Risk: {row['risk_level']}  |  Action: {row['suggested_action']}"
        c.drawString(50, y, text)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 60

    c.save()
    return send_file(file_path, as_attachment=True)


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html", user=current_user())


@app.route("/sample")
@login_required
def sample():
    global LAST_RESULT_DF

    if not os.path.exists("sample_students.csv"):
        flash("sample_students.csv missing! Run train_model.py first.", "error")
        return redirect(url_for("index"))

    df = pd.read_csv("sample_students.csv")
    result_df = compute_risk_levels(df)
    LAST_RESULT_DF = result_df

    # --- Load behaviour/issue responses ---
    issues_dict = load_all_issues()  # {reg_no : dict-of-answers}
    result_df["Behaviour / Issues"] = ""   # create new column

    for i, row in result_df.iterrows():
        reg = str(row["reg_no"]).strip()
        if reg in issues_dict:
            summary = build_issue_summary(issues_dict[reg])
            result_df.at[i, "Behaviour / Issues"] = summary

    # Count levels
    counts = result_df["risk_level"].value_counts().to_dict()
    total = len(result_df)

    return render_template(
        "results.html",
        title="Sample Data Analysis",
        tables=[result_df.to_html(classes="table table-striped table-bordered table-dark", index=False)],
        risk_counts=counts,
        total=total,
        critical_students=[
            row for _, row in result_df[result_df["risk_level"] == "Critical"][["reg_no", "name", "dept", "semester"]].iterrows()
        ],
        user=current_user()
    )







@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    global LAST_RESULT_DF

    if "file" not in request.files:
        flash("No file uploaded!", "error")
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("Please select a CSV file.", "error")
        return redirect(url_for("index"))

    # Read uploaded CSV
    df = pd.read_csv(file)
    result_df = compute_risk_levels(df)
    LAST_RESULT_DF = result_df

    # Merge Behaviour / Issues
    issues_dict = load_all_issues()
    result_df["Behaviour / Issues"] = ""
    for i, row in result_df.iterrows():
        reg = str(row["reg_no"]).strip()
        if reg in issues_dict:
            result_df.at[i, "Behaviour / Issues"] = build_issue_summary(issues_dict[reg])

    # Count levels for dashboard
    counts = result_df["risk_level"].value_counts().to_dict()
    total = len(result_df)

    # Save analysis to history
    save_analysis_summary(
        username=current_user()["username"],
        source=file.filename,
        counts=counts,
        total=total,
    )

    html_table = result_df.to_html(
        classes="table table-dark table-striped table-bordered align-middle",
        index=False,
    )

    # Get ONLY critical students for form section
    critical_students = [
        row for _, row in result_df[result_df["risk_level"] == "Critical"][
            ["reg_no", "name", "dept", "semester"]
        ].iterrows()
    ]

    return render_template(
        "results.html",
        title="Uploaded Data Analysis",
        tables=[html_table],
        risk_counts=counts,
        total=total,
        critical_students=critical_students,
        user=current_user(),
    )


@app.route("/summary_ai")
def summary_ai():
    global LAST_RESULT_DF

    if 'LAST_RESULT_DF' not in globals() or LAST_RESULT_DF is None:
        return "No data available. Upload a file or analyze sample first."

    df = LAST_RESULT_DF.copy()

    if "top_factors" not in df.columns:
        return "No risk factors found for summary."

    critical = df[df["risk_level"] == "Critical"]
    warning = df[df["risk_level"] == "Warning"]

    # Most common factor words
    words = df["top_factors"].astype(str).str.lower().str.cat(sep=" ")
    keywords = ["attendance", "assignment", "internal", "cgpa", "backlog", "stress", "participation"]

    causes = [w for w in keywords if w in words][:3]
    causes = [c.capitalize() for c in causes] or ["Multiple Academic Risks"]

    summary = f"""
📌 AI Intervention Summary

Critical: {len(critical)}
Warning: {len(warning)}

⚠ Major Causes Identified:
- {", ".join(causes)}

🎯 Suggested Academic Interventions:
• Weekly mentor monitoring for at-risk students
• Stricter attendance policy
• Academic support for low-performing subjects
"""

    return summary



@app.route("/history")
@login_required
def history():
    conn = get_db()
    cur = conn.cursor()
    if current_user()["role"] == "admin":
        cur.execute("SELECT * FROM analyses ORDER BY created_at DESC")
    else:
        cur.execute(
            "SELECT * FROM analyses WHERE user = ? ORDER BY created_at DESC",
            (current_user()["username"],),
        )
    rows = cur.fetchall()
    conn.close()
    return render_template("history.html", analyses=rows, user=current_user())


@app.route("/download_last")
@login_required
def download_last():
    global LAST_RESULT_DF
    if LAST_RESULT_DF is None:
        flash("No analysis available to download. Please run an analysis first.", "error")
        return redirect(url_for("index"))

    buf = BytesIO()
    LAST_RESULT_DF.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name="student_risk_analysis.csv",
    )
    
def load_all_issues():
    """Load all issue/behaviour responses from CSV into a dict keyed by reg_no."""
    issues = {}
    if not os.path.exists(ISSUES_FILE):
        return issues
    with open(ISSUES_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reg = str(row.get("reg_no", "")).strip()
            if reg:
                issues[reg] = row
    return issues


def build_issue_summary(issue_row: dict) -> str:
    """Combine multiple question answers into one readable summary text."""
    parts = []
    if issue_row.get("q_academic_reason"):
        parts.append("Academic: " + issue_row["q_academic_reason"])
    if issue_row.get("q_personal_reason"):
        parts.append("Personal: " + issue_row["q_personal_reason"])
    if issue_row.get("q_family_issue"):
        parts.append("Family: " + issue_row["q_family_issue"])
    if issue_row.get("q_finance_issue"):
        parts.append("Finance: " + issue_row["q_finance_issue"])
    if issue_row.get("q_health_issue"):
        parts.append("Health: " + issue_row["q_health_issue"])
    if issue_row.get("q_support_needed"):
        parts.append("Support needed: " + issue_row["q_support_needed"])

    return " | ".join(parts) if parts else ""

 
ISSUES_FILE = "critical_issues.csv"

def load_issue_for_student(reg_no):
    """Return dict of saved issue data for this reg_no, or None."""
    if not os.path.exists(ISSUES_FILE):
        return None
    with open(ISSUES_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("reg_no") == reg_no:
                return row
    return None

def save_issue_for_student(data):
    """Insert or update issue row for a student into CSV."""
    rows = []
    exists = os.path.exists(ISSUES_FILE)

    if exists:
        with open(ISSUES_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Replace existing row for same reg_no, if any
    updated = False
    for row in rows:
        if row.get("reg_no") == data["reg_no"]:
            row.update(data)
            updated = True
            break
    if not updated:
        rows.append(data)

    fieldnames = [
        "reg_no", "name", "dept", "semester", "risk_level",
        "q_academic_reason", "q_personal_reason", "q_family_issue",
        "q_finance_issue", "q_health_issue", "q_support_needed"
    ]
    with open(ISSUES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.route("/issues/<reg_no>", methods=["GET", "POST"])
def issues_form(reg_no):
    # We expect LAST_RESULT_DF to be filled from last analysis
    global LAST_RESULT_DF

    if LAST_RESULT_DF is None:
        return "No analysis data found. Please run an analysis first."

    # locate this student from the last result df
    df = LAST_RESULT_DF
    row = df[df["reg_no"].astype(str) == str(reg_no)]
    if row.empty:
        return f"No student found with reg_no {reg_no} in last analysis."

    student = row.iloc[0].to_dict()

    if request.method == "POST":
        data = {
            "reg_no": str(reg_no),
            "name": str(student.get("name", "")),
            "dept": str(student.get("dept", "")),
            "semester": str(student.get("semester", "")),
            "risk_level": str(student.get("risk_level", "")),
            "q_academic_reason": request.form.get("q_academic_reason", ""),
            "q_personal_reason": request.form.get("q_personal_reason", ""),
            "q_family_issue": request.form.get("q_family_issue", ""),
            "q_finance_issue": request.form.get("q_finance_issue", ""),
            "q_health_issue": request.form.get("q_health_issue", ""),
            "q_support_needed": request.form.get("q_support_needed", ""),
        }
        save_issue_for_student(data)
        return redirect(url_for("sample"))  # <-- we will adjust this name below

    saved = load_issue_for_student(str(reg_no))
    return render_template("issues_form.html", student=student, saved=saved)



if __name__ == "__main__":
    app.run(debug=True, port=5001)
