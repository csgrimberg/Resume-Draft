import streamlit as st
import joblib
import pandas as pd
import re
import ast

# -------------------------
# LOAD FILES
# -------------------------
MODEL_PATH = "resume_job_matcher_model.pkl"
TRAINING_PATH = "training_pairs_debug.csv"

model = joblib.load(MODEL_PATH)
training_df = pd.read_csv(TRAINING_PATH)

# -------------------------
# HELPERS
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s\+\#\.\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_match(resume_text, job_text):
    combined = "resume: " + clean_text(resume_text) + " job: " + clean_text(job_text)
    pred = model.predict([combined])[0]
    prob = model.predict_proba([combined])[0][1]
    return pred, prob

def safe_parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

def read_uploaded_file(uploaded_file):
    raw = uploaded_file.read()
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")

# Fix skill columns
if "matched_skills" in training_df.columns:
    training_df["matched_skills"] = training_df["matched_skills"].apply(safe_parse_list)
else:
    training_df["matched_skills"] = [[] for _ in range(len(training_df))]

if "missing_skills" in training_df.columns:
    training_df["missing_skills"] = training_df["missing_skills"].apply(safe_parse_list)
else:
    training_df["missing_skills"] = [[] for _ in range(len(training_df))]

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="AI Resume Matcher", page_icon="💼")

st.title("💼 AI Resume Matcher")
st.write("Upload your resume and a job description to get a match score and feedback.")

resume_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
job_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])

if resume_file and job_file:
    resume_text = read_uploaded_file(resume_file)
    job_text = read_uploaded_file(job_file)

    pred, score = predict_match(resume_text, job_text)

    st.subheader(f"📊 Match Score: {round(score * 100, 2)}%")

    # -------------------------
    # Skill explanation
    # -------------------------
    if "similarity" in training_df.columns:
        training_df["score_diff"] = abs(training_df["similarity"] - score)
        closest = training_df.sort_values("score_diff").iloc[0]

        st.subheader("✅ Matched Skills")
        st.write(", ".join(closest["matched_skills"]) if closest["matched_skills"] else "None")

        st.subheader("❌ Missing Skills")
        st.write(", ".join(closest["missing_skills"][:10]) if closest["missing_skills"] else "None")

    # -------------------------
    # Feedback
    # -------------------------
    st.subheader("🧠 Feedback")

    if score > 0.7:
        st.success("Strong match! You are well aligned with this job.")
    elif score > 0.4:
        st.warning("Moderate match. Improve missing skills and tailor your resume.")
    else:
        st.error("Low match. You may need to significantly tailor your resume.")

    # -------------------------
    # Expandable raw text (nice UX)
    # -------------------------
    with st.expander("📄 View Resume Text"):
        st.write(resume_text[:2000])

    with st.expander("📄 View Job Description"):
        st.write(job_text[:2000])
