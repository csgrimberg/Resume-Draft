import streamlit as st
import joblib
import pandas as pd
import re
import os
import ast

# -------------------------
# LOAD FILES
# -------------------------
MODEL_PATH = "resume_job_matcher_model.pkl"
TRAINING_PATH = "training_pairs_debug.csv"
SKILLS_PATH = "skills (1).txt"

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
    try:
        return ast.literal_eval(x)
    except:
        return []

training_df["matched_skills"] = training_df["matched_skills"].apply(safe_parse_list)
training_df["missing_skills"] = training_df["missing_skills"].apply(safe_parse_list)

# -------------------------
# UI
# -------------------------
st.title("💼 AI Resume Matcher")

st.write("Upload your resume and a job description to get a match score and feedback.")

resume_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
job_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])

if resume_file and job_file:
    resume_text = resume_file.read().decode("utf-8")
    job_text = job_file.read().decode("utf-8")

    pred, score = predict_match(resume_text, job_text)

    st.subheader(f"📊 Match Score: {round(score * 100, 2)}%")

    # Find closest example for explanation
    training_df["score_diff"] = abs(training_df["similarity"] - score)
    closest = training_df.sort_values("score_diff").iloc[0]

    st.subheader("✅ Matched Skills")
    st.write(", ".join(closest["matched_skills"]) or "None")

    st.subheader("❌ Missing Skills")
    st.write(", ".join(closest["missing_skills"][:10]) or "None")

    st.subheader("🧠 Basic Feedback")
    if score > 0.7:
        st.success("Strong match! You are well aligned with this job.")
    elif score > 0.4:
        st.warning("Moderate match. Improve missing skills to increase chances.")
    else:
        st.error("Low match. Consider tailoring your resume significantly.")
