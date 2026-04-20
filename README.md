# Resume Matcher Web App

This is a Streamlit web app built from your resume/job matching workflow.

## What it does
- lets users upload a resume and a job description
- scores the match using your trained scikit-learn model
- extracts matched and missing skills using `skills.txt`
- generates feedback with Groq if `GROQ_API_KEY` is set
- falls back to rule-based feedback if Groq is not configured





