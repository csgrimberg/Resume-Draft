# Resume Matcher Web App

This is a Streamlit web app built from your resume/job matching workflow.

## What it does
- lets users upload a resume and a job description
- scores the match using your trained scikit-learn model
- extracts matched and missing skills using `skills.txt`
- generates feedback with Groq if `GROQ_API_KEY` is set
- falls back to rule-based feedback if Groq is not configured

## Required files
Put these files in the same folder as `app.py`:
- `resume_job_matcher_model.pkl`
- `training_pairs_debug.csv`
- `skills.txt`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional Groq setup
Mac/Linux:
```bash
export GROQ_API_KEY="your_real_key_here"
streamlit run app.py
```

Windows PowerShell:
```powershell
$env:GROQ_API_KEY="your_real_key_here"
streamlit run app.py
```

## Important fixes from your original script
- removed the hard-coded Groq key from the app
- added the missing `ast` import needed to parse list columns from CSV
- separated inference from training so the app does not retrain every time it starts
- added a fallback feedback mode so the app still works without Groq

## Recommended folder structure
```text
resume_matcher_app/
├── app.py
├── requirements.txt
├── README.md
├── resume_job_matcher_model.pkl
├── training_pairs_debug.csv
└── skills.txt
```
