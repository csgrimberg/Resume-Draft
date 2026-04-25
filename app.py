import os
import re
import ast
import io
import joblib
import pandas as pd
import streamlit as st

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="💼",
    layout="wide"
)

# -------------------------
# THEME COLORS
# -------------------------
PRIMARY = "#FF2D2D"
SECONDARY = "#B30000"
BG = "#0E0E0E"
CARD = "#1A1A1A"
TEXT = "#FFFFFF"
MUTED = "#B3B3B3"
BORDER = "#2A2A2A"
SUCCESS = "#1DB954"
WARNING = "#FFB020"
DANGER = "#FF4B4B"

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    h1, h2, h3, h4 {{
        color: {PRIMARY};
    }}

    p, label, div, span {{
        color: {TEXT};
    }}

    .hero-card {{
        background: linear-gradient(135deg, #151515 0%, #1d0f0f 100%);
        border: 1px solid {BORDER};
        border-radius: 22px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 0 0 1px rgba(255,45,45,0.05);
    }}

    .info-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }}

    div[data-testid="stFileUploader"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 0.75rem;
    }}

    div[data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 0.75rem;
    }}

    .stButton > button {{
        background: linear-gradient(90deg, {PRIMARY} 0%, {SECONDARY} 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.1rem;
        font-weight: 700;
        width: 100%;
    }}

    .stButton > button:hover {{
        filter: brightness(1.08);
        color: white;
    }}

    div[data-testid="stExpander"] {{
        background: {CARD};
        border-radius: 14px;
        border: 1px solid {BORDER};
        overflow: hidden;
    }}

    textarea, input {{
        background-color: {CARD} !important;
        color: white !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}

    div[data-testid="stProgressBar"] > div > div > div > div {{
        background: linear-gradient(90deg, {PRIMARY} 0%, {SECONDARY} 100%);
    }}

    .section-label {{
        font-size: 0.95rem;
        font-weight: 700;
        color: {PRIMARY};
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .muted {{
        color: {MUTED};
        font-size: 0.95rem;
    }}

    .pill {{
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
        border: 1px solid {BORDER};
        background: #121212;
        color: white;
    }}

    .pill-match {{
        background: rgba(29, 185, 84, 0.12);
        border: 1px solid rgba(29, 185, 84, 0.35);
    }}

    .pill-missing {{
        background: rgba(255, 45, 45, 0.12);
        border: 1px solid rgba(255, 45, 45, 0.35);
    }}

    .result-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 1rem 1.15rem;
        margin-top: 1rem;
    }}

    .small-note {{
        font-size: 0.85rem;
        color: {MUTED};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# LOAD FILES
# -------------------------
MODEL_PATH = "resume_job_matcher_model.pkl"
TRAINING_PATH = "training_pairs_debug.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_training_data():
    return pd.read_csv(TRAINING_PATH)

model = load_model()
training_df = load_training_data()

# -------------------------
# GROQ
# -------------------------
def get_groq_client():
    api_key = None

    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    elif os.getenv("GROQ_API_KEY"):
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key or Groq is None:
        return None

    return Groq(api_key=api_key)

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
    return int(pred), float(prob)

def safe_parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        parsed = ast.literal_eval(x)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

def decode_text_bytes(raw):
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")

def read_txt_file(uploaded_file):
    raw = uploaded_file.read()
    return decode_text_bytes(raw)

def read_pdf_file(uploaded_file):
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is not installed. Add it to requirements.txt")

    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    pages = []

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n".join(pages).strip()

def read_docx_file(uploaded_file):
    if Document is None:
        raise ImportError("python-docx is not installed. Add it to requirements.txt")

    doc = Document(io.BytesIO(uploaded_file.read()))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()

def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return read_txt_file(uploaded_file)
    if file_name.endswith(".pdf"):
        return read_pdf_file(uploaded_file)
    if file_name.endswith(".docx"):
        return read_docx_file(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a TXT, PDF, or DOCX file.")

def get_closest_training_example(score):
    df = training_df.copy()
    if "similarity" not in df.columns:
        return None
    df["score_diff"] = (df["similarity"] - score).abs()
    return df.sort_values("score_diff").iloc[0]

def render_skill_pills(skills, pill_class):
    if not skills:
        st.write("None")
        return
    html = "".join([f'<span class="pill {pill_class}">{skill}</span>' for skill in skills])
    st.markdown(html, unsafe_allow_html=True)

def get_match_label(score):
    if score >= 0.75:
        return "Strong Match", SUCCESS
    if score >= 0.45:
        return "Moderate Match", WARNING
    return "Low Match", DANGER

def build_resume_rewrite_prompt(resume_text, job_text, score, matched_skills, missing_skills):
    matched = ", ".join(matched_skills[:15]) if matched_skills else "None identified"
    missing = ", ".join(missing_skills[:15]) if missing_skills else "None identified"

    return f"""
You are an expert resume writer and career coach.

Your job is to help improve resume content for a specific role.

Rules:
- Only use information already present in the resume.
- Do not invent employers, job titles, tools, metrics, dates, achievements, or experiences.
- Make bullet points stronger, clearer, and more aligned with the job.
- Keep the tone professional and concise.

Match Score: {round(score * 100, 2)}%
Matched Skills: {matched}
Missing Skills: {missing}

Resume:
{resume_text[:5000]}

Job Description:
{job_text[:5000]}

Return exactly these sections with markdown headings:

## Overall Fit
Write 2 concise sentences.

## Improved Resume Lines
Write 5 improved bullet points.
Each bullet must:
- start with a strong action verb
- be one line
- stay truthful to the resume
- reflect the job description where possible

## Keywords To Add If Truthful
List 8 keywords from the job description.

## Top Gaps
List the top 3 gaps between the resume and the role.
"""

def generate_ai_feedback(resume_text, job_text, score, matched_skills, missing_skills):
    client = get_groq_client()
    if client is None:
        return None

    prompt = build_resume_rewrite_prompt(
        resume_text=resume_text,
        job_text=job_text,
        score=score,
        matched_skills=matched_skills,
        missing_skills=missing_skills
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise resume rewriting assistant. You never invent experience."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI feedback could not be generated: {e}"

# -------------------------
# PREP DATA
# -------------------------
if "matched_skills" in training_df.columns:
    training_df["matched_skills"] = training_df["matched_skills"].apply(safe_parse_list)
else:
    training_df["matched_skills"] = [[] for _ in range(len(training_df))]

if "missing_skills" in training_df.columns:
    training_df["missing_skills"] = training_df["missing_skills"].apply(safe_parse_list)
else:
    training_df["missing_skills"] = [[] for _ in range(len(training_df))]

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom:0.4rem;">💼 AI Resume Matcher</h1>
        <p style="margin:0; font-size:1.05rem;">
            Match smarter. Apply better.
        </p>
        <p class="muted" style="margin-top:0.6rem;">
            Upload a resume and a job description to get a match score, skill insights, and AI-tailored resume bullet rewrites.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# INPUT SECTION
# -------------------------
left, right = st.columns(2)

with left:
    st.markdown('<div class="section-label">Resume Upload</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader(
        "Upload Resume",
        type=["txt", "pdf", "docx"],
        label_visibility="collapsed"
    )

with right:
    st.markdown('<div class="section-label">Job Description Upload</div>', unsafe_allow_html=True)
    job_file = st.file_uploader(
        "Upload Job Description",
        type=["txt", "pdf", "docx"],
        label_visibility="collapsed"
    )

# -------------------------
# MAIN ACTION
# -------------------------
if resume_file and job_file:
    try:
        resume_text = read_uploaded_file(resume_file)
        job_text = read_uploaded_file(job_file)

        if not resume_text.strip():
            st.error("The uploaded resume appears empty or could not be read.")
            st.stop()

        if not job_text.strip():
            st.error("The uploaded job description appears empty or could not be read.")
            st.stop()

    except Exception as e:
        st.error(f"File reading error: {e}")
        st.stop()

    pred, score = predict_match(resume_text, job_text)
    label, label_color = get_match_label(score)

    resume_skills, job_skills, matched_skills, missing_skills = compare_uploaded_skills(
        resume_text,
        job_text
    )
    st.markdown('<div class="section-label">Match Results</div>', unsafe_allow_html=True)
    st.progress(min(max(score, 0.0), 1.0))
    st.caption(f"Overall match confidence: {round(score * 100, 2)}%")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Match Score", f"{round(score * 100, 2)}%")
    with c2:
        st.metric("Prediction", label)
    with c3:
        st.metric("Matched Skills", f"{len(matched_skills)} / {len(job_skills)}")

    skill_col1, skill_col2 = st.columns(2)

    with skill_col1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("✅ Matched Skills")
        render_skill_pills(matched_skills[:15], "pill-match")
        st.markdown("</div>", unsafe_allow_html=True)

    with skill_col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("❌ Missing Skills")
        render_skill_pills(missing_skills[:15], "pill-missing")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("🧠 Quick Feedback")
    if score > 0.75:
        st.success("Strong alignment. This resume appears to fit the role well.")
    elif score > 0.45:
        st.warning("Decent alignment, but there is room to tailor the resume more directly to the job.")
    else:
        st.error("Lower alignment. The resume likely needs stronger tailoring for this position.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">AI Resume Rewrite</div>', unsafe_allow_html=True)
    ai_col1, ai_col2 = st.columns([2, 1])

    with ai_col1:
        generate_clicked = st.button("Generate AI Rewrite Suggestions")

    with ai_col2:
        groq_ready = get_groq_client() is not None
        st.markdown(
            f"<p class='small-note'>{'Groq connected' if groq_ready else 'Groq key not detected'}</p>",
            unsafe_allow_html=True
        )

    if generate_clicked:
        with st.spinner("Generating tailored resume improvements..."):
            ai_output = generate_ai_feedback(
                resume_text=resume_text,
                job_text=job_text,
                score=score,
                matched_skills=matched_skills,
                missing_skills=missing_skills
            )

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if ai_output:
            st.markdown(ai_output)
        else:
            st.info("Add a Groq API key in Streamlit secrets to enable AI-generated rewrites.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📄 View Uploaded Resume Text"):
        st.write(resume_text[:4000])

    with st.expander("📄 View Uploaded Job Description"):
        st.write(job_text[:4000])

else:
    st.markdown(
        """
        <div class="info-card">
            <h3 style="margin-top:0;">How to use</h3>
            <p style="margin-bottom:0.45rem;">1. Upload your resume as TXT, PDF, or DOCX.</p>
            <p style="margin-bottom:0.45rem;">2. Upload the job description as TXT, PDF, or DOCX.</p>
            <p style="margin-bottom:0.45rem;">3. Review your match score and skills insights.</p>
            <p style="margin-bottom:0;">4. Generate AI rewrite suggestions for stronger resume bullets.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
