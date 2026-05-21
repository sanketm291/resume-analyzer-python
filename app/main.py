from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import PyPDF2
import tempfile
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(file_path):

    try:
        reader = PyPDF2.PdfReader(file_path)

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text.lower()

    except Exception as e:
        print("PDF ERROR:", e)
        return ""

# ---------------- SKILL EXTRACTION ----------------
GENERIC_WORDS = {
    "teamwork","communication","problem","solving",
    "bachelor","master","degree","location",
    "india","bangalore","years","experience",
    "looking","strong","good","skills","skill",
    "using","develop","build","create","maintain",
    "optimize","efficient","systems","services",
    "knowledge","understanding","ability","working"
}

TECH_SKILLS = {
    "java","python","sql","mysql","postgresql",
    "spring","springboot","hibernate",
    "aws","docker","kubernetes","linux",
    "javascript","react","angular","nodejs",
    "html","css","git","rest","api",
    "microservices","redis","mongodb",
    "fastapi","flask","django","azure","gcp"
}

def extract_skills(text):

    text = text.lower()

    words = re.findall(r'\b[a-zA-Z0-9+#.]+\b', text)

    skills = set()

    for word in words:

        word = word.strip()

        if (
            word not in GENERIC_WORDS
            and len(word) > 2
        ):

            if (
                word in TECH_SKILLS
                or any(char.isdigit() for char in word)
            ):
                skills.add(word)

    return list(skills)

# ---------------- TF-IDF MATCHING ----------------
def calculate_similarity(jd_text, resume_text):

    documents = [jd_text, resume_text]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)

# ---------------- SKILL MATCHING ----------------
def match_skills(jd_skills, resume_skills):

    matched = []
    missing = []

    resume_skill_set = set(
        skill.lower() for skill in resume_skills
    )

    for skill in jd_skills:

        if skill.lower() in resume_skill_set:

            matched.append({
                "skill": skill,
                "score": 95
            })

        else:

            missing.append({
                "skill": skill,
                "score": 30
            })

    return matched, missing

# ---------------- HOME ----------------
@app.get("/")
def home():

    return {
        "message": "Resume Analyzer API Running"
    }

# ---------------- MAIN API ----------------
@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):

    try:

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp:

            temp.write(await file.read())

            temp.flush()

            temp_path = temp.name

        # Extract resume text
        resume_text = extract_text_from_pdf(temp_path)

        # Extract skills
        jd_skills = extract_skills(job_description or "")
        resume_skills = extract_skills(resume_text or "")

        # Calculate similarity
        match_percentage = calculate_similarity(
            job_description,
            resume_text
        )

        # Match skills
        matched_skills, missing_skills = match_skills(
            jd_skills,
            resume_skills
        )

        return {

            "match_percentage": match_percentage,

            "resume_skills": resume_skills,

            "jd_skills": jd_skills,

            "matched_skills": matched_skills,

            "missing_skills": missing_skills
        }

    except Exception as e:

        print("MAIN ERROR:", e)

        return {
            "error": str(e)
        }