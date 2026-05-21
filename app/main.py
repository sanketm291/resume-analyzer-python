from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import tempfile
import re
import os
import requests
import math

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HUGGINGFACE CONFIG ----------------
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

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
    "optimize","efficient","systems","services"
}

def extract_jd_skills(text):
    text = text.lower()

    words = re.findall(r'\b[a-zA-Z0-9+#.]+\b', text)

    skills = set()

    for word in words:
        if word not in GENERIC_WORDS and len(word) > 2:
            skills.add(word)

    return list(skills)

# ---------------- GET EMBEDDING ----------------
def get_embedding(text):

    payload = {
        "inputs": text
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload
    )

    result = response.json()

    print("HF RESPONSE:", result)

    # Error handling
    if isinstance(result, dict) and result.get("error"):
        raise Exception(result["error"])

    # Some HF models return nested arrays
    if isinstance(result[0], list):
        embedding = result[0]
    else:
        embedding = result

    return embedding

# ---------------- COSINE SIMILARITY ----------------
def cosine_similarity(vec1, vec2):

    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)

# ---------------- MATCHING ----------------
def hybrid_match_with_score(jd_skills, resume_text):

    results = []

    resume_embedding = get_embedding(resume_text)

    for skill in jd_skills:

        try:

            # Keyword exact match boost
            if skill in resume_text:
                score = 0.95

            else:
                skill_embedding = get_embedding(skill)

                score = cosine_similarity(
                    skill_embedding,
                    resume_embedding
                )

            results.append({
                "skill": skill,
                "score": round(score * 100, 2)
            })

        except Exception as e:
            print("MATCH ERROR:", e)

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    matched = [
        r for r in results
        if r["score"] >= 60
    ]

    missing = [
        r for r in results
        if r["score"] < 60
    ]

    return matched, missing, results

# ---------------- API ----------------
@app.get("/")
def home():
    return {
        "message": "Resume Analyzer API Running"
    }

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):

    try:

        with tempfile.NamedTemporaryFile(delete=False) as temp:

            temp.write(await file.read())

            temp.flush()

            temp_path = temp.name

        # Extract resume text
        resume_text = extract_text_from_pdf(temp_path)

        # Extract skills
        jd_skills = extract_jd_skills(job_description or "")
        resume_skills = extract_jd_skills(resume_text or "")

        # Match
        matched, missing, all_results = hybrid_match_with_score(
            jd_skills,
            resume_text
        )

        # Match percentage
        match_percentage = (
            int((len(matched) / len(jd_skills)) * 100)
            if jd_skills else 0
        )

        return {
            "match_percentage": match_percentage,
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "matched_skills": matched,
            "missing_skills": missing,
            "all_skills": all_results
        }

    except Exception as e:

        print("MAIN ERROR:", e)

        return {
            "error": str(e)
        }