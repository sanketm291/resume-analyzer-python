from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import tempfile
import re
import os
import requests

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(file_path):
    try:
        reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.lower()
    except:
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

TECH_HINTS = {
    "java","python","sql","aws","azure","gcp",
    "docker","kubernetes","spring","react","angular",
    "node","api","rest","microservices","linux"
}

def extract_jd_skills(text):
    text = text.lower()
    words = re.findall(r'\b[a-zA-Z0-9+#.]+\b', text)

    skills = set()

    for word in words:
        if word not in GENERIC_WORDS and len(word) > 2:
            skills.add(word)

    return list(skills)

# ---------------- OPENAI EMBEDDING ----------------
def get_embedding(text):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": text
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["data"][0]["embedding"]

# ---------------- COSINE SIMILARITY ----------------
def cosine_similarity(vec1, vec2):
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = sum(a*a for a in vec1) ** 0.5
    norm2 = sum(b*b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot / (norm1 * norm2)

# ---------------- MATCH ----------------
def hybrid_match_with_score(jd_skills, resume_text):
    results = []

    resume_embedding = get_embedding(resume_text)

    for skill in jd_skills:

        if skill in resume_text:
            score = 0.95
        else:
            skill_embedding = get_embedding(skill)
            score = cosine_similarity(skill_embedding, resume_embedding)

        results.append({
            "skill": skill,
            "score": round(score * 100, 2)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    matched = [r for r in results if r["score"] >= 60]
    missing = [r for r in results if r["score"] < 60]

    return matched, missing, results

# ---------------- API ----------------
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

        resume_text = extract_text_from_pdf(temp_path)

        jd_skills = extract_jd_skills(job_description or "")
        resume_skills = extract_jd_skills(resume_text or "")

        matched, missing, all_results = hybrid_match_with_score(jd_skills, resume_text)

        match_percentage = int((len(matched) / len(jd_skills)) * 100) if jd_skills else 0

        return {
            "match_percentage": match_percentage,
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "matched_skills": matched,
            "missing_skills": missing,
            "all_skills": all_results
        }

    except Exception as e:
        return {"error": str(e)}