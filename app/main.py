from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import tempfile
import re
import spacy

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# CORS
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

    # 🔹 Context filtering
    lines = text.split("\n")
    context_keywords = [
        "skills","technologies","tools",
        "experience","stack","framework","proficient"
    ]

    relevant_lines = [
        line for line in lines
        if any(k in line for k in context_keywords)
    ]

    if not relevant_lines:
        relevant_lines = lines

    text = " ".join(relevant_lines)

    # 🔹 NLP
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            word = token.text.strip()

            if (
                word not in GENERIC_WORDS and
                len(word) > 2 and
                word.isalpha()
            ):
                tokens.append(word)

    # 🔹 Phrase building
    skills = set(tokens)

    for i in range(len(tokens) - 1):
        phrase = tokens[i] + " " + tokens[i + 1]
        if len(phrase) < 25:
            skills.add(phrase)

    # 🔹 Validation
    def is_valid(skill):
        return (
            any(char.isdigit() for char in skill) or
            skill in TECH_HINTS or
            len(skill) <= 15
        )

    final_skills = [s for s in skills if is_valid(s)]

    return list(set(final_skills))

# ---------------- MATCH WITH SCORE ----------------
def hybrid_match_with_score(jd_skills, resume_text):
    results = []

    resume_lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    if not resume_lines:
        resume_lines = [resume_text]

    resume_embeddings = model.encode(resume_lines)
    jd_embeddings = model.encode(jd_skills)

    for i, skill in enumerate(jd_skills):

        # keyword boost
        if skill in resume_text:
            score = 0.95
        else:
            scores = cosine_similarity([jd_embeddings[i]], resume_embeddings)[0]
            score = float(max(scores))

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