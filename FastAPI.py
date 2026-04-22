from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import tempfile

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root endpoint
@app.get("/")
def home():
    return {"message": "Resume Analyzer API is running"}

# Skills DB
SKILLS_DB = [
    "python", "java", "sql", "spring boot",
    "machine learning", "docker", "aws", "react"
]

# Extract text from PDF
def extract_text_from_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()

# ✅ UPDATED API (resume + job description)
@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        content = await file.read()
        temp.write(content)
        temp_path = temp.name

    # Extract resume text
    resume_text = extract_text_from_pdf(temp_path)

    # Lowercase JD
    jd_text = job_description.lower()

    # Extract skills from resume
    resume_skills = []
    for skill in SKILLS_DB:
        if skill in resume_text:
            resume_skills.append(skill)

    # Extract skills from JD
    jd_skills = []
    for skill in SKILLS_DB:
        if skill in jd_text:
            jd_skills.append(skill)

    # Match calculation
    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    match_percentage = 0
    if len(jd_skills) > 0:
        match_percentage = int((len(matched) / len(jd_skills)) * 100)

    return {
        "resume_skills": list(set(resume_skills)),
        "jd_skills": list(set(jd_skills)),
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": match_percentage
    }