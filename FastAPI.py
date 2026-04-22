from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import tempfile

app = FastAPI()

# Enable CORS (important for Java/frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint 
@app.get("/")
def home():
    return {"message": "Resume Analyzer API is running"}

# Skills database
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

# Main API endpoint
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        content = await file.read()
        temp.write(content)
        temp_path = temp.name

    # Extract text from PDF
    text = extract_text_from_pdf(temp_path)

    # Match skills
    extracted_skills = []
    for skill in SKILLS_DB:
        if skill in text:
            extracted_skills.append(skill)

    return {
        "skills": list(set(extracted_skills))
    }