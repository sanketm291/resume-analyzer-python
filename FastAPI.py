from fastapi import FastAPI, UploadFile, File, Form
import PyPDF2
import os

app = FastAPI()

SKILLS_DB = [
    "python", "java", "sql", "spring boot",
    "machine learning", "docker", "aws", "react"
]

def extract_text_from_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    return text.lower()

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), jd: str = Form(...)):
    try:
        temp_file_path = "temp.pdf"

        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        resume_text = extract_text_from_pdf(temp_file_path)
        jd_text = jd.lower()

        resume_skills = [skill for skill in SKILLS_DB if skill in resume_text]
        jd_skills = [skill for skill in SKILLS_DB if skill in jd_text]

        matched_skills = list(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))

        match_percentage = 0
        if jd_skills:
            match_percentage = int((len(matched_skills) / len(jd_skills)) * 100)

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return {
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "match_percentage": match_percentage
        }

    except Exception as e:
        return {"error": str(e)}