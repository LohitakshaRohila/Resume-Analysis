import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import docx2txt
import textstat
import io
import requests
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

# ---------------------------
# Setup spaCy & SkillNER
# ---------------------------
import spacy
from spacy.matcher import PhraseMatcher  # We will pass the class, not an instance
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Instead of creating an instance, pass the PhraseMatcher class itself.
matcher = PhraseMatcher

# Initialize SkillExtractor with the skills database and the matcher callable.
skill_extractor = SkillExtractor(nlp, SKILL_DB, matcher)

# ---------------------------
# (Optional) Load ESCO Skills Database dynamically
# ---------------------------
def load_ESCO_skills():
    try:
        url = "https://esco-occupations-skills.org/api/skills"
        response = requests.get(url)
        data = response.json()
        esco_skills = set(skill["preferredLabel"]["en"] for skill in data)
        return esco_skills
    except Exception as e:
        st.warning("ESCO skills could not be loaded. Using SkillNER only.")
        return set()

ESCO_SKILLS = load_ESCO_skills()  # Load ESCO skills dynamically

# ---------------------------
# Functions for text extraction
# ---------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    return docx2txt.process(docx_file)

# ---------------------------
# Enhanced Skill Extraction
# ---------------------------
def extract_skills(text):
    # Extract skills using SkillNER
    annotations = skill_extractor.annotate(text)
    extracted_skills = [skill['doc_node_value'] for skill in annotations["results"]["full_matches"]]

    # If ESCO database is available, cross-check extracted skills
    if ESCO_SKILLS:
        detected_skills = [skill for skill in extracted_skills if skill in ESCO_SKILLS]
    else:
        detected_skills = extracted_skills

    return list(set(detected_skills))

# ---------------------------
# Analyze resume text
# ---------------------------
def analyze_resume(text):
    readability_score = round(textstat.flesch_reading_ease(text), 2)
    gunning_fog = round(textstat.gunning_fog(text), 2)
    smog_index = round(textstat.smog_index(text), 2)
    detected_skills = extract_skills(text)

    return {
        "total_words": len(text.split()),
        "readability_score": readability_score,
        "gunning_fog": gunning_fog,
        "smog_index": smog_index,
        "detected_skills": detected_skills
    }

# ---------------------------
# Generate a formatted PDF
# ---------------------------
def generate_pdf_bytes(resume_text, detected_skills):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "AI-Formatted Resume")
    c.line(50, height - 55, width - 50, height - 55)

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, "Detected Skills:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(150, height - 80, ", ".join(detected_skills) if detected_skills else "None")

    c.setFont("Helvetica", 10)
    text_lines = simpleSplit(resume_text, "Helvetica", 10, width - 100)
    y = height - 110
    for line in text_lines:
        c.drawString(50, y, line)
        y -= 14
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("AI-Powered Resume Analyzer")
st.write("Upload your resume (PDF or DOCX) to get an AI-based analysis.")

uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_resume:
    if uploaded_resume.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_resume)
    else:
        resume_text = extract_text_from_docx(uploaded_resume)

    analysis = analyze_resume(resume_text)

    st.subheader("Resume Analysis")
    st.write(f"**Total Words:** {analysis['total_words']}")
    st.write(f"**Flesch Reading Ease:** {analysis['readability_score']}")
    st.write(f"**Gunning Fog Index:** {analysis['gunning_fog']}")
    st.write(f"**SMOG Index:** {analysis['smog_index']}")
    st.write(
        f"**Detected Skills:** {', '.join(analysis['detected_skills']) if analysis['detected_skills'] else 'None'}"
    )

    pdf_bytes = generate_pdf_bytes(resume_text, analysis['detected_skills'])
    st.download_button("Download Formatted Resume PDF", data=pdf_bytes, file_name="formatted_resume.pdf",
                       mime="application/pdf")
