import os
import io
import fitz  # PyMuPDF for PDF text extraction
import docx2txt
import textstat
import spacy
import torch
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

app = Flask(__name__, static_folder="../public", static_url_path="/")
CORS(app)

# Load spaCy model (for any additional processing if needed)
nlp = spacy.load("en_core_web_sm")

# Load the DeepSeek R1 1.5B model locally.
# Update MODEL_PATH if necessary.
MODEL_PATH = r"C:\Users\HP\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B\snapshots\ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained

# Ensure the temporary directory exists
os.makedirs("temp", exist_ok=True)


# ---------------------------
# Helper Functions for Text Extraction
# ---------------------------
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with fitz.open(pdf_file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_docx(docx_file_path):
    return docx2txt.process(docx_file_path)


# ---------------------------
# Skill Extraction using DeepSeek R1
# ---------------------------
def extract_skills(text):
    # Construct a prompt for the model to extract skills
    prompt = f"Extract key skills from the following resume:\n{text}\nSkills:"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=300)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to extract the list after "Skills:"
    if "Skills:" in generated_text:
        skills_str = generated_text.split("Skills:")[-1].strip()
    else:
        skills_str = generated_text.strip()
    # Split the skills string by commas and strip spaces
    skills_list = [skill.strip() for skill in skills_str.split(",") if skill.strip()]
    return list(set(skills_list))


# ---------------------------
# Resume Analysis Function
# ---------------------------
def analyze_resume(text):
    return {
        "total_words": len(text.split()),
        "readability_score": round(textstat.flesch_reading_ease(text), 2),
        "gunning_fog": round(textstat.gunning_fog(text), 2),
        "smog_index": round(textstat.smog_index(text), 2),
        "detected_skills": extract_skills(text),
        "resume_text": text  # Include full text for PDF generation
    }


# ---------------------------
# PDF Generation Function using ReportLab
# ---------------------------
def generate_pdf(resume_text, detected_skills):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header Section
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "AI-Formatted Resume")
    c.line(50, height - 55, width - 50, height - 55)

    # Skills Section
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, "Detected Skills:")
    c.setFont("Helvetica-Bold", 12)
    skills_text = ", ".join(detected_skills) if detected_skills else "None"
    wrapped_skills = simpleSplit(skills_text, "Helvetica-Bold", 12, width - 100)
    y = height - 100
    for line in wrapped_skills:
        c.drawString(50, y, line)
        y -= 16

    # Resume Text Section
    c.setFont("Helvetica", 10)
    y -= 20  # Add extra spacing
    wrapped_text = simpleSplit(resume_text, "Helvetica", 10, width - 100)
    for line in wrapped_text:
        if y < 50:  # Start a new page if the space is insufficient
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 14

    c.save()
    buffer.seek(0)
    return buffer


# ---------------------------
# Flask Routes / Endpoints
# ---------------------------

# Serve landing page (from public folder)
@app.route('/')
def index():
    return send_from_directory(app.static_folder, "index.html")


# POST /analyze: Process uploaded resume and return analysis results in JSON
@app.route('/analyze', methods=['POST'])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    temp_path = os.path.join("temp", file.filename)
    file.save(temp_path)

    if file.filename.lower().endswith(".pdf"):
        resume_text = extract_text_from_pdf(temp_path)
    elif file.filename.lower().endswith(".docx"):
        resume_text = extract_text_from_docx(temp_path)
    else:
        os.remove(temp_path)
        return jsonify({"error": "Invalid file format"}), 400

    os.remove(temp_path)  # Clean up temporary file
    analysis_result = analyze_resume(resume_text)
    return jsonify(analysis_result)


# POST /generate-pdf: Generate a formatted PDF resume from JSON data
@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    data = request.get_json()
    resume_text = data.get("resume_text", "")
    detected_skills = data.get("detected_skills", [])
    pdf_buffer = generate_pdf(resume_text, detected_skills)
    return send_file(pdf_buffer, as_attachment=True, download_name="formatted_resume.pdf", mimetype="application/pdf")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
