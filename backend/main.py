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

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ✅ Load DeepSeek R1 from Hugging Face
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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
    prompt = f"Extract key skills from the following resume:\n{text}\nSkills:"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ✅ Ensure model is properly used
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=300)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract skills from output
    skills_list = generated_text.split("Skills:")[-1].strip().split(
        ",") if "Skills:" in generated_text else generated_text.split(",")
    return list(set(skill.strip() for skill in skills_list if skill.strip()))


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
# Flask Server
# ---------------------------
app = Flask(__name__, static_folder="../public", static_url_path="/")
CORS(app)


@app.route('/')
def index():
    return send_from_directory(app.static_folder, "index.html")


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


# Start Flask Server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860)
