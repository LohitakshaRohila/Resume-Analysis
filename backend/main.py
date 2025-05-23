import os
import io
import json
import tempfile
import logging

import fitz  # PyMuPDF for PDF text extraction
import docx2txt
import pytesseract
from PIL import Image
import textstat
import spacy
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model (used if needed later)
nlp = spacy.load("en_core_web_sm")

# Load DeepSeek R1 from Hugging Face
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

app = Flask(__name__)
CORS(app)


# ---------------------------
# Text Extraction Helpers
# ---------------------------
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return ""


def extract_text_from_docx(docx_path):
    try:
        text = docx2txt.process(docx_path)
        return text.strip()
    except Exception as e:
        logger.warning(f"DOCX extraction failed: {e}")
        return ""


def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return ""


def ocr_fallback(file_path):
    """Try OCR fallback on PDF or image if normal extraction fails"""
    ext = file_path.lower().split('.')[-1]
    if ext == "pdf":
        # Extract images from PDF pages for OCR
        text = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                img_path = f"/tmp/page_{page_num}.png"
                pix.save(img_path)
                text += extract_text_from_image(img_path) + "\n"
                os.remove(img_path)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR fallback PDF failed: {e}")
            return ""
    else:
        # If not PDF, try OCR directly (e.g. scanned image resume)
        return extract_text_from_image(file_path)


# ---------------------------
# DeepSeek R1 Structured Extraction
# ---------------------------
def extract_structured_resume_info(text):
    prompt = (
        "Extract the following structured info from the resume:\n\n"
        "- Skills\n"
        "- Education\n"
        "- Experience\n"
        "- Certifications\n"
        "- Summary\n\n"
        "Format the output as JSON with keys: skills, education, experience, certifications, summary.\n\n"
        f"Resume text:\n{text}\n\nJSON:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract JSON substring from the output
    json_start = generated_text.find("{")
    json_end = generated_text.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        logger.warning("No JSON structure detected in model output")
        return {}

    json_str = generated_text[json_start:json_end]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        data = {}

    # Normalize data fields
    for key in ["skills", "education", "experience", "certifications", "summary"]:
        if key not in data:
            data[key] = [] if key == "skills" else ""

    # Ensure skills is a list of strings
    if isinstance(data["skills"], str):
        data["skills"] = [s.strip() for s in data["skills"].split(",") if s.strip()]

    return data


# ---------------------------
# Resume Analysis
# ---------------------------
def analyze_resume(text):
    total_words = len(text.split())
    readability = round(textstat.flesch_reading_ease(text), 2)
    gunning = round(textstat.gunning_fog(text), 2)
    smog = round(textstat.smog_index(text), 2)

    deepseek_data = extract_structured_resume_info(text)

    analysis = {
        "total_words": total_words,
        "readability_score": readability,
        "gunning_fog": gunning,
        "smog_index": smog,
        "detected_skills": deepseek_data.get("skills", []),
        "education": deepseek_data.get("education", ""),
        "experience": deepseek_data.get("experience", ""),
        "certifications": deepseek_data.get("certifications", ""),
        "summary": deepseek_data.get("summary", ""),
        "resume_text": text,
    }
    return analysis


# ---------------------------
# PDF Generation
# ---------------------------
def generate_pdf(data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin = 40
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, y, "Formatted Resume Analysis")
    y -= 40

    # Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Summary:")
    y -= 20
    c.setFont("Helvetica", 12)
    for line in simpleSplit(data.get("summary", ""), "Helvetica", 12, width - 2 * margin):
        c.drawString(margin, y, line)
        y -= 15

    y -= 10

    # Sections function
    def draw_section(title, content):
        nonlocal y
        if not content:
            return
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, title)
        y -= 20
        c.setFont("Helvetica", 12)
        if isinstance(content, list):
            for item in content:
                lines = simpleSplit(f"• {item}", "Helvetica", 12, width - 2 * margin)
                for line in lines:
                    c.drawString(margin + 10, y, line)
                    y -= 15
        else:
            lines = simpleSplit(content, "Helvetica", 12, width - 2 * margin)
            for line in lines:
                c.drawString(margin, y, line)
                y -= 15
        y -= 10

    draw_section("Skills", data.get("detected_skills", []))
    draw_section("Education", data.get("education", ""))
    draw_section("Experience", data.get("experience", ""))
    draw_section("Certifications", data.get("certifications", ""))

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/")
def index():
    return "Resume Analyzer API running."


@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Use tempfile for secure handling
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        file.save(tmp_file.name)
        temp_path = tmp_file.name

    # Extract text based on file type
    ext = file.filename.lower().split(".")[-1]
    if ext == "pdf":
        text = extract_text_from_pdf(temp_path)
        if not text.strip():
            logger.info("PDF text extraction empty, trying OCR fallback...")
            text = ocr_fallback(temp_path)
    elif ext == "docx":
        text = extract_text_from_docx(temp_path)
    else:
        os.remove(temp_path)
        return jsonify({"error": "Unsupported file format"}), 400

    os.remove(temp_path)

    if not text.strip():
        return jsonify({"error": "Failed to extract text from resume"}), 400

    analysis = analyze_resume(text)
    return jsonify(analysis)


@app.route("/generate-pdf", methods=["POST"])
def generate_pdf_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pdf_buffer = generate_pdf(data)
    return send_file(pdf_buffer, as_attachment=True, download_name="formatted_resume.pdf", mimetype="application/pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
