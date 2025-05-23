import os
import io
import sys
import fitz  # PyMuPDF
import docx2txt
import textstat
import spacy
import torch
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit

PUBLIC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public'))

# OCR fallback
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # path inside Colab, usually pre-installed

# NLP and Model Setup
nlp = spacy.load("en_core_web_sm")
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# App Setup
app = Flask(__name__, static_folder="public", static_url_path="/")
CORS(app)
os.makedirs("temp", exist_ok=True)

# ------------- UTILS -------------
def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            pg_text = page.get_text()
            if not pg_text.strip():
                img = page.get_pixmap()
                image = Image.open(io.BytesIO(img.tobytes("png")))
                pg_text = pytesseract.image_to_string(image)
            text += pg_text
    return text

def extract_text_from_docx(path):
    return docx2txt.process(path)

def extract_skills(text):
    prompt = f"Extract key technical and soft skills from this resume:\n{text}\nSkills:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=300)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    skills = output.split("Skills:")[-1].strip().split(",")
    return list(set(skill.strip() for skill in skills if skill.strip()))

def analyze_resume(text):
    return {
        "total_words": len(text.split()),
        "readability_score": round(textstat.flesch_reading_ease(text), 2),
        "gunning_fog": round(textstat.gunning_fog(text), 2),
        "smog_index": round(textstat.smog_index(text), 2),
        "detected_skills": extract_skills(text),
        "resume_text": text
    }

def create_pdf(resume_text, detected_skills):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    lines = simpleSplit(resume_text, "Helvetica", 10, width - 100)
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Formatted Resume Summary")
    y -= 30

    c.setFont("Helvetica", 10)
    for line in lines:
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 12

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detected Skills:")
    y -= 15
    c.setFont("Helvetica", 10)
    for skill in detected_skills:
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(60, y, f"- {skill}")
        y -= 12

    c.save()
    buffer.seek(0)
    return buffer

# ------------- ROUTES -------------

@app.route('/')
def index():
    return send_from_directory(PUBLIC_FOLDER, "index.html")

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
        return jsonify({"error": "Unsupported file format"}), 400

    os.remove(temp_path)
    analysis = analyze_resume(resume_text)
    return jsonify(analysis)

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.get_json()
    pdf_buffer = create_pdf(data.get("resume_text", ""), data.get("detected_skills", []))
    return send_file(pdf_buffer, as_attachment=True, download_name="formatted_resume.pdf", mimetype='application/pdf')

# ------------- START SERVER -------------

if __name__ == '__main__':
    if 'google.colab' in sys.modules:
        from pyngrok import ngrok
        port = 7860
        public_url = ngrok.connect(port)
        print(f" * ngrok tunnel running at: {public_url}")
        app.run(port=port)
    else:
        app.run(host="0.0.0.0", port=7860)
