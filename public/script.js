let analysisData = {};
let expandedSections = new Set();

function createTruncatedText(text, sectionId, maxLength = 300) {
  if (!text || text.length <= maxLength) {
    return `<p id="${sectionId}">${text || "No information found."}</p>`;
  }
  const truncated = text.slice(0, maxLength) + "...";
  return `
    <p id="${sectionId}">${truncated}</p>
    <button onclick="toggleReadMore('${sectionId}')"
      id="${sectionId}-btn" class="read-more-btn">Read more</button>
  `;
}

function toggleReadMore(sectionId) {
  const p = document.getElementById(sectionId);
  const btn = document.getElementById(sectionId + "-btn");
  if (expandedSections.has(sectionId)) {
    // Collapse
    p.textContent = analysisData[sectionId].slice(0, 300) + "...";
    btn.textContent = "Read more";
    expandedSections.delete(sectionId);
  } else {
    // Expand
    p.textContent = analysisData[sectionId] || "No information found.";
    btn.textContent = "Show less";
    expandedSections.add(sectionId);
  }
}

async function analyzeResume() {
  const input = document.getElementById("resumeInput");
  const analyzeBtn = document.querySelector('button[onclick="analyzeResume()"]');
  if (input.files.length === 0) {
    alert("Please select a resume file.");
    return;
  }

  analyzeBtn.disabled = true;
  const resultDiv = document.getElementById("analysisResult");
  resultDiv.innerHTML = `<div class="spinner"></div><p>Analyzing... ⏳</p>`;
  document.getElementById("downloadBtn").style.display = "none";

  const formData = new FormData();
  formData.append("resume", input.files[0]);

  try {
    const response = await fetch("/analyze", { method: "POST", body: formData });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Failed to analyze resume");
    }
    const data = await response.json();
    analysisData = data;

    // Reset expanded sections state
    expandedSections.clear();

    resultDiv.innerHTML = `
      <h2>Analysis Results</h2>
      <p><strong>Total Words:</strong> ${data.total_words}</p>
      <p><strong>Flesch Reading Ease:</strong> ${data.readability_score}</p>
      <p><strong>Gunning Fog Index:</strong> ${data.gunning_fog}</p>
      <p><strong>SMOG Index:</strong> ${data.smog_index}</p>

      <h3>Summary</h3>
      ${createTruncatedText(data.summary || "", "summary")}

      <h3>Skills</h3>
      <p>${data.detected_skills.length ? data.detected_skills.join(", ") : "No skills detected."}</p>

      <h3>Education</h3>
      ${createTruncatedText(data.education || "", "education")}

      <h3>Experience</h3>
      ${createTruncatedText(data.experience || "", "experience")}

      <h3>Certifications</h3>
      ${createTruncatedText(data.certifications || "", "certifications")}
    `;

    document.getElementById("downloadBtn").style.display = "inline-block";
  } catch (error) {
    resultDiv.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
    console.error("Analyze error:", error);
  } finally {
    analyzeBtn.disabled = false;
  }
}

async function downloadPDF() {
  if (!analysisData.resume_text) {
    alert("No resume data to generate PDF.");
    return;
  }

  try {
    const response = await fetch("/generate-pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        resume_text: analysisData.resume_text,
        detected_skills: analysisData.detected_skills,
        education: analysisData.education,
        experience: analysisData.experience,
        certifications: analysisData.certifications,
        summary: analysisData.summary,
      }),
    });

    if (!response.ok) throw new Error("Failed to generate PDF");

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "formatted_resume.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    alert("Error generating PDF: " + error.message);
    console.error("PDF generation error:", error);
  }
}
