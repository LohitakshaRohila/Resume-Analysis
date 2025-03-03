let analysisData = {};

function analyzeResume() {
    const input = document.getElementById("resumeInput");
    if (input.files.length === 0) {
        alert("Please select a resume file.");
        return;
    }
    const formData = new FormData();
    formData.append("resume", input.files[0]);

    fetch("/analyze", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        analysisData = data;
        const resultDiv = document.getElementById("analysisResult");
        resultDiv.innerHTML = `
            <h2>Analysis Results</h2>
            <p><strong>Total Words:</strong> ${data.total_words}</p>
            <p><strong>Flesch Reading Ease:</strong> ${data.readability_score}</p>
            <p><strong>Gunning Fog Index:</strong> ${data.gunning_fog}</p>
            <p><strong>SMOG Index:</strong> ${data.smog_index}</p>
            <p><strong>Detected Skills:</strong> ${data.detected_skills.join(", ") || "None"}</p>
        `;
        document.getElementById("downloadBtn").style.display = "inline-block";
    })
    .catch(error => console.error("Error:", error));
}

function downloadPDF() {
    fetch("/generate-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            resume_text: analysisData.resume_text,
            detected_skills: analysisData.detected_skills
        })
    })
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "formatted_resume.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    })
    .catch(error => console.error("Error generating PDF:", error));
}
