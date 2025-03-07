# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (used by Hugging Face)
EXPOSE 7860

# Run Flask app
CMD ["python", "app.py"]
