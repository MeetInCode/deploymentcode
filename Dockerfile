# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies - explicit copy for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY chatbot.py .
COPY ingest.py .

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "7860"]
