FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose port 8000 (HF Spaces expects port 7860, configurable)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
