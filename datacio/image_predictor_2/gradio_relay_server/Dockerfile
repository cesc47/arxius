# Dockerfile for Combined Gradio + Relay Server
FROM python:3.8-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the combined Gradio and Relay Server code
COPY gradio_relay.py .

# Expose port 8000 for FastAPI and 7860 for Gradio
EXPOSE 8000 7860

# Run the combined Gradio + Relay Server
CMD ["uvicorn", "gradio_relay:app", "--host", "0.0.0.0", "--port", "8000"]
