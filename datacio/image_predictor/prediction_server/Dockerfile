# Dockerfile
FROM python:3.9-slim

# Install system dependencies, including curl
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Download the model weights
RUN mkdir -p /root/.cache/torch/hub/checkpoints/ && \
    curl -o /root/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth \
    https://download.pytorch.org/models/convnext_base-6075fbad.pth

# Copy the model and script files into the container
COPY convnext_DEWB.pth .
COPY service.py .

# Expose the port
EXPOSE 5000

# Set the entry point
CMD ["python", "service.py"]
