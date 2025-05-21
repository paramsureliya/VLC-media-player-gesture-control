# Use official Python base image (slim version is smaller)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt (create this file with your dependencies)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and model files
COPY app.py .
COPY lambda_handler.py .
COPY hand_gesture_model1.h5 .
# Copy any other data files you need at runtime:
# COPY data/ ./data/

# Expose port for local testing (optional)
EXPOSE 8000

# Default command (for local testing)
CMD ["python", "app.py"]
