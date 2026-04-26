# Use official Python 3.10 on Debian Slim — TensorFlow CPU works perfectly here
FROM python:3.10-slim

# Prevents .pyc files and enables real-time logs in Render
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies TensorFlow needs
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Expose Flask port
EXPOSE 5050

# Download pre-trained models then start the Flask API
CMD python download_models.py && python app.py