FROM python:3.12-slim

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and models
COPY . .

# Expose port for Cloud Run
EXPOSE 8080

# Start the app
CMD ["python", "app.py"]