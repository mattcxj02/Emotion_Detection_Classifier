FROM python:3.12-slim

WORKDIR /app

# Install deps for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and models
COPY . .

# Expose port for Cloud Run
EXPOSE 8080

# Start the app
CMD ["python", "app.py"]