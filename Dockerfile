FROM python:3.12-slim

# ffmpeg lets librosa decode mp3/m4a uploads
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY swara_backend.py .

# One worker keeps the in-memory job list consistent; threads let polling work during a conversion
CMD gunicorn swara_backend:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
