FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PORT=7860
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face model
COPY download_hf_model.py .
RUN python download_hf_model.py

COPY . .

CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 flask_app:app