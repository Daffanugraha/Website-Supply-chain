FROM python:3.11-slim@sha256:0c55578f585984aff90e1b0d1ac648e8b3e8b0c0b0f0c0d0e0f0c0d0e0f0c0d
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY . .
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]