FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# System deps (kept minimal)
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better build cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

EXPOSE 8000

# IMPORTANT: bind 0.0.0.0 so other containers can reach it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

