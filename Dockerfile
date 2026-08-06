FROM python:3.12-slim

# libgomp1: lightgbm's OpenMP runtime dependency, not always pulled in by the wheel.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (defaults to 8080); exec form so uvicorn receives
# SIGTERM directly for clean shutdown instead of a wrapping shell swallowing it.
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
