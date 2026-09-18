# QED-Net 2.0 — backend image (Python 3.11 quantum/ML core + FastAPI)
FROM python:3.11-slim

WORKDIR /app

# system deps for numpy/scipy/xgboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# application code
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/raw/ ./data/raw/

ENV PYTHONPATH=/app/backend \
    PYTHONUNBUFFERED=1

WORKDIR /app
EXPOSE 8000

# default: FastAPI service (frontend runs in its own image/serve)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
