FROM python:3.11-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 pkg-config libfreetype6-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY qednet ./qednet
COPY api ./api
COPY configs ./configs
COPY scripts ./scripts

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
