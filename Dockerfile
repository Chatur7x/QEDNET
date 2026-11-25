FROM node:22-bookworm-slim AS web-build

WORKDIR /app
COPY package.json bun.lock ./
RUN npm install --global bun@1.4.4 && bun install --frozen-lockfile
COPY . .
RUN bun run build

FROM node:22-bookworm-slim AS runtime
ENV NODE_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QEDNET_WORKSPACE_ROOT=/app \
    QEDNET_PYTHON=/opt/qednet-venv/bin/python \
    PYTHONPATH=/app/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/qednet-venv

WORKDIR /app
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY data/ ./data/
RUN /opt/qednet-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/qednet-venv/bin/pip install --no-cache-dir './backend[api]'

COPY --from=web-build /app/.next/standalone/ ./
COPY --from=web-build /app/.next/static/ ./.next/static/
COPY --from=web-build /app/public/ ./public/
COPY scripts/start-qednet.sh ./scripts/start-qednet.sh

EXPOSE 10000
CMD ["bash", "/app/scripts/start-qednet.sh"]
