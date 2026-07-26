# syntax=docker/dockerfile:1

FROM python:3.13-slim

# Curl is useful for the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached when proxy.py changes.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Application code. The API key is provided at runtime via AGENTROUTER_API_KEY
# (see docker-compose.yml), never baked into the image.
COPY proxy.py ./

# The proxy binds 127.0.0.1 by default (see proxy.py __main__); inside a
# container that would make it unreachable from the host, so override the
# bind host to 0.0.0.0 here while keeping the same port.
ENV PROXY_HOST=0.0.0.0 \
    PROXY_PORT=7187 \
    PYTHONUNBUFFERED=1

EXPOSE 7187

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PROXY_PORT}/v1/models || exit 1

# uvicorn.run() hardcodes host=127.0.0.1 inside proxy.py's __main__, so invoke
# uvicorn directly here to bind 0.0.0.0 (host 127.0.0.1 is unreachable from
# outside the container). --log-level info emits startup + per-request access
# logs so `docker compose logs -f` shows activity (warning level is silent
# when idle). PYTHONUNBUFFERED above (equivalent to python -u) flushes stdout.
# Use a shell form so ${PROXY_*} expand at runtime from the ENV above.
CMD ["sh", "-c", "exec python -m uvicorn proxy:app --host ${PROXY_HOST} --port ${PROXY_PORT} --log-level info"]
