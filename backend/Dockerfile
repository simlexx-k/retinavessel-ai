FROM python:3.15.2-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/backend \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/tmp

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install "torch>=2.15" --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r /app/backend/requirements.txt

COPY src /app/src
COPY backend/retinavessel_api /app/backend/retinavessel_api

RUN groupadd --system --gid 10001 retinavessel \
    && useradd --system --uid 10001 --gid retinavessel --home-dir /nonexistent retinavessel \
    && mkdir -p /app/config /app/models \
    && chown -R retinavessel:retinavessel /app/config /app/models

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]

CMD ["uvicorn", "retinavessel_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
