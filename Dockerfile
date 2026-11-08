FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UA_HOST=0.0.0.0 \
    UA_PORT=8765

WORKDIR /app

RUN pip install --upgrade pip

COPY pyproject.toml README.md ./
COPY unified_agent ./unified_agent

RUN pip install .

EXPOSE 8765

CMD ["ua", "serve", "--port", "8765"]
