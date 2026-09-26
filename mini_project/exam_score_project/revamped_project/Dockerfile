# One final image runs the FastAPI prediction service and its Streamlit client.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    API_URL=http://localhost:8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY artifacts/ artifacts/
COPY streamlit_app.py .
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
