# ─── Defense-AI — OpenEnv Environment ───────────────────────────────────────
# HuggingFace Spaces compatible image.
# Exposes the FastAPI server on port 7860 (HF Spaces default).
#
# Build:  docker build -t defense-ai .
# Run:    docker run -p 7860:7860 defense-ai
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# HF Spaces requires port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir httpx

# Copy application source
COPY models.py           ./models.py
COPY openenv.yaml        ./openenv.yaml
COPY agent.py            ./agent.py
COPY inference.py        ./inference.py
COPY defense_env/        ./defense_env/
COPY server/             ./server/

# Create __init__.py files
RUN touch defense_env/__init__.py

# Switch to non-root
RUN chown -R appuser:appuser /app
USER appuser

# Expose HF Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start server from the /app directory so all imports resolve
CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1"]
