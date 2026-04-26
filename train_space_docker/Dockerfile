FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Create user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install all training dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    trl>=0.11.0 \
    peft>=0.12.0 \
    accelerate>=0.30.0 \
    datasets>=2.20.0 \
    bitsandbytes>=0.43.0 \
    huggingface_hub \
    fastapi \
    uvicorn \
    pydantic>=2.0.0 \
    httpx \
    openai \
    torch

# Copy full codebase
COPY models.py           ./models.py
COPY train.py            ./train.py
COPY train_space.py      ./train_space.py
COPY defense_env/        ./defense_env/
COPY agents/             ./agents/
COPY training/           ./training/
COPY server/             ./server/
COPY openenv.yaml        ./openenv.yaml

# Init files
RUN touch defense_env/__init__.py agents/__init__.py training/__init__.py

# Dirs for checkpoints and data
RUN mkdir -p checkpoints/radar checkpoints/actor data logs

COPY train_entry.sh ./train_entry.sh
RUN chmod +x train_entry.sh

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Health check hits the status server
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["/bin/bash", "train_entry.sh"]
