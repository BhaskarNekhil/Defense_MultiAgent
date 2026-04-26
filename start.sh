#!/bin/bash
# Start the Defense-AI server from the project root
cd "$(dirname "$0")"

# ── Optional background GRPO training ────────────────────────────────────────
# Set RUN_TRAINING=true as a Space Variable (Settings → Variables) to trigger.
# Training runs in the background after server starts.
# Set back to false (or remove) after training completes to load adapters.
if [ "${RUN_TRAINING}" = "true" ]; then
    echo "[Space] RUN_TRAINING=true — launching GRPO training in background..."
    echo "[Space] Model: ${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
    echo "[Space] Episodes: ${TRAIN_EPISODES:-100} | Epochs: ${TRAIN_EPOCHS:-3}"

    # Wait 15s for the server to fully start and pass health checks first
    (sleep 15 && python train_space.py 2>&1 | tee /tmp/training.log) &

    echo "[Space] Training PID launched. Server starting now..."
else
    echo "[Space] RUN_TRAINING not set — starting server only (no training)."
fi

# ── Start the web server (always) ────────────────────────────────────────────
exec uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-7860}" --workers 1
