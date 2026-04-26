#!/bin/bash
# ── Defense-AI Training Space Entry Point ──────────────────────────────────
# 1. Starts a minimal status HTTP server immediately (for health checks)
# 2. Runs GRPO training in the foreground
# 3. Uploads adapters to HF Hub when done
# 4. Status server shows live training progress

set -e
cd "$(dirname "$0")"

MODEL="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
EPISODES="${TRAIN_EPISODES:-100}"
EPOCHS="${TRAIN_EPOCHS:-3}"
USERNAME="${HF_USERNAME:-Bhaskar111}"
LOG_FILE="/app/logs/training.log"

mkdir -p /app/logs

echo "============================================================"
echo "  DEFENSE-AI — DEDICATED TRAINING SPACE"
echo "============================================================"
echo "  Model:    $MODEL"
echo "  Episodes: $EPISODES"
echo "  Epochs:   $EPOCHS"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU only')"
echo "============================================================"

# ── Start minimal status HTTP server in background ─────────────────────────
# This keeps HF Space health checks passing during training
python3 -c "
import http.server, threading, os, time

class StatusHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass  # silence access logs
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            log = ''
            try:
                with open('/app/logs/training.log', 'r') as f:
                    lines = f.readlines()
                    log = ''.join(lines[-50:])  # last 50 lines
            except: pass
            html = f'''<!DOCTYPE html>
<html><head><meta http-equiv=\"refresh\" content=\"10\">
<style>body{{background:#0a0a0a;color:#00ff88;font-family:monospace;padding:20px}}
pre{{background:#111;padding:20px;border-radius:8px;overflow-x:auto}}</style></head>
<body><h2>🎯 Defense-AI GRPO Training</h2>
<p>Auto-refreshes every 10 seconds</p>
<pre>{log if log else 'Training starting soon...'}</pre>
</body></html>'''
            self.wfile.write(html.encode())

server = http.server.HTTPServer(('0.0.0.0', 7860), StatusHandler)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
print('[Status] Server running on :7860')
# Keep alive until training script signals done
while not os.path.exists('/app/logs/training_done'):
    time.sleep(5)
print('[Status] Training complete — server staying alive.')
server.serve_forever()  # keep running after training
" &

STATUS_PID=$!
sleep 3
echo "[Entry] Status server started (PID $STATUS_PID)"

# ── Run GRPO Training ───────────────────────────────────────────────────────
echo "[Entry] Starting GRPO training..."
python train.py \
    --backend local \
    --model "$MODEL" \
    --episodes "$EPISODES" \
    --epochs "$EPOCHS" \
    --device cuda \
    --batch-size 1 \
    --lora-rank 16 \
    --checkpoint-dir checkpoints \
    2>&1 | tee "$LOG_FILE"

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "[Entry] Training succeeded!"

    # ── Upload checkpoints to HF Hub ────────────────────────────────────────
    if [ -n "$HF_TOKEN" ]; then
        echo "[Entry] Uploading adapters to HuggingFace Hub..."
        python3 -c "
from huggingface_hub import HfApi, create_repo
import os
api = HfApi(token=os.environ['HF_TOKEN'])
username = os.environ.get('HF_USERNAME', 'Bhaskar111')
for agent in ['radar', 'actor']:
    path = f'checkpoints/{agent}'
    if os.path.exists(path):
        repo_id = f'{username}/defense-rl-{agent}-adapter'
        create_repo(repo_id, repo_type='model', exist_ok=True, token=os.environ['HF_TOKEN'])
        api.upload_folder(folder_path=path, repo_id=repo_id, repo_type='model', token=os.environ['HF_TOKEN'])
        print(f'  Uploaded {agent} → https://huggingface.co/{repo_id}')
"
        echo "[Entry] ✅ Adapters uploaded to HuggingFace Hub!"
    else
        echo "[Entry] HF_TOKEN not set — skipping upload. Checkpoints in ./checkpoints/"
    fi
else
    echo "[Entry] ❌ Training failed with exit code $TRAIN_EXIT"
fi

# Signal status server that training is done
touch /app/logs/training_done
echo "[Entry] Training space done. Status page available at port 7860."

# Keep container alive so you can see logs
wait $STATUS_PID
