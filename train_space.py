"""
train_space.py — Training entrypoint for HF Spaces with GPU hardware.

When your Space is upgraded to T4/A10G, run this script via the Space terminal:
    python train_space.py

Or add it as a background job via the Space's app.py.
"""
import os
import sys
import subprocess

def run_training():
    """Run GRPO training using the GPU available in this Space."""
    
    model     = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    episodes  = int(os.environ.get("TRAIN_EPISODES", "100"))
    epochs    = int(os.environ.get("TRAIN_EPOCHS", "3"))
    hf_token  = os.environ.get("HF_TOKEN", "")
    username  = os.environ.get("HF_USERNAME", "Bhaskar111")

    print("=" * 55)
    print("  DEFENSE-RL — GRPO TRAINING ON HF SPACE GPU")
    print("=" * 55)
    print(f"  Model    : {model}")
    print(f"  Episodes : {episodes}")
    print(f"  Epochs   : {epochs}")
    print("=" * 55)

    # Check GPU
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                             "--format=csv,noheader"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print(f"\n  GPU: {result.stdout.strip()}")
    else:
        print("\n  WARNING: No GPU detected. Training will be very slow on CPU.")

    # Run GRPO training
    cmd = [
        sys.executable, "train.py",
        "--backend",    "local",
        "--model",      model,
        "--episodes",   str(episodes),
        "--epochs",     str(epochs),
        "--device",     "cuda",
        "--batch-size", "1",
        "--lora-rank",  "16",
        "--checkpoint-dir", "checkpoints",
    ]
    print(f"\n  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nERROR: Training failed.")
        sys.exit(1)

    # Push checkpoints to HF Hub
    if hf_token:
        print("\n  Uploading trained adapters to HF Hub...")
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=hf_token)
        for agent in ["radar", "actor"]:
            path = f"checkpoints/{agent}"
            if os.path.exists(path):
                repo_id = f"{username}/defense-rl-{agent}-adapter"
                create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
                api.upload_folder(folder_path=path, repo_id=repo_id,
                                  repo_type="model", token=hf_token)
                print(f"  Saved: https://huggingface.co/{repo_id}")
    else:
        print("\n  HF_TOKEN not set — checkpoints saved locally only.")
        print("  Add HF_TOKEN secret in Space settings to auto-upload.")

    print("\n  Training complete!")

if __name__ == "__main__":
    run_training()
