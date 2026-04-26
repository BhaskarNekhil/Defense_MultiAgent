"""
deploy_to_hf.py — One-shot HF Space deployment
Reads token from HF_TOKEN environment variable.

Usage (PowerShell):
    $env:HF_TOKEN = "hf_your_token_here"
    python deploy_to_hf.py
"""
import os, sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
HF_USERNAME  = "Bhaskar111"
SPACE_NAME   = "defense-ai"
REPO_ID      = f"{HF_USERNAME}/{SPACE_NAME}"
PROJECT_ROOT = Path(__file__).parent
EXCLUDE_DIRS  = {"__pycache__", ".git", "defense-rl", "checkpoints", ".gemini", "hf-deploy-temp"}
EXCLUDE_FILES = {"deploy_to_hf.py", "deploy.ps1", "uv.lock"}
# ─────────────────────────────────────────────────────────────────

# ── Get token from env var ────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set.")
    print("")
    print("  Run this in PowerShell first:")
    print('  $env:HF_TOKEN = "hf_your_token_here"')
    print("  Then run:  python deploy_to_hf.py")
    sys.exit(1)

print(f"Token loaded: {HF_TOKEN[:8]}{'*' * (len(HF_TOKEN)-8)}")

# ── Install huggingface_hub if missing ────────────────────────────
try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("[Setup] Installing huggingface_hub...")
    os.system(f'"{sys.executable}" -m pip install -q huggingface_hub')
    from huggingface_hub import HfApi, create_repo

api = HfApi(token=HF_TOKEN)

# ── Step 1: Create the Space ──────────────────────────────────────
print(f"\n[1/3] Creating Space: {REPO_ID}")
try:
    url = create_repo(
        repo_id   = REPO_ID,
        repo_type = "space",
        space_sdk = "docker",
        token     = HF_TOKEN,
        exist_ok  = True,
        private   = False,
    )
    print(f"      OK -> {url}")
except Exception as e:
    print(f"      Note: {e}")

# ── Step 2: Collect files ─────────────────────────────────────────
print(f"\n[2/3] Collecting files from {PROJECT_ROOT}")
files = []
for path in PROJECT_ROOT.rglob("*"):
    if path.is_dir():
        continue
    rel = path.relative_to(PROJECT_ROOT)
    rel_parts = set(rel.parts[:-1])
    if rel_parts & EXCLUDE_DIRS:
        continue
    if "__pycache__" in str(path):
        continue
    if path.name in EXCLUDE_FILES:
        continue
    files.append(path)
print(f"      Found {len(files)} files")

# ── Step 3: Upload all files as a single commit ───────────────────
print(f"\n[3/3] Uploading to {REPO_ID} ...")

from huggingface_hub import CommitOperationAdd

operations = []
for path in files:
    rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    print(f"  + {rel}")
    operations.append(
        CommitOperationAdd(
            path_in_repo = rel,
            path_or_fileobj = str(path),
        )
    )

try:
    api.create_commit(
        repo_id    = REPO_ID,
        repo_type  = "space",
        operations = operations,
        commit_message = "Deploy Defense-RL environment server",
        token      = HF_TOKEN,
    )
    print("\nUpload successful!")
except Exception as e:
    print(f"\nERROR during upload: {e}")
    sys.exit(1)

# ── Done ──────────────────────────────────────────────────────────
live = f"https://{HF_USERNAME}-{SPACE_NAME}.hf.space"
print("\n" + "="*55)
print("  DEPLOYMENT COMPLETE")
print("="*55)
print(f"  Space    : https://huggingface.co/spaces/{REPO_ID}")
print(f"  Live URL : {live}   (ready in ~5 min)")
print(f"  API Docs : {live}/docs")
print(f"  Health   : {live}/health")
print("="*55 + "\n")
