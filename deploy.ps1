# deploy.ps1 — Push Defense-RL to Hugging Face Spaces via Git
# Usage: .\deploy.ps1
# Run from inside the Defense-RL-main folder

# ── CONFIG — edit these ──────────────────────────────────────────
$HF_USERNAME = "Bhaskar111"
$SPACE_NAME  = "defense-ai"    # change to your new Space name
# ────────────────────────────────────────────────────────────────

$PROJECT_DIR = "C:\Users\Acer\Downloads\Defense-RL-main (1)\Defense-RL-main"
$CLONE_DIR   = "$PROJECT_DIR\hf-deploy-temp"
$SPACE_URL   = "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

# Ask for token securely
$TOKEN = Read-Host "Enter your HF Write token (hf_...)" -AsSecureString
$TOKEN_PLAIN = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($TOKEN)
)

$AUTH_URL = "https://$HF_USERNAME`:$TOKEN_PLAIN@huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

Write-Host "`n[1/4] Cloning Space repo..." -ForegroundColor Cyan
if (Test-Path $CLONE_DIR) { Remove-Item -Recurse -Force $CLONE_DIR }
git clone $AUTH_URL $CLONE_DIR

Write-Host "`n[2/4] Copying project files..." -ForegroundColor Cyan
$EXCLUDE = @("hf-deploy-temp", "defense-rl", "__pycache__", ".git", "checkpoints")
Get-ChildItem $PROJECT_DIR | Where-Object { $_.Name -notin $EXCLUDE } | ForEach-Object {
    Copy-Item $_.FullName "$CLONE_DIR\" -Recurse -Force
    Write-Host "  Copied: $($_.Name)"
}

Write-Host "`n[3/4] Committing..." -ForegroundColor Cyan
Set-Location $CLONE_DIR
git add .
git commit -m "Deploy Defense-RL environment server"

Write-Host "`n[4/4] Pushing to HF Spaces..." -ForegroundColor Cyan
git push

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "  DONE! Space is now building." -ForegroundColor Green
Write-Host "  Check: $SPACE_URL" -ForegroundColor Green
Write-Host "  Live (3-5 min): https://$HF_USERNAME-$SPACE_NAME.hf.space" -ForegroundColor Green
Write-Host "============================================`n" -ForegroundColor Green

# Cleanup temp folder
Set-Location $PROJECT_DIR
Remove-Item -Recurse -Force $CLONE_DIR
