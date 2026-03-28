"""
AI issue responder for wsl-benchmark GitHub issues.
Pattern-matches common problems and posts targeted help.
"""
import os
import re
import sys
import json
try:
    import requests
except ImportError:
    print("requests not installed")
    sys.exit(0)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = os.environ.get("REPO", "")
ISSUE_NUMBER = os.environ.get("ISSUE_NUMBER", "")
ISSUE_TITLE = os.environ.get("ISSUE_TITLE", "")
ISSUE_BODY = os.environ.get("ISSUE_BODY", "")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

PATTERNS = [
    {
        "name": "no_torch",
        "match": ["no module named torch", "torch not installed", "torch not found", "importerror: no module"],
        "response": (
            "## Torch Not Installed\n\n"
            "wsl-benchmark works without PyTorch (falls back to numpy), but for GPU benchmarking you need torch:\n\n"
            "```bash\n"
            "# CPU-only (always works):\n"
            "python -m wsl_benchmark --ops matmul --json\n\n"
            "# Install torch first for GPU support:\n"
            "pip install torch  # CUDA\n"
            "pip install torch-directml  # Windows AMD/Intel (DirectML)\n"
            "pip install torch  # ROCm -- use AMD ROCm-enabled wheel\n"
            "```\n"
        ),
    },
    {
        "name": "hsa_override",
        "match": ["hsa_override", "gfx1010", "rx 5700", "rx5700", "not supported gpu", "unsupported gpu"],
        "response": (
            "## AMD RX 5700 XT / gfx1010 Fix\n\n"
            "wsl-benchmark automatically sets HSA_OVERRIDE_GFX_VERSION=10.3.0 before importing torch, "
            "which is required for RX 5700 XT on ROCm. If it still fails:\n\n"
            "```bash\n"
            "# Set manually before running:\n"
            "export HSA_OVERRIDE_GFX_VERSION=10.3.0\n"
            "python -m wsl_benchmark\n\n"
            "# On Windows (PowerShell):\n"
            "\\$env:HSA_OVERRIDE_GFX_VERSION='10.3.0'\n"
            "python -m wsl_benchmark\n"
            "```\n\n"
            "Also see: gpu-doctor for automatic GPU detection and env setup.\n"
        ),
    },
    {
        "name": "directml",
        "match": ["directml", "torch_directml", "torch-directml", "windows amd", "windows gpu"],
        "response": (
            "## DirectML (Windows AMD/Intel/NVIDIA)\n\n"
            "For GPU benchmarking on Windows without ROCm or CUDA, use DirectML:\n\n"
            "```bash\n"
            "pip install torch-directml\n"
            "python -m wsl_benchmark\n"
            "```\n\n"
            "wsl-benchmark auto-detects DirectML if torch_directml is installed.\n"
            "You do NOT need to install CUDA or ROCm on Windows.\n"
        ),
    },
    {
        "name": "out_of_memory",
        "match": ["out of memory", "oom", "cuda out of memory", "not enough memory", "allocationerror"],
        "response": (
            "## Out of Memory\n\n"
            "Reduce the matrix size with --size:\n\n"
            "```bash\n"
            "python -m wsl_benchmark --size 512   # smaller (uses ~1 GB VRAM)\n"
            "python -m wsl_benchmark --size 256   # minimal\n"
            "python -m wsl_benchmark --size 128   # very safe\n"
            "```\n\n"
            "Default size is 1024, which needs ~2-4 GB VRAM for matmul.\n"
            "For 8 GB VRAM (RX 5700 XT), --size 2048 is usually safe.\n"
        ),
    },
    {
        "name": "wsl_no_space",
        "match": ["no space left", "disk full", "wsl", "ext4", "vhdx"],
        "response": (
            "## WSL2 Disk Full\n\n"
            "If you running wsl-benchmark inside WSL2 and hit disk space errors:\n\n"
            "```bash\n"
            "# Check free space:\n"
            "df -h /\n\n"
            "# Fix with wsl-disk-doctor:\n"
            "# https://github.com/ChharithOeun/wsl-disk-doctor\n"
            "# Double-click FIX-WSL-DISK.bat on Windows\n"
            "```\n"
        ),
    },
    {
        "name": "install_error",
        "match": ["pip install", "failed to install", "error: could not", "setup.py", "build error", "externally-managed"],
        "response": (
            "## Installation Error\n\n"
            "```bash\n"
            "# Standard install:\n"
            "pip install wsl-benchmark\n\n"
            "# If you get externally-managed-environment on Linux:\n"
            "pip install wsl-benchmark --break-system-packages\n\n"
            "# Or use a virtual environment:\n"
            "python -m venv venv\n"
            "source venv/bin/activate  # Linux/macOS\n"
            "venv\\Scripts\\activate  # Windows\n"
            "pip install wsl-benchmark\n"
            "```\n"
        ),
    },
]

BOT_TAG = "<!-- wsl-benchmark-bot -->"

def already_responded(issue_number):
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        return False
    for comment in resp.json():
        if BOT_TAG in comment.get("body", ""):
            return True
    return False

def post_comment(issue_number, body):
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    data = {"body": body + f"\n\n{BOT_TAG}"}
    resp = requests.post(url, headers=HEADERS, json=data, timeout=10)
    return resp.status_code == 201

def find_response(title, body):
    text = (title + " " + body).lower()
    for pattern in PATTERNS:
        if any(kw in text for kw in pattern["match"]):
            return pattern["response"]
    return None

def main():
    if not all([GITHUB_TOKEN, REPO, ISSUE_NUMBER]):
        print("Missing env vars")
        sys.exit(0)
    if already_responded(ISSUE_NUMBER):
        print(f"Already responded to issue #{ISSUE_NUMBER}")
        sys.exit(0)
    response = find_response(ISSUE_TITLE, ISSUE_BODY)
    if response:
        ok = post_comment(ISSUE_NUMBER, response)
        print(f"Posted response to #{ISSUE_NUMBER}: {ok}")
    else:
        print(f"No matching pattern for issue #{ISSUE_NUMBER}")

if __name__ == "__main__":
    main()
