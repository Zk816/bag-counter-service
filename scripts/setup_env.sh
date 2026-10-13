#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
venv/bin/pip install -r requirements-dev.txt -r requirements-train.txt
venv/bin/pip install -e .
echo "Activate with: source venv/bin/activate"
