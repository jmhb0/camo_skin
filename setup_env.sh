#!/bin/bash
# Setup UV environment for DINOv2 image similarity

set -e

echo "Creating UV virtual environment..."
uv venv .venv

echo "Activating environment..."
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install torch torchvision transformers pillow matplotlib numpy ipdb opencv-python scikit-learn tqdm

echo ""
echo "Environment setup complete!"
echo "To activate: source .venv/bin/activate"
echo "Then run: python compute_similarity.py"
