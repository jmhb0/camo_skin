# Image Similarity with DINOv2

Compute pairwise image similarities using DINOv2 embeddings.

## Setup

1. Run the setup script to create the environment and install dependencies:
   ```bash
   ./setup_env.sh
   ```

2. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

## Usage

Place your PNG images in the `data/` directory, then run:

```bash
python compute_similarity.py
```

## Output

The script generates:
- `similarity_grid.png` - Each row shows a query image with others ranked by similarity
- `similarity_heatmap.png` - Heatmap of the full similarity matrix
