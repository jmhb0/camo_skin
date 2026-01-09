# Image Similarity with DINOv2

Compute pairwise image similarities using DINOv2 or CLIP embeddings.

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

```bash
python compute_similarity.py [--model MODEL] [--dataset INDEX] [--layer LAYER]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--model` | `dinov2`, `clip` | `dinov2` | Embedding model |
| `--dataset` | `0`, `1` | `0` | Dataset index |
| `--layer` | `1-12` | final | DINOv2 layer to use |

### Layer Selection (DINOv2 only)

DINOv2 has 12 transformer layers. Different layers capture different features:

- **Early layers (1-4)**: More sensitive to color, texture, low-level patterns
- **Middle layers (5-8)**: Mix of low-level and semantic features
- **Late layers (9-12)**: More semantic, shape-focused, object identity

Use `--layer` to bias similarity toward specific feature types:

```bash
python compute_similarity.py --layer 2   # color/texture sensitive
python compute_similarity.py --layer 12  # semantic/shape focused
python compute_similarity.py             # default: final layer (most semantic)
```

### Examples

```bash
# Dataset 0 with default settings
python compute_similarity.py

# Dataset 1 with CLIP
python compute_similarity.py --dataset 1 --model clip

# Dataset 0 with early DINOv2 layer (more color sensitive)
python compute_similarity.py --layer 3
```

## Output

Results are saved to `results/<dataset_name>/`:
- `similarity_grid_<model>.png` - Each row shows a query image with others ranked by similarity
- `similarity_heatmap_<model>.png` - Heatmap of the full similarity matrix

When using `--layer`, output files include the layer number (e.g., `similarity_grid_dinov2_layer3.png`).

## Video Frame Trajectory Analysis

Visualize how video frames move through DINOv2 embedding space over time.

### Usage

```bash
python trajectory_frame_s.py <video_path> [options]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `video_path` | path | required | Main video file to analyze |
| `--layer` | `1-12` | final | DINOv2 layer to use |
| `--fps` | float | `1.0` | Frames per second to extract |
| `--reference-videos` | paths | none | Additional videos for PCA context |

### Examples

```bash
# Basic usage - extract 1 frame/sec, use final layer
python trajectory_frame_s.py data/paper_vids/vid1_41586_2025_9948_MOESM6_ESM.mp4

# Use early layer (more texture-sensitive)
python trajectory_frame_s.py data/paper_vids/vid1_41586_2025_9948_MOESM6_ESM.mp4 --layer 2

# Extract more frames
python trajectory_frame_s.py data/paper_vids/vid1_41586_2025_9948_MOESM6_ESM.mp4 --fps 5

# Compare against reference videos
python trajectory_frame_s.py data/paper_vids/vid1_41586_2025_9948_MOESM6_ESM.mp4 \
    --reference-videos data/paper_vids/vid2_41586_2025_9948_MOESM7_ESM.mp4 \
                       data/paper_vids/vid3_41586_2025_9948_MOESM8_ESM.mp4
```

### Batch Processing

Run analysis for all paper videos:

```bash
./scripts/run_trajectory_analysis.sh
```

### Output

Results are saved to `results/trajectory_frame_s/<video_name>/`:
- `trajectory_fps<N>.png` - Trajectory visualization
- `trajectory_fps<N>.txt` - Command used to generate it

Filename pattern: `trajectory_fps{N}[_layer{L}][_refs].{png,txt}`

The visualization shows:
- **Main video**: Connected trajectory with start (square) and end (triangle) markers
- **Reference videos**: Scattered X markers in different colors (if provided)
