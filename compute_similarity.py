#!/usr/bin/env python3
"""
Compute image similarities using DINOv2 or CLIP embeddings and visualize ranked results.
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import ipdb


def load_model(model_type, device):
    """Load the specified model and processor."""
    if model_type == "dinov2":
        print("Loading DINOv2 model...")
        model_name = "facebook/dinov2-base"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    elif model_type == "clip":
        print("Loading CLIP model...")
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    print(f"Model loaded on {device}")
    return model, processor


def get_embedding(model, processor, image, model_type, device):
    """Get embedding for a single image."""
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        if model_type == "dinov2":
            outputs = model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        elif model_type == "clip":
            embedding = model.get_image_features(**inputs).cpu().numpy()

    return embedding[0]


def main():
    parser = argparse.ArgumentParser(description="Compute image similarities")
    parser.add_argument("--model", type=str, default="dinov2", choices=["dinov2", "clip"],
                        help="Model to use for embeddings (default: dinov2)")
    args = parser.parse_args()

    model_type = args.model
    model_display_name = "DINOv2" if model_type == "dinov2" else "CLIP"

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, processor = load_model(model_type, device)

    # List PNG files in data/actual_device_images/
    data_dir = Path("data/actual_device_images")
    # Order: target first, then one, two, three, four
    order = ["target.png", "one.png", "two.png", "three.png", "four.png"]
    png_files = [data_dir / name for name in order if (data_dir / name).exists()]
    print(f"\nFound {len(png_files)} PNG files:")
    for f in png_files:
        print(f"  - {f.name}")

    if len(png_files) < 2:
        print("Need at least 2 images to compute similarities!")
        return

    # Load images and compute embeddings
    print("\nComputing embeddings...")
    images = []
    embeddings = []

    for img_path in png_files:
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        ipdb.set_trace()
        embedding = get_embedding(model, processor, img, model_type, device)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute cosine similarities
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Similarity matrix (cosine similarity)
    similarity_matrix = normalized @ normalized.T
    print("\nSimilarity matrix:")
    print(similarity_matrix)

    # Convert to distance (1 - similarity) for ranking
    distance_matrix = 1 - similarity_matrix

    # Create visualization grid
    n_images = len(png_files)
    fig, axes = plt.subplots(n_images, n_images, figsize=(3 * n_images, 3 * n_images))

    # Handle single row case
    if n_images == 1:
        axes = np.array([[axes]])

    for i in range(n_images):
        # Get distances from image i to all others
        distances = distance_matrix[i]

        # Rank by similarity (excluding self)
        ranked_indices = np.argsort(distances)

        for rank, j in enumerate(ranked_indices):
            ax = axes[i, rank]
            ax.imshow(images[j])
            ax.axis("off")

            dist = distances[j]

            if i == j:
                title = f"Query: {png_files[i].stem}\n(self)"
            else:
                title = f"{png_files[j].stem}\ndist={dist:.3f}"

            ax.set_title(title, fontsize=9)

        # Add row label
        axes[i, 0].set_ylabel(f"Query {i+1}", fontsize=10, rotation=0, ha="right", va="center")

    plt.suptitle(f"Image Similarity Ranking ({model_display_name})\nEach row: query image ranked by similarity to others",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Create output directory
    output_dir = Path("results/actual_device_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"similarity_grid_{model_type}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved similarity grid to: {output_path}")
    plt.close()

    # Also save the similarity matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Cosine Similarity")

    # Add labels
    labels = [f.stem for f in png_files]
    ax.set_xticks(range(n_images))
    ax.set_yticks(range(n_images))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Add values in cells
    for i in range(n_images):
        for j in range(n_images):
            ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                   ha="center", va="center", color="white", fontsize=8)

    ax.set_title(f"{model_display_name} Cosine Similarity Matrix")
    plt.tight_layout()

    heatmap_path = output_dir / f"similarity_heatmap_{model_type}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"Saved similarity heatmap to: {heatmap_path}")
    plt.close()


if __name__ == "__main__":
    main()
