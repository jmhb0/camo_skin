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


def get_embedding(model, processor, image, model_type, device, layer=None):
    """Get embedding for a single image.

    Args:
        layer: For DINOv2, which layer to use (1-12). None = final layer.
               Early layers (1-4) are more color/texture sensitive.
               Late layers (10-12) are more semantic/shape focused.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        if model_type == "dinov2":
            outputs = model(**inputs, output_hidden_states=True)
            if layer is not None:
                # Use specified layer (1-indexed, so layer 1 = hidden_states[1])
                embedding = outputs.hidden_states[layer][:, 0].cpu().numpy()
            else:
                # Use final layer CLS token
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        elif model_type == "clip":
            embedding = model.get_image_features(**inputs).cpu().numpy()

    return embedding[0]


def get_dataset_config(dataset_idx):
    """Get dataset configuration based on index."""
    if dataset_idx == 0:
        return {
            "name": "actual_device_images",
            "data_dir": Path("data/actual_device_images"),
            "output_dir": Path("results/actual_device_images"),
            "mode": "symmetric",  # All images compared to each other
            "image_order": ["target.png", "one.png", "two.png", "three.png", "four.png"],
        }
    elif dataset_idx == 1:
        return {
            "name": "actual_device_images_multi_target",
            "data_dir": Path("data/actual_device_images_multi_target"),
            "output_dir": Path("results/actual_device_images_multi_target"),
            "mode": "rows_vs_columns",  # Rows compared against columns
            "row_images": [
                "targets/target_one.png",
                "targets/target_two.png",
                "targets/target_two_v2.png",
                "targets/target_three.png",
                "targets/target_four.png",
                "targets/target_4_small.png",
            ],
            "column_images": ["four.png", "one.png", "two.png", "three.png"],
        }
    elif dataset_idx == 2:
        return {
            "name": "texture_v1_12_8_2025",
            "data_dir": Path("data/texture_v1_12_8_2025"),
            "output_dir": Path("results/texture_v1_12_8_2025"),
            "mode": "rows_vs_columns",
            "row_images": [
                "target/t1.jpg",
                "target/t2.jpg",
                "target/t3.jpg",
                "target/t3_l2.jpg",
                "target/t4.jpg",
                "target/t4_v2.jpg",
                "target/t5.jpg",
                "target/t6.jpg",
                "target/t7.jpg",
            ],
            "column_images": [
                "device/b1.jpg",
                "device/b2.jpg",
                "device/b3.jpg",
                "device/b4.jpg",
                "device/b5.jpg",
                "device/b6.jpg",
                "device/b7.jpg",
            ],
        }
    else:
        raise ValueError(f"Unknown dataset index: {dataset_idx}")


def compute_embeddings(image_paths, model, processor, model_type, device, layer=None):
    """Compute embeddings for a list of image paths."""
    images = []
    embeddings = []

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        embedding = get_embedding(model, processor, img, model_type, device, layer=layer)
        embeddings.append(embedding)

    return images, np.array(embeddings)


def cosine_distance_matrix(embeddings1, embeddings2=None):
    """Compute cosine distance matrix between two sets of embeddings."""
    # Normalize embeddings
    norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    normalized1 = embeddings1 / norms1

    if embeddings2 is None:
        normalized2 = normalized1
    else:
        norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        normalized2 = embeddings2 / norms2

    # Similarity matrix (cosine similarity)
    similarity_matrix = normalized1 @ normalized2.T

    # Convert to distance
    distance_matrix = 1 - similarity_matrix

    return distance_matrix, similarity_matrix


def visualize_symmetric(images, png_files, distance_matrix, similarity_matrix,
                        model_display_name, output_dir, model_suffix):
    """Visualize symmetric similarity (all images compared to each other)."""
    n_images = len(png_files)
    fig, axes = plt.subplots(n_images, n_images, figsize=(3 * n_images, 3 * n_images))

    if n_images == 1:
        axes = np.array([[axes]])

    for i in range(n_images):
        distances = distance_matrix[i]
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

        axes[i, 0].set_ylabel(f"Query {i+1}", fontsize=10, rotation=0, ha="right", va="center")

    plt.suptitle(f"Image Similarity Ranking ({model_display_name})\nEach row: query image ranked by similarity to others",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"similarity_grid_{model_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved similarity grid to: {output_path}")
    plt.close()

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Cosine Similarity")

    labels = [f.stem for f in png_files]
    ax.set_xticks(range(n_images))
    ax.set_yticks(range(n_images))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n_images):
        for j in range(n_images):
            ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                   ha="center", va="center", color="white", fontsize=8)

    ax.set_title(f"{model_display_name} Cosine Similarity Matrix")
    plt.tight_layout()

    heatmap_path = output_dir / f"similarity_heatmap_{model_suffix}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"Saved similarity heatmap to: {heatmap_path}")
    plt.close()


def visualize_rows_vs_columns(row_images, row_paths, col_images, col_paths,
                               distance_matrix, similarity_matrix,
                               model_display_name, output_dir, model_suffix):
    """Visualize rows vs columns similarity (each row image compared to all column images)."""
    n_rows = len(row_paths)
    n_cols = len(col_paths)

    # Grid: first column is the row image, remaining columns are targets ranked by similarity
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(3 * (n_cols + 1), 3 * n_rows))

    if n_rows == 1:
        axes = np.array([axes])

    for i in range(n_rows):
        # First column: the query image
        ax = axes[i, 0]
        ax.imshow(row_images[i])
        ax.axis("off")
        ax.set_title(f"Query: {row_paths[i].stem}", fontsize=9)

        # Remaining columns: targets ranked by similarity
        distances = distance_matrix[i]
        ranked_indices = np.argsort(distances)

        for rank, j in enumerate(ranked_indices):
            ax = axes[i, rank + 1]
            ax.imshow(col_images[j])
            ax.axis("off")

            dist = distances[j]
            title = f"{col_paths[j].stem}\ndist={dist:.3f}"
            ax.set_title(title, fontsize=9)

    plt.suptitle(f"Image Similarity Ranking ({model_display_name})\nRows: query images, Columns: targets ranked by similarity",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"similarity_grid_{model_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved similarity grid to: {output_path}")
    plt.close()

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Cosine Similarity")

    row_labels = [f.stem for f in row_paths]
    col_labels = [f.stem for f in col_paths]
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                   ha="center", va="center", color="white", fontsize=8)

    ax.set_title(f"{model_display_name} Cosine Similarity Matrix")
    ax.set_xlabel("Targets")
    ax.set_ylabel("Query Images")
    plt.tight_layout()

    heatmap_path = output_dir / f"similarity_heatmap_{model_suffix}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"Saved similarity heatmap to: {heatmap_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute image similarities")
    parser.add_argument("--model", type=str, default="dinov2", choices=["dinov2", "clip"],
                        help="Model to use for embeddings (default: dinov2)")
    parser.add_argument("--dataset", type=int, default=0,
                        help="Dataset index (default: 0)")
    parser.add_argument("--layer", type=int, default=None,
                        help="DINOv2 layer to use (1-12). Early layers (1-4) are more color/texture sensitive. Default: final layer.")
    args = parser.parse_args()

    model_type = args.model
    dataset_idx = args.dataset
    layer = args.layer

    if model_type == "dinov2":
        layer_str = f" layer {layer}" if layer else " (final layer)"
        model_display_name = f"DINOv2{layer_str}"
        model_suffix = f"{model_type}_layer{layer}" if layer else model_type
    else:
        model_display_name = "CLIP"
        model_suffix = model_type
        if layer is not None:
            print("Warning: --layer is ignored for CLIP model")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, processor = load_model(model_type, device)

    # Get dataset config
    config = get_dataset_config(dataset_idx)
    print(f"\nDataset: {config['name']}")

    if config["mode"] == "symmetric":
        # Dataset 0: symmetric comparison
        png_files = [config["data_dir"] / name for name in config["image_order"]
                     if (config["data_dir"] / name).exists()]
        print(f"Found {len(png_files)} PNG files:")
        for f in png_files:
            print(f"  - {f.name}")

        if len(png_files) < 2:
            print("Need at least 2 images to compute similarities!")
            return

        print("\nComputing embeddings...")
        images, embeddings = compute_embeddings(png_files, model, processor, model_type, device, layer=layer)
        print(f"Embeddings shape: {embeddings.shape}")

        distance_matrix, similarity_matrix = cosine_distance_matrix(embeddings)
        print("\nSimilarity matrix:")
        print(similarity_matrix)

        visualize_symmetric(images, png_files, distance_matrix, similarity_matrix,
                           model_display_name, config["output_dir"], model_suffix)

    elif config["mode"] == "rows_vs_columns":
        # Dataset 1: rows compared against columns
        row_paths = [config["data_dir"] / name for name in config["row_images"]
                     if (config["data_dir"] / name).exists()]
        col_paths = [config["data_dir"] / name for name in config["column_images"]
                     if (config["data_dir"] / name).exists()]

        print(f"Row images (queries): {len(row_paths)}")
        for f in row_paths:
            print(f"  - {f.name}")
        print(f"Column images (targets): {len(col_paths)}")
        for f in col_paths:
            print(f"  - {f.name}")

        if len(row_paths) < 1 or len(col_paths) < 1:
            print("Need at least 1 row image and 1 column image!")
            return

        print("\nComputing embeddings for row images...")
        row_images, row_embeddings = compute_embeddings(row_paths, model, processor, model_type, device, layer=layer)
        print(f"Row embeddings shape: {row_embeddings.shape}")

        print("Computing embeddings for column images...")
        col_images, col_embeddings = compute_embeddings(col_paths, model, processor, model_type, device, layer=layer)
        print(f"Column embeddings shape: {col_embeddings.shape}")

        distance_matrix, similarity_matrix = cosine_distance_matrix(row_embeddings, col_embeddings)
        print("\nSimilarity matrix (rows vs columns):")
        print(similarity_matrix)

        visualize_rows_vs_columns(row_images, row_paths, col_images, col_paths,
                                   distance_matrix, similarity_matrix,
                                   model_display_name, config["output_dir"], model_suffix)


if __name__ == "__main__":
    main()
