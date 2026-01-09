#!/usr/bin/env python3
"""
Compute and visualize frame embedding trajectories for video files.

Uses DINOv2 to embed video frames and projects them to 2D via PCA,
showing the temporal trajectory through embedding space.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import torch
from tqdm import tqdm

# Import from existing module
from compute_similarity import load_model, get_embedding


# Visualization constants
MAIN_COLOR = "#2563eb"  # Blue
REFERENCE_COLORS = [
    "#dc2626",  # Red
    "#16a34a",  # Green
    "#9333ea",  # Purple
    "#ea580c",  # Orange
    "#0891b2",  # Cyan
]


def extract_frames(video_path: str, fps_target: float) -> tuple[list, float]:
    """
    Extract frames from video at specified FPS rate.

    Args:
        video_path: Path to video file
        fps_target: Desired frames per second to extract

    Returns:
        frames: List of frames as PIL Images
        video_fps: The video's actual FPS
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print(f"Warning: Could not determine video FPS, using default 30")
        video_fps = 30.0

    frame_interval = video_fps / fps_target
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_idx = 0
    next_extract_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= next_extract_idx:
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            next_extract_idx += frame_interval

        frame_idx += 1

    cap.release()
    return frames, video_fps


def compute_frame_embeddings(
    frames: list,
    model,
    processor,
    model_type: str,
    device,
    layer: int = None,
) -> np.ndarray:
    """
    Compute embeddings for a list of PIL Image frames.

    Returns:
        embeddings: numpy array of shape (n_frames, embedding_dim)
    """
    embeddings = []
    for frame in tqdm(frames, desc="Computing embeddings"):
        emb = get_embedding(model, processor, frame, model_type, device, layer)
        embeddings.append(emb)
    return np.array(embeddings)


def compute_pca_2d(embeddings_dict: dict) -> tuple[dict, PCA]:
    """
    Compute 2D PCA projection of all embeddings together.

    Args:
        embeddings_dict: {"main": main_embeddings, "ref_0": ref0_embeddings, ...}

    Returns:
        pca_dict: Same structure with 2D coordinates
        pca: Fitted PCA object
    """
    # Concatenate all embeddings
    all_embeddings = []
    keys = []
    counts = []

    for key, emb in embeddings_dict.items():
        all_embeddings.append(emb)
        keys.append(key)
        counts.append(len(emb))

    combined = np.vstack(all_embeddings)

    # Fit PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    # Split back by video
    pca_dict = {}
    start = 0
    for key, count in zip(keys, counts):
        pca_dict[key] = projected[start : start + count]
        start += count

    return pca_dict, pca


def plot_trajectory(
    pca_dict: dict,
    output_path: str,
    main_video_name: str,
    reference_names: list = None,
    layer: int = None,
    fps: float = None,
):
    """
    Plot 2D PCA trajectory visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    main_pca = pca_dict["main"]

    # Plot main trajectory line
    ax.plot(
        main_pca[:, 0],
        main_pca[:, 1],
        color=MAIN_COLOR,
        linewidth=1.5,
        alpha=0.7,
        zorder=1,
    )

    # Plot intermediate points (excluding first and last)
    if len(main_pca) > 2:
        ax.scatter(
            main_pca[1:-1, 0],
            main_pca[1:-1, 1],
            c=MAIN_COLOR,
            s=60,
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
            label=f"{main_video_name}",
        )

    # Plot first frame (square marker)
    ax.scatter(
        main_pca[0, 0],
        main_pca[0, 1],
        c=MAIN_COLOR,
        s=120,
        marker="s",
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
        label="Start frame",
    )

    # Plot last frame (triangle marker)
    ax.scatter(
        main_pca[-1, 0],
        main_pca[-1, 1],
        c=MAIN_COLOR,
        s=120,
        marker="^",
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
        label="End frame",
    )

    # Plot reference videos
    if reference_names:
        for i, ref_name in enumerate(reference_names):
            ref_key = f"ref_{i}"
            ref_pca = pca_dict[ref_key]
            color = REFERENCE_COLORS[i % len(REFERENCE_COLORS)]

            ax.scatter(
                ref_pca[:, 0],
                ref_pca[:, 1],
                c=color,
                s=40,
                marker="x",
                alpha=0.6,
                zorder=1,
                label=f"{ref_name}",
            )

    # Labels and title
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)

    layer_str = f"Layer {layer}" if layer else "Final Layer"
    title = f"Frame Embedding Trajectory - {main_video_name}\n(DINOv2 {layer_str}, {fps} FPS)"
    ax.set_title(title, fontsize=14)

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_filename(fps: float, layer: int = None, has_refs: bool = False) -> str:
    """Generate filename based on parameters."""
    parts = ["trajectory"]
    parts.append(f"fps{fps}")

    if layer is not None:
        parts.append(f"layer{layer}")

    if has_refs:
        parts.append("refs")

    return "_".join(parts)


def get_output_paths(
    video_path: str, layer: int, fps: float, has_references: bool
) -> tuple[Path, str]:
    """
    Generate output directory and filename based on parameters.

    Returns:
        output_dir: Path to results/trajectory_frame_s/<vid_name>/
        filename_base: Base filename reflecting options
    """
    video_name = Path(video_path).stem
    output_dir = Path("results") / "trajectory_frame_s" / video_name
    filename_base = generate_filename(fps, layer, has_references)
    return output_dir, filename_base


def save_command_log(output_dir: Path, filename_base: str, sys_argv: list):
    """Save command that generated this output."""
    log_path = output_dir / f"{filename_base}.txt"
    with open(log_path, "w") as f:
        f.write("# Command used to generate this visualization\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        f.write("python " + " ".join(sys_argv) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and visualize frame embedding trajectories for video files."
    )
    parser.add_argument("video_path", type=str, help="Path to the main video file")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        choices=range(1, 13),
        metavar="1-12",
        help="DINOv2 layer to use (1-12). Default: final layer",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract. Default: 1.0",
    )
    parser.add_argument(
        "--reference-videos",
        type=str,
        nargs="+",
        default=None,
        help="Optional paths to reference video files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model (always DINOv2 for this script)
    model, processor = load_model("dinov2", device)

    # Extract frames from main video
    print(f"\nExtracting frames from: {args.video_path}")
    main_frames, video_fps = extract_frames(args.video_path, args.fps)
    print(
        f"  Video FPS: {video_fps:.1f}, Extracted {len(main_frames)} frames at {args.fps} FPS"
    )

    # Compute embeddings for main video
    print("Computing embeddings for main video...")
    main_embeddings = compute_frame_embeddings(
        main_frames, model, processor, "dinov2", device, args.layer
    )
    print(f"  Embeddings shape: {main_embeddings.shape}")

    # Process reference videos if provided
    embeddings_dict = {"main": main_embeddings}
    reference_names = []

    if args.reference_videos:
        for i, ref_path in enumerate(args.reference_videos):
            print(f"\nProcessing reference video: {ref_path}")
            ref_frames, _ = extract_frames(ref_path, args.fps)
            print(f"  Extracted {len(ref_frames)} frames")

            ref_embeddings = compute_frame_embeddings(
                ref_frames, model, processor, "dinov2", device, args.layer
            )

            ref_name = Path(ref_path).stem
            embeddings_dict[f"ref_{i}"] = ref_embeddings
            reference_names.append(ref_name)

    # Compute PCA
    print("\nComputing PCA projection...")
    pca_dict, pca = compute_pca_2d(embeddings_dict)
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")

    # Setup output paths
    main_video_name = Path(args.video_path).stem
    output_dir, filename_base = get_output_paths(
        args.video_path,
        args.layer,
        args.fps,
        has_references=(args.reference_videos is not None),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save
    output_path = output_dir / f"{filename_base}.png"
    print(f"\nSaving visualization to: {output_path}")
    plot_trajectory(
        pca_dict,
        output_path,
        main_video_name,
        reference_names=reference_names if reference_names else None,
        layer=args.layer,
        fps=args.fps,
    )

    # Save command log
    save_command_log(output_dir, filename_base, sys.argv)
    print(f"Saved command log to: {output_dir / f'{filename_base}.txt'}")


if __name__ == "__main__":
    main()
