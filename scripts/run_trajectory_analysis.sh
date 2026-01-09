#!/bin/bash
# Run trajectory analysis for all paper videos with various configurations

set -e

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/paper_vids"

# Video files
VID1="$DATA_DIR/vid1_41586_2025_9948_MOESM6_ESM.mp4"
VID2="$DATA_DIR/vid2_41586_2025_9948_MOESM7_ESM.mp4"
VID3="$DATA_DIR/vid3_41586_2025_9948_MOESM8_ESM.mp4"

echo "========================================"
echo "Running trajectory analysis for all videos"
echo "========================================"

# Function to run analysis for a single video without references
run_single_video() {
    local video=$1
    local video_name=$(basename "$video" .mp4)

    echo ""
    echo "Processing: $video_name"
    echo "----------------------------------------"

    # Without references
    echo "  - Final layer, no refs..."
    python "$PROJECT_DIR/trajectory_frame_s.py" "$video"

    echo "  - Layer 2, no refs..."
    python "$PROJECT_DIR/trajectory_frame_s.py" "$video" --layer 2

    echo "  - Layer 12, no refs..."
    python "$PROJECT_DIR/trajectory_frame_s.py" "$video" --layer 12
}

# Function to run analysis with all other videos as references
run_with_references() {
    local video=$1
    local ref1=$2
    local ref2=$3
    local video_name=$(basename "$video" .mp4)

    echo ""
    echo "Processing with references: $video_name"
    echo "----------------------------------------"

    # With references
    echo "  - Final layer, with refs..."
    python "$PROJECT_DIR/trajectory_frame_s.py" "$video" \
        --reference-videos "$ref1" "$ref2"

    echo "  - Layer 2, with refs..."
    python "$PROJECT_DIR/trajectory_frame_s.py" "$video" --layer 2 \
        --reference-videos "$ref1" "$ref2"

    echo "  - Layer 12, with refs..."
    python "$PROJECT_DIR/trajectory_frame_s.py" "$video" --layer 12 \
        --reference-videos "$ref1" "$ref2"
}

# Activate environment
source "$PROJECT_DIR/.venv/bin/activate"

# Run for each video without references
run_single_video "$VID1"
run_single_video "$VID2"
run_single_video "$VID3"

# Run for each video with the other two as references
run_with_references "$VID1" "$VID2" "$VID3"
run_with_references "$VID2" "$VID1" "$VID3"
run_with_references "$VID3" "$VID1" "$VID2"

echo ""
echo "========================================"
echo "All analyses complete!"
echo "Results saved to: $PROJECT_DIR/results/trajectory_frame_s/"
echo "========================================"
