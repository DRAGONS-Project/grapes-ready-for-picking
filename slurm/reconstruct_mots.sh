#!/bin/bash
#SBATCH --job-name=reconstruct_mots
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/reconstruct_mots/%j.out
#SBATCH --error=logs/reconstruct_mots/%j.err

set -euo pipefail

module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
WORK_DIR=/lustre/tmp/slurm/$SLURM_JOB_ID/work
BASE_DIR=$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1
OUTPUT_DIR=$BASE_DIR/runs/$SLURM_JOB_ID

# Frames and SfM are shared across runs (expensive to recompute).
FRAMES_DIR=$BASE_DIR/frames
COLMAP_DIR=$BASE_DIR/colmap

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

# MOTS-Annotated UAV Vineyard, PathPlanning_1 (Zenodo 10625595, ~67.7 MB),
# multi-angle harvest-time UAV video designed to minimize leaf occlusion.
VIDEO="$PROJECT_DIR/data/mots/PathPlanning_1.mp4"

# 1. Frame extraction — cached in OUTPUT_DIR so SfM can be skipped on reruns.
# Higher fps than the standardized protocol: this video is only ~27s, so 2.0
# fps yields just 54 frames; 7.5 fps gets close to the max-frames=200 cap.
if [ -d "$FRAMES_DIR" ] && [ "$(ls -A "$FRAMES_DIR")" ]; then
    echo "Reusing cached frames from $FRAMES_DIR"
else
    uv run --extra recon python src/reconstruction/extract_frames.py \
        "$VIDEO" "$FRAMES_DIR" --fps 7.5
fi

# 2. COLMAP SfM (pycolmap) — cached alongside frames.
if [ -f "$COLMAP_DIR/best_sparse_dir.txt" ]; then
    echo "Reusing cached SfM from $COLMAP_DIR"
else
    uv run --extra recon python src/reconstruction/sfm.py \
        "$FRAMES_DIR" "$COLMAP_DIR"
fi
SPARSE_DIR=$(cat "$COLMAP_DIR/best_sparse_dir.txt")

# 3. 3DGS training, with adaptive density control.
uv run --extra recon python src/reconstruction/train_gs.py \
    "$SPARSE_DIR" "$FRAMES_DIR" "$OUTPUT_DIR" \
    --iters 30000

echo "Done. Outputs in $OUTPUT_DIR"

# Keep a symlink pointing to the latest run for convenience.
ln -sfn "$OUTPUT_DIR" "$BASE_DIR/latest"
