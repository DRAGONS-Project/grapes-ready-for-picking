#!/bin/bash
#SBATCH --job-name=smoke_test_reconstruction
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_test_reconstruction/%j.out
#SBATCH --error=logs/smoke_test_reconstruction/%j.err

set -euo pipefail

module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
WORK_DIR=/lustre/tmp/slurm/$SLURM_JOB_ID/work
OUTPUT_DIR=$PROJECT_DIR/outputs/reconstruction/smoke_test

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

# Smallest clip from Bodegas Terras Gauda UAV RGB (Zenodo 7330951, ~40.9 MB)
VIDEO="$WORK_DIR/Row4.3_2.mp4"
wget -c -O "$VIDEO" "https://zenodo.org/api/records/7330951/files/Row4.3_2.mp4/content"

# 1. Frame extraction (uses standardized fps/max-dim/max-frames from configs/reconstruction.yaml)
uv run --extra recon python src/reconstruction/extract_frames.py \
    "$VIDEO" "$WORK_DIR/frames"

# 2. COLMAP SfM (pycolmap)
uv run --extra recon python src/reconstruction/sfm.py \
    "$WORK_DIR/frames" "$WORK_DIR/colmap"
SPARSE_DIR=$(cat "$WORK_DIR/colmap/best_sparse_dir.txt")

# 3. 3DGS training — short run for smoke test (overrides the standardized iters/save-every,
# which are tuned for quality runs, not pipeline validation). Holdout/eval still exercised.
uv run --extra recon python src/reconstruction/train_gs.py \
    "$SPARSE_DIR" "$WORK_DIR/frames" "$OUTPUT_DIR" \
    --iters 500 --save-every 100

echo "Done. Outputs in $OUTPUT_DIR"
