#!/bin/bash
#SBATCH --job-name=reconstruct_terras_gauda
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/reconstruct_terras_gauda/%j.out
#SBATCH --error=logs/reconstruct_terras_gauda/%j.err

set -euo pipefail

module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
WORK_DIR=/lustre/tmp/slurm/$SLURM_JOB_ID/work
OUTPUT_DIR=$PROJECT_DIR/outputs/reconstruction/terras_gauda_row4_3_1

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

# Bodegas Terras Gauda UAV RGB (Zenodo 7330951, pre-harvest 28 Jun 2021),
# Row4.3_1.mp4 (~305 MB, 1467 frames @ 60fps, 24.5s) — full lateral pass along row 4.3.
# GPU compute nodes lack outbound internet; use pre-cached copy on shared storage.
VIDEO="$PROJECT_DIR/data/bodegas_terras_gauda/Row4.3_1.mp4"

# 1. Frame extraction. fps=10 with this 60fps source gives step=6 -> 244 frames,
#    hitting the max_frames=200 cap exactly (175 train / 25 test at holdout_every=8).
uv run --extra recon python src/reconstruction/extract_frames.py \
    "$VIDEO" "$WORK_DIR/frames" --fps 10

# 2. COLMAP SfM (pycolmap)
uv run --extra recon python src/reconstruction/sfm.py \
    "$WORK_DIR/frames" "$WORK_DIR/colmap"
SPARSE_DIR=$(cat "$WORK_DIR/colmap/best_sparse_dir.txt")

# 3. 3DGS training -- extended to 10k iters to check whether more views +
#    more training closes the gap toward the ~35 PSNR / 0.8 SSIM target.
uv run --extra recon python src/reconstruction/train_gs.py \
    "$SPARSE_DIR" "$WORK_DIR/frames" "$OUTPUT_DIR" --iters 30000

echo "Done. Outputs in $OUTPUT_DIR"
