#!/bin/bash
#SBATCH --job-name=blt_prep_dedup
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/blt_prep_dedup/%j.out
#SBATCH --error=logs/blt_prep_dedup/%j.err

set -euo pipefail
module load CUDA/13.0.0

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export UV_CACHE_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction/.uv_cache

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
BLT_DIR=/home/grid/users/plgmwlodarzc/synthetic-grapes/data
OUTPUT_DIR=$PROJECT_DIR/outputs/reconstruction/blt_preprocessed_gs_dedup/$SLURM_JOB_ID

mkdir -p "$OUTPUT_DIR" logs/blt_prep_dedup
cd "$PROJECT_DIR"

# Existing COLMAP from the original preprocessed run — no SfM needed.
SPARSE_DIR=$BLT_DIR/preprocessed/colmap/sparse/0
IMAGE_DIR=$BLT_DIR/preprocessed/images

# --min-camera-dist 0.1: drops frames within 10 cm of an already-kept frame.
# This removes the ~5 stationary frames at the start of the sequence (0.0001–0.0003 m apart)
# while keeping all frames once the robot starts moving (next gap ~0.2 m).
uv run --extra recon python src/reconstruction/train_gs.py \
    "$SPARSE_DIR" "$IMAGE_DIR" "$OUTPUT_DIR" \
    --iters 50000 \
    --min-camera-dist 0.1

echo "Done. Outputs in $OUTPUT_DIR"
ln -sfn "$OUTPUT_DIR" "$PROJECT_DIR/outputs/reconstruction/blt_preprocessed_gs_dedup/latest"
