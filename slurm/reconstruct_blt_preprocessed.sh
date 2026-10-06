#!/bin/bash
#SBATCH --job-name=blt_prep_gs
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/blt_preprocessed_gs/%j.out
#SBATCH --error=logs/blt_preprocessed_gs/%j.err

set -euo pipefail

module load CUDA/13.0.0

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
BLT_DIR=/home/grid/users/plgmwlodarzc/synthetic-grapes/data
OUTPUT_DIR=$PROJECT_DIR/outputs/reconstruction/blt_preprocessed_gs/$SLURM_JOB_ID

mkdir -p "$OUTPUT_DIR" logs/blt_preprocessed_gs
cd "$PROJECT_DIR"

# COLMAP sparse/0 has 87 registered images (sparse/1 has only 25 — skip it).
SPARSE_DIR=$BLT_DIR/preprocessed/colmap/sparse/0
IMAGE_DIR=$BLT_DIR/preprocessed/images

uv run --extra recon python src/reconstruction/train_gs.py \
    "$SPARSE_DIR" "$IMAGE_DIR" "$OUTPUT_DIR" \
    --iters 50000

echo "Done. Outputs in $OUTPUT_DIR"
ln -sfn "$OUTPUT_DIR" "$PROJECT_DIR/outputs/reconstruction/blt_preprocessed_gs/latest"
