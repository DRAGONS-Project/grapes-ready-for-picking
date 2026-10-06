#!/bin/bash
#SBATCH --job-name=blt_raw_gs
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:30:00
#SBATCH --output=logs/blt_raw_gs/%j.out
#SBATCH --error=logs/blt_raw_gs/%j.err

set -euo pipefail

module load CUDA/13.0.0

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
BLT_DIR=/home/grid/users/plgmwlodarzc/synthetic-grapes/data
COLMAP_DIR=$PROJECT_DIR/outputs/reconstruction/blt_raw_gs/colmap
OUTPUT_DIR=$PROJECT_DIR/outputs/reconstruction/blt_raw_gs/$SLURM_JOB_ID

mkdir -p "$COLMAP_DIR" "$OUTPUT_DIR" logs/blt_raw_gs
cd "$PROJECT_DIR"

# Raw (unmasked) images — robot body visible, used to demonstrate SfM/GS failure case.
IMAGE_DIR=$BLT_DIR/extracted/images

# SfM: cached across runs in a persistent dir alongside outputs.
if [ -f "$COLMAP_DIR/best_sparse_dir.txt" ]; then
    echo "Reusing cached SfM from $COLMAP_DIR"
else
    uv run --extra recon python src/reconstruction/sfm.py \
        "$IMAGE_DIR" "$COLMAP_DIR" --camera-model OPENCV
fi
SPARSE_DIR=$(cat "$COLMAP_DIR/best_sparse_dir.txt")

uv run --extra recon python src/reconstruction/train_gs.py \
    "$SPARSE_DIR" "$IMAGE_DIR" "$OUTPUT_DIR" \
    --iters 50000

echo "Done. Outputs in $OUTPUT_DIR"
ln -sfn "$OUTPUT_DIR" "$PROJECT_DIR/outputs/reconstruction/blt_raw_gs/latest"
