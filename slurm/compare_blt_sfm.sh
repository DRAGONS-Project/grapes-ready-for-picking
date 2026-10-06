#!/bin/bash
#SBATCH --job-name=compare_blt_sfm
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/compare_blt_sfm/%j.out
#SBATCH --error=logs/compare_blt_sfm/%j.err

set -euo pipefail

module load CUDA/13.0.0

# Work around an OpenBLAS thread-pool bug ("Bad memory unallocation") that can
# segfault pycolmap during incremental mapping.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
BLT_DIR=/home/grid/users/plgmwlodarzc/synthetic-grapes/data
WORK_DIR=/lustre/tmp/slurm/$SLURM_JOB_ID/work
OUTPUT_DIR=$PROJECT_DIR/outputs/reconstruction/blt_sfm_comparison

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

# SfM on raw (unmasked, robot body visible) BLT frames vs. the existing
# preprocessed (masked) SfM reconstruction.
uv run --extra recon python src/reconstruction/compare_blt_sfm.py \
    "$BLT_DIR/extracted/images" \
    "$BLT_DIR/preprocessed/colmap/sparse/0" \
    "$WORK_DIR/raw_colmap" \
    "$OUTPUT_DIR"

echo "Done. Outputs in $OUTPUT_DIR"
