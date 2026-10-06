#!/bin/bash
#SBATCH --job-name=blt_greece_gs
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:30:00
#SBATCH --output=logs/blt_greece/%j.out
#SBATCH --error=logs/blt_greece/%j.err

set -euo pipefail

module load CUDA/13.0.0

# Work around an OpenBLAS thread-pool bug ("Bad memory unallocation") that can
# segfault pycolmap during incremental mapping.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
export UV_CACHE_DIR="$PROJECT_DIR/.uv_cache"
export UV_DATA_DIR="$PROJECT_DIR/.uv_data"

# Camera to reconstruct: front (default, like-for-like with the Riseholme run)
# or side (cluster-height canopy view). Override with:
#   sbatch --export=ALL,CAM=side slurm/reconstruct_blt_greece.sh
CAM=${CAM:-front}

# BLT, Ktima Gerovassiliou (Greece), 15 Sep 2022 (harvest) session: one corridor
# pass, 118 frames per camera at 2 fps, 1600x900 (standardized protocol). Frames
# were pulled from the remote bag with HTTP Range requests; see
# data/blt_greece_20220915/segment_info.json.
IMAGE_DIR=$PROJECT_DIR/data/blt_greece_20220915/$CAM
BASE_DIR=$PROJECT_DIR/outputs/reconstruction/blt_greece_20220915_$CAM
COLMAP_DIR=$BASE_DIR/colmap
OUTPUT_DIR=$BASE_DIR/$SLURM_JOB_ID

mkdir -p "$COLMAP_DIR" "$OUTPUT_DIR" logs/blt_greece
cd "$PROJECT_DIR"

# SfM: cached across runs in a persistent dir alongside outputs.
if [ -f "$COLMAP_DIR/best_sparse_dir.txt" ]; then
    echo "Reusing cached SfM from $COLMAP_DIR"
else
    uv run --extra recon python src/reconstruction/sfm.py \
        "$IMAGE_DIR" "$COLMAP_DIR" --camera-model OPENCV
fi
SPARSE_DIR=$(cat "$COLMAP_DIR/best_sparse_dir.txt")

# 3DGS training — 50k iters, matching the other BLT runs.
uv run --extra recon python src/reconstruction/train_gs.py \
    "$SPARSE_DIR" "$IMAGE_DIR" "$OUTPUT_DIR" \
    --iters 50000

# PSNR / SSIM / LPIPS on the held-out views.
uv run --extra recon python src/reconstruction/compute_metrics.py "$OUTPUT_DIR/eval"

echo "Done. Outputs in $OUTPUT_DIR"
ln -sfn "$OUTPUT_DIR" "$BASE_DIR/latest"
