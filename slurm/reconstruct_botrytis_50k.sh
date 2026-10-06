#!/bin/bash
#SBATCH --job-name=botrytis_50k
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/botrytis_50k/%j.out
#SBATCH --error=logs/botrytis_50k/%j.err

set -euo pipefail
module load CUDA/13.0.0

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export UV_CACHE_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction/.uv_cache

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
WORK_DIR=/lustre/tmp/slurm/$SLURM_JOB_ID/work
COLMAP_DIR=$PROJECT_DIR/outputs/reconstruction/botrytis_50k/colmap
OUTPUT_DIR=$PROJECT_DIR/outputs/reconstruction/botrytis_50k/$SLURM_JOB_ID

mkdir -p "$WORK_DIR" "$COLMAP_DIR" "$OUTPUT_DIR" logs/botrytis_50k
cd "$PROJECT_DIR"

# Multispectral Botrytis vineyard (Zenodo 7383601), Flight 3 (45 degree tilt),
# Micasense RedEdge-MX 5-band captures.
ZIP="$WORK_DIR/45_V1.zip"
wget -c -O "$ZIP" "https://zenodo.org/api/records/7383601/files/45_V1.zip/content"

# 1. Compose RGB from visible Micasense bands; frames land in WORK_DIR
FRAMES_DIR="$WORK_DIR/frames"
uv run --extra recon python src/reconstruction/extract_botrytis_rgb.py \
    "$ZIP" "$FRAMES_DIR"

# 2. SfM — cached persistently so reruns skip this step
if [ -f "$COLMAP_DIR/best_sparse_dir.txt" ]; then
    echo "Reusing cached SfM from $COLMAP_DIR"
    # Frames must still exist for training; re-extract if work dir was cleaned
else
    uv run --extra recon python src/reconstruction/sfm.py \
        "$FRAMES_DIR" "$COLMAP_DIR" --camera-model OPENCV
fi
SPARSE_DIR=$(cat "$COLMAP_DIR/best_sparse_dir.txt")

# 3. GS training — 50k iters, matching BLT preprocessed standard
uv run --extra recon python src/reconstruction/train_gs.py \
    "$SPARSE_DIR" "$FRAMES_DIR" "$OUTPUT_DIR" \
    --iters 50000

echo "Done. Outputs in $OUTPUT_DIR"
ln -sfn "$OUTPUT_DIR" "$PROJECT_DIR/outputs/reconstruction/botrytis_50k/latest"
