#!/bin/bash
#SBATCH --job-name=reconstruct_botrytis
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/reconstruct_botrytis/%j.out
#SBATCH --error=logs/reconstruct_botrytis/%j.err

set -euo pipefail

module load CUDA/13.0.0

# Work around an OpenBLAS thread-pool bug ("Bad memory unallocation") that can
# segfault pycolmap during incremental mapping.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
WORK_DIR=/lustre/tmp/slurm/$SLURM_JOB_ID/work
OUTPUT_DIR=$PROJECT_DIR/outputs/reconstruction/botrytis_45_v1

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
cd "$PROJECT_DIR"

# Multispectral Botrytis vineyard (Zenodo 7383601), Flight 3 (45 degree tilt),
# Micasense RedEdge-MX 5-band captures. 45_V1.zip (~5.7 GB), 599 capture sets.
ZIP="$WORK_DIR/45_V1.zip"
wget -c -O "$ZIP" "https://zenodo.org/api/records/7383601/files/45_V1.zip/content"

# 1. Compose RGB frames from the 3 visible Micasense bands (standardized
#    max-dim/max-frames from configs/reconstruction.yaml)
uv run --extra recon python src/reconstruction/extract_botrytis_rgb.py \
    "$ZIP" "$WORK_DIR/frames"

# 2. COLMAP SfM (pycolmap)
uv run --extra recon python src/reconstruction/sfm.py \
    "$WORK_DIR/frames" "$WORK_DIR/colmap"
SPARSE_DIR=$(cat "$WORK_DIR/colmap/best_sparse_dir.txt")

# 3. 3DGS training — standardized iters/save-every/holdout from configs/reconstruction.yaml
uv run --extra recon python src/reconstruction/train_gs.py \
    "$SPARSE_DIR" "$WORK_DIR/frames" "$OUTPUT_DIR"

echo "Done. Outputs in $OUTPUT_DIR"
