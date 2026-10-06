#!/bin/bash
#SBATCH --job-name=compute_metrics
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/compute_metrics/%j.out
#SBATCH --error=logs/compute_metrics/%j.err

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
mkdir -p "$PROJECT_DIR/logs/compute_metrics"
cd "$PROJECT_DIR"

export UV_CACHE_DIR="$PROJECT_DIR/.uv_cache"
export UV_DATA_DIR="$PROJECT_DIR/.uv_data"
mkdir -p "$UV_CACHE_DIR" "$UV_DATA_DIR"

echo "=== BLT preprocessed (50k, fixed) ==="
uv run --extra recon python src/reconstruction/compute_metrics.py \
    outputs/reconstruction/blt_preprocessed_gs/5357100/eval

echo ""
echo "=== Terras Gauda row4_3_1 ==="
uv run --extra recon python src/reconstruction/compute_metrics.py \
    outputs/reconstruction/terras_gauda_row4_3_1/eval

echo ""
echo "=== BLT raw (50k, failure case) ==="
uv run --extra recon python src/reconstruction/compute_metrics.py \
    outputs/reconstruction/blt_raw_gs/5357237/eval

echo "Done."
