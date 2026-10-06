#!/bin/bash
#SBATCH --job-name=render_gs_views
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/render_gs_views/%j.out
#SBATCH --error=logs/render_gs_views/%j.err

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
mkdir -p "$PROJECT_DIR/logs/render_gs_views"
cd "$PROJECT_DIR"

# Use the best splatfacto model (PSNR 23.34, SSIM 0.697 — 30k iterations)
MODEL="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/mots_pathplanning_1/splatfacto/2026-06-16_001706/nerfstudio_models/step-000029999.ckpt"
TRANSFORMS="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/work/ns_data/transforms.json"
OUTPUT="$PROJECT_DIR/outputs/teaser/views_grid_4x3.png"

uv run --extra recon python src/visualization/render_gs_views.py \
    --model           "$MODEL" \
    --transforms-json "$TRANSFORMS" \
    --output          "$OUTPUT" \
    --width           640 \
    --height          480 \
    --fov             60

echo "Done. Grid: $OUTPUT"
echo "Individual views: outputs/teaser/view_*.png"
