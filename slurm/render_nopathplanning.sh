#!/bin/bash
#SBATCH --job-name=render_nopathplan
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/render_nopathplanning/%j.out
#SBATCH --error=logs/render_nopathplanning/%j.err

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
mkdir -p "$PROJECT_DIR/logs/render_nopathplanning"
cd "$PROJECT_DIR"

TRANSFORMS="$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1/work/ns_data/transforms.json"
NS_BASE="$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1"
OUT_BASE="$PROJECT_DIR/outputs/figures/mots_nopathplanning"

# ── baseline (step 29999) ─────────────────────────────────────────────────────
BASELINE_CKPT="$NS_BASE/nopathplanning_baseline/splatfacto/2026-06-16_032024/nerfstudio_models/step-000029999.ckpt"
BASELINE_OUT="$OUT_BASE/baseline"
mkdir -p "$BASELINE_OUT"

echo "=== Rendering baseline ==="
uv run --extra recon python src/visualization/render_gs_views.py \
    --model           "$BASELINE_CKPT" \
    --transforms-json "$TRANSFORMS" \
    --output          "$BASELINE_OUT/views_grid.png" \
    --width  960 --height 720 --fov 60

# Also save individual single-view angles useful for paper figures
for AZ in 0 45 90 180; do
    uv run --extra recon python src/visualization/render_gs_views.py \
        --model           "$BASELINE_CKPT" \
        --transforms-json "$TRANSFORMS" \
        --output          "$BASELINE_OUT/view_az${AZ}.png" \
        --width 960 --height 720 --fov 60 \
        --az "$AZ" --el 25
done

echo "Baseline renders done."

# ── bilateral (use whatever the latest saved checkpoint is; training was cut
#    off by the SLURM time limit at step 28000/30000, 93% done) ───────────────
BILATERAL_DIR="$NS_BASE/nopathplanning_bilateral/splatfacto/2026-06-16_032024/nerfstudio_models"
BILATERAL_CKPT=$(ls -1 "$BILATERAL_DIR"/step-*.ckpt | sort | tail -1)
BILATERAL_OUT="$OUT_BASE/bilateral"
mkdir -p "$BILATERAL_OUT"

echo "=== Rendering bilateral (ckpt: $(basename $BILATERAL_CKPT)) ==="
uv run --extra recon python src/visualization/render_gs_views.py \
    --model           "$BILATERAL_CKPT" \
    --transforms-json "$TRANSFORMS" \
    --output          "$BILATERAL_OUT/views_grid.png" \
    --width  960 --height 720 --fov 60

for AZ in 0 45 90 180; do
    uv run --extra recon python src/visualization/render_gs_views.py \
        --model           "$BILATERAL_CKPT" \
        --transforms-json "$TRANSFORMS" \
        --output          "$BILATERAL_OUT/view_az${AZ}.png" \
        --width 960 --height 720 --fov 60 \
        --az "$AZ" --el 25
done

echo "Bilateral renders done."

# ── masked (flare-region excluded from training loss; use whatever the latest
#    run directory is, in case of reruns) ─────────────────────────────────────
MASKED_DIR=$(ls -td "$NS_BASE"/nopathplanning_masked/splatfacto/*/ | head -1)
MASKED_CKPT="${MASKED_DIR}nerfstudio_models/step-000029999.ckpt"
MASKED_CONFIG="${MASKED_DIR}config.yml"
MASKED_OUT="$OUT_BASE/masked"
mkdir -p "$MASKED_OUT"

echo "=== Rendering masked (ckpt: $MASKED_CKPT) ==="
uv run --extra recon python src/visualization/render_gs_views.py \
    --model           "$MASKED_CKPT" \
    --transforms-json "$TRANSFORMS" \
    --output          "$MASKED_OUT/views_grid.png" \
    --width  960 --height 720 --fov 60

for AZ in 0 45 90 180; do
    uv run --extra recon python src/visualization/render_gs_views.py \
        --model           "$MASKED_CKPT" \
        --transforms-json "$TRANSFORMS" \
        --output          "$MASKED_OUT/view_az${AZ}.png" \
        --width 960 --height 720 --fov 60 \
        --az "$AZ" --el 25
done

echo "Masked renders done."

# ── flare-region before/after: baseline vs. masked on the actual held-out
#    frame with the most flare, cropped to the flare bounding box ────────────
BASELINE_CONFIG="$NS_BASE/nopathplanning_baseline/splatfacto/2026-06-16_032024/config.yml"
FLARE_MASKS_DIR="$NS_BASE/work/ns_data_masked/masks"

echo "=== Rendering flare-region before/after comparison ==="
uv run --extra recon-ns python src/reconstruction/render_flare_comparison.py \
    --baseline-config "$BASELINE_CONFIG" \
    --masked-config   "$MASKED_CONFIG" \
    --masks-dir       "$FLARE_MASKS_DIR" \
    --output-dir      "$OUT_BASE"

echo "Flare-region comparison done."
echo "All outputs under $OUT_BASE"
