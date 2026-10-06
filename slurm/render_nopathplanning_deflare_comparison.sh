#!/bin/bash
#SBATCH --job-name=render_nopathplan_deflare_cmp
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/render_nopathplanning_deflare_comparison/%j.out
#SBATCH --error=logs/render_nopathplanning_deflare_comparison/%j.err

# 4-way qualitative comparison (ground truth / baseline / masked / deflare) on
# the most flare-affected held-out frame of MOTS NoPathPlanning_1. This is the
# primary signal for judging the deflare fix per docs/flare_fix_research.md
# (full-frame PSNR against a possibly-hazy ground truth may not move even if
# the fix is visually correct).

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
NS_BASE=$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1
MASKS_DIR=$NS_BASE/work/ns_data_masked/masks
mkdir -p "$PROJECT_DIR/logs/render_nopathplanning_deflare_comparison"
cd "$PROJECT_DIR"

BASELINE_CONFIG="$NS_BASE/nopathplanning_baseline/splatfacto/2026-06-16_032024/config.yml"
MASKED_CONFIG=$(ls -td "$NS_BASE"/nopathplanning_masked/splatfacto/*/ | head -1)config.yml
DEFLARE_CONFIG=$(ls -td "$NS_BASE"/nopathplanning_deflare/splatfacto/*/ | head -1)config.yml

uv run --extra recon-ns python src/reconstruction/render_flare_comparison_multi.py \
    --config "baseline=$BASELINE_CONFIG" \
    --config "masked=$MASKED_CONFIG" \
    --config "deflare=$DEFLARE_CONFIG" \
    --masks-dir "$MASKS_DIR" \
    --output-dir "$PROJECT_DIR/outputs/figures/mots_nopathplanning"

echo "Done."
