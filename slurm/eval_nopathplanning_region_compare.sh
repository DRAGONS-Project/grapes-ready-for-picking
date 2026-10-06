#!/bin/bash
#SBATCH --job-name=eval_nopathplan_region
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_nopathplanning_region_compare/%j.out
#SBATCH --error=logs/eval_nopathplanning_region_compare/%j.err

# Fair (non-flare-region-only) comparison of baseline / bilateral / masked on
# MOTS NoPathPlanning_1. nerfstudio's own eval scores the full frame including
# the flare pixels the masked run was never trained on -- see
# ns_eval_masked_region.py docstring. This applies the SAME external flare
# mask to all three checkpoints so the comparison isolates reconstruction
# quality on the pixels every run was actually trying to get right.

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
NS_BASE=$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1
MASKS_DIR=$NS_BASE/work/ns_data_masked/masks
mkdir -p "$PROJECT_DIR/logs/eval_nopathplanning_region_compare"
cd "$PROJECT_DIR"

BASELINE_CONFIG="$NS_BASE/nopathplanning_baseline/splatfacto/2026-06-16_032024/config.yml"
BILATERAL_CONFIG="$NS_BASE/nopathplanning_bilateral/splatfacto/2026-06-16_032024/config.yml"
MASKED_CONFIG=$(ls -td "$NS_BASE"/nopathplanning_masked/splatfacto/*/ | head -1)config.yml

for NAME_CONFIG in "baseline:$BASELINE_CONFIG" "bilateral:$BILATERAL_CONFIG" "masked:$MASKED_CONFIG"; do
    NAME="${NAME_CONFIG%%:*}"
    CONFIG="${NAME_CONFIG#*:}"
    echo "=== Region-compare eval: $NAME ($CONFIG) ==="
    uv run --extra recon-ns python src/reconstruction/ns_eval_masked_region.py \
        --load-config "$CONFIG" \
        --masks-dir "$MASKS_DIR" \
        --output-path "$NS_BASE/nopathplanning_${NAME}/eval_metrics_region_compare.json"
done

echo "Done."
