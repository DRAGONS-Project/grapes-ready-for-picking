#!/bin/bash
#SBATCH --job-name=eval_nopathplan_deflare
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_nopathplanning_deflare/%j.out
#SBATCH --error=logs/eval_nopathplanning_deflare/%j.err

# Quantitative eval for the dark-channel-deflared splatfacto run on MOTS
# NoPathPlanning_1: standard full-frame ns-eval (for direct comparison against
# baseline 20.94/0.674, bilateral 19.89/0.676, masked 20.89/0.674 -- see
# eval_nopathplanning.sh, eval_nopathplanning_masked.sh) plus the
# non-flare-region-only metric (ns_eval_masked_region.py) using the SAME
# external HSV mask as the other arms, for apples-to-apples comparison.
#
# Caveat (see docs/flare_fix_research.md): if the deflare genuinely removes a
# haze that's present in the (unmodified) held-out ground truth too, PSNR/SSIM
# against that ground truth may not improve even if the fix is visually
# correct -- treat the render_flare_comparison_multi.py qualitative panel as
# the primary signal, these numbers as secondary.

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
NS_BASE=$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1
MASKS_DIR=$NS_BASE/work/ns_data_masked/masks
mkdir -p "$PROJECT_DIR/logs/eval_nopathplanning_deflare"
cd "$PROJECT_DIR"

DEFLARE_CONFIG=$(ls -td "$NS_BASE"/nopathplanning_deflare/splatfacto/*/ | head -1)config.yml

echo "=== Evaluating deflare ($DEFLARE_CONFIG) ==="
uv run --extra recon-ns python src/reconstruction/ns_eval_patched.py \
    --load-config "$DEFLARE_CONFIG" \
    --output-path "$NS_BASE/nopathplanning_deflare/eval_metrics.json"

echo "=== Region-compare eval: deflare ==="
uv run --extra recon-ns python src/reconstruction/ns_eval_masked_region.py \
    --load-config "$DEFLARE_CONFIG" \
    --masks-dir "$MASKS_DIR" \
    --output-path "$NS_BASE/nopathplanning_deflare/eval_metrics_region_compare.json"

echo "Done."
echo "Deflare: $NS_BASE/nopathplanning_deflare/eval_metrics.json"
echo "Deflare region-compare: $NS_BASE/nopathplanning_deflare/eval_metrics_region_compare.json"
