#!/bin/bash
#SBATCH --job-name=eval_nopathplan_deflare_ghostmask
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_nopathplanning_deflare_ghostmasked/%j.out
#SBATCH --error=logs/eval_nopathplanning_deflare_ghostmasked/%j.err

# Quantitative eval for the deflare+ghostmasked splatfacto run on MOTS
# NoPathPlanning_1: standard full-frame ns-eval (for comparison against
# baseline 20.94/0.674, bilateral 19.89/0.676, masked 20.89/0.674, deflare --
# see eval_nopathplanning_deflare.sh) plus the non-flare-region-only metric
# using the same external veiling-glare HSV mask as the other arms.
#
# Same PSNR/SSIM-vs-hazy-ground-truth caveat as the deflare arm applies here
# too (docs/flare_fix_research.md) -- treat the qualitative render comparison
# (render_nopathplanning_deflare_ghostmasked_comparison.sh) as primary signal.

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
NS_BASE=$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1
MASKS_DIR=$NS_BASE/work/ns_data_masked/masks
mkdir -p "$PROJECT_DIR/logs/eval_nopathplanning_deflare_ghostmasked"
cd "$PROJECT_DIR"

CONFIG=$(ls -td "$NS_BASE"/nopathplanning_deflare_ghostmasked/splatfacto/*/ | head -1)config.yml

echo "=== Evaluating deflare_ghostmasked ($CONFIG) ==="
uv run --extra recon-ns python src/reconstruction/ns_eval_patched.py \
    --load-config "$CONFIG" \
    --output-path "$NS_BASE/nopathplanning_deflare_ghostmasked/eval_metrics.json"

echo "=== Region-compare eval: deflare_ghostmasked ==="
uv run --extra recon-ns python src/reconstruction/ns_eval_masked_region.py \
    --load-config "$CONFIG" \
    --masks-dir "$MASKS_DIR" \
    --output-path "$NS_BASE/nopathplanning_deflare_ghostmasked/eval_metrics_region_compare.json"

echo "Done."
echo "Deflare_ghostmasked: $NS_BASE/nopathplanning_deflare_ghostmasked/eval_metrics.json"
echo "Deflare_ghostmasked region-compare: $NS_BASE/nopathplanning_deflare_ghostmasked/eval_metrics_region_compare.json"
