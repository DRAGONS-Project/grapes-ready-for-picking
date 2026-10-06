#!/bin/bash
#SBATCH --job-name=eval_nopathplan_mask
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_nopathplanning_masked/%j.out
#SBATCH --error=logs/eval_nopathplanning_masked/%j.err

# Quantitative eval (PSNR/SSIM/LPIPS via ns-eval) for the flare-masked
# splatfacto run on MOTS NoPathPlanning_1, for comparison against baseline
# (PSNR 20.94 / SSIM 0.674) and bilateral (PSNR 19.89 / SSIM 0.676) -- see
# eval_nopathplanning.sh.

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
NS_BASE=$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1
mkdir -p "$PROJECT_DIR/logs/eval_nopathplanning_masked"
cd "$PROJECT_DIR"

MASKED_CONFIG=$(ls -td "$NS_BASE"/nopathplanning_masked/splatfacto/*/ | head -1)config.yml

echo "=== Evaluating masked ($MASKED_CONFIG) ==="
uv run --extra recon-ns python src/reconstruction/ns_eval_patched.py \
    --load-config "$MASKED_CONFIG" \
    --output-path "$NS_BASE/nopathplanning_masked/eval_metrics.json"

echo "Done."
echo "Masked:    $NS_BASE/nopathplanning_masked/eval_metrics.json"
echo "Baseline:  $NS_BASE/nopathplanning_baseline/eval_metrics.json"
echo "Bilateral: $NS_BASE/nopathplanning_bilateral/eval_metrics.json"
