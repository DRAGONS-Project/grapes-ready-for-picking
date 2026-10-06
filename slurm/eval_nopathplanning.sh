#!/bin/bash
#SBATCH --job-name=eval_nopathplan
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_nopathplanning/%j.out
#SBATCH --error=logs/eval_nopathplanning/%j.err

# Quantitative eval (PSNR/SSIM/LPIPS via ns-eval) for the baseline vs.
# bilateral-grid (sunbeam fix) splatfacto runs on MOTS NoPathPlanning_1.
# Bilateral was cut off by the SLURM time limit at step 28000/30000 (93%
# done, see resume_nopathplanning_bilateral.sh) -- evaluated as-is rather
# than waiting for a further resume.

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
NS_BASE=$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1
mkdir -p "$PROJECT_DIR/logs/eval_nopathplanning"
cd "$PROJECT_DIR"

echo "=== Evaluating baseline ==="
uv run --extra recon-ns python src/reconstruction/ns_eval_patched.py \
    --load-config "$NS_BASE/nopathplanning_baseline/splatfacto/2026-06-16_032024/config.yml" \
    --output-path "$NS_BASE/nopathplanning_baseline/eval_metrics.json"

echo "=== Evaluating bilateral (step 28000/30000) ==="
uv run --extra recon-ns python src/reconstruction/ns_eval_patched.py \
    --load-config "$NS_BASE/nopathplanning_bilateral/splatfacto/2026-06-16_032024/config.yml" \
    --output-path "$NS_BASE/nopathplanning_bilateral/eval_metrics.json"

echo "Done."
echo "Baseline:  $NS_BASE/nopathplanning_baseline/eval_metrics.json"
echo "Bilateral: $NS_BASE/nopathplanning_bilateral/eval_metrics.json"
