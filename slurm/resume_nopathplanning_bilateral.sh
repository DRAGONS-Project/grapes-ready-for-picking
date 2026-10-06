#!/bin/bash
#SBATCH --job-name=nopathplan_bil
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/nopathplanning_bilateral/%j.out
#SBATCH --error=logs/nopathplanning_bilateral/%j.err

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
mkdir -p "$PROJECT_DIR/logs/nopathplanning_bilateral"
cd "$PROJECT_DIR"

CONFIG="$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1/nopathplanning_bilateral/splatfacto/2026-06-16_032024/config.yml"

# Resume from step 24000 → 30000.
# --load-config re-uses the existing experiment dir + picks the latest checkpoint.
# ns_train_resume.py patches torch.load for PyTorch 2.6 weights_only compat.
uv run --extra recon-ns python src/reconstruction/ns_train_resume.py \
    splatfacto \
    --load-config "$CONFIG" \
    --max-num-iterations 30000 \
    --vis wandb

echo "Done. Bilateral training complete."
