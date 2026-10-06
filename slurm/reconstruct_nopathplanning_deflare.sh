#!/bin/bash
#SBATCH --job-name=nopathplan_deflare
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/reconstruct_nopathplanning_deflare/%j.out
#SBATCH --error=logs/reconstruct_nopathplanning_deflare/%j.err

# Lens-flare fix, attempt 3: pixel-level dark-channel-prior veiling-glare
# correction (deflare_dark_channel.py) applied to every frame before
# training, instead of excluding a hard-thresholded hotspot from the loss
# (masked: PSNR 20.89 vs baseline 20.94, statistically flat -- see
# eval_nopathplanning_masked.sh). Multi-view outlier analysis
# (flare_multiview_outliers.py) showed the damage is a low-level haze spread
# through the whole frame, not confined to the hotspot the mask excludes --
# see docs/flare_fix_research.md. Same camera poses/eval split as
# baseline/masked/bilateral (transforms.json reused verbatim, only the
# images/ directory changes) so this isolates the pixel-correction effect.

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
NS_BASE=$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1
SRC_DATA=$NS_BASE/work/ns_data
FRAMES_DEFLARED=$NS_BASE/work/frames_deflared
DEFLARE_DATA=$NS_BASE/work/ns_data_deflare

mkdir -p "$PROJECT_DIR/logs/reconstruct_nopathplanning_deflare"
cd "$PROJECT_DIR"

# 1. Deflare all frames (cheap, CPU-only, ~16s for 60 frames) -- cached.
if [ -d "$FRAMES_DEFLARED" ] && [ "$(ls -A "$FRAMES_DEFLARED")" ]; then
    echo "Reusing cached deflared frames at $FRAMES_DEFLARED"
else
    uv run --extra recon python src/reconstruction/deflare_dark_channel.py \
        "$NS_BASE/work/frames" "$FRAMES_DEFLARED" \
        --masks-dir "$NS_BASE/work/ns_data_masked/masks" \
        --sanity-grid "$PROJECT_DIR/outputs/figures/mots_nopathplanning/deflare_sanity_check.png"
fi

# 2. Build the deflared nerfstudio dataset (reuses baseline's camera poses
#    and eval split, only images/ differs) -- cached.
if [ -f "$DEFLARE_DATA/transforms.json" ]; then
    echo "Reusing cached deflared dataset at $DEFLARE_DATA"
else
    uv run --extra recon python src/reconstruction/prepare_deflared_dataset.py \
        "$SRC_DATA" "$FRAMES_DEFLARED" "$DEFLARE_DATA"
fi

# 3. Train splatfacto, identical config/eval-split flags to
#    nopathplanning_masked so the only variable is the pixel correction.
uv run --extra recon-ns ns-train splatfacto \
    --data "$DEFLARE_DATA" \
    --output-dir "$NS_BASE" \
    --experiment-name nopathplanning_deflare \
    --max-num-iterations 30000 \
    --vis tensorboard \
    nerfstudio-data --eval-mode interval --eval-interval 8

echo "Done. Deflare training complete."
