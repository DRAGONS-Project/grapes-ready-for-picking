#!/bin/bash
#SBATCH --job-name=nopathplan_deflare_ghostmask
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/reconstruct_nopathplanning_deflare_ghostmasked/%j.out
#SBATCH --error=logs/reconstruct_nopathplanning_deflare_ghostmasked/%j.err

# Lens-flare fix, attempt 4: layer lens-ghost mask exclusion on top of the
# (bug-fixed) dark-channel deflare arm. Deflaring corrects the whole-frame
# veiling-glare haze but not the lens ghost -- a discrete secondary internal-
# lens reflection outside the dark-channel-prior's single-veil model, which
# dehaze's contrast/saturation boost actually makes MORE visible, not less.
# generate_ghost_masks.py (Approach A: local excess-green elevated over its
# own large-radius baseline, gated on low small-radius V texture) isolates
# that ghost region tightly across the sequence in visual inspection; the
# multi-view color-anomaly alternative (Approach B, flare_multiview_outliers.py
# --score-mode color) was tested side by side and does not concentrate on the
# ghost -- its flagged points don't rank above the general view-to-view color
# noise floor even after fixing a MAD-floor bug, likely because COLMAP only
# triangulates ~35% of expected point density in the ghost's smooth interior
# to begin with. See docs/flare_fix_research.md.
#
# Same camera poses/eval split as baseline/masked/bilateral/deflare
# (transforms.json reused verbatim); images/ point at the (fixed) deflared
# frames and masks/ exclude the ghost region from the loss.

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
NS_BASE=$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1
DEFLARE_GHOSTMASKED_DATA=$NS_BASE/work/ns_data_deflare_ghostmasked

mkdir -p "$PROJECT_DIR/logs/reconstruct_nopathplanning_deflare_ghostmasked"
cd "$PROJECT_DIR"

# Dataset (deflared images + ghost masks wired into transforms.json) is
# already built via prepare_deflared_ghostmasked_dataset.py -- fail loudly if
# it's missing rather than silently regenerating with stale inputs.
if [ ! -f "$DEFLARE_GHOSTMASKED_DATA/transforms.json" ]; then
    echo "Missing $DEFLARE_GHOSTMASKED_DATA/transforms.json -- run prepare_deflared_ghostmasked_dataset.py first." >&2
    exit 1
fi

uv run --extra recon-ns ns-train splatfacto \
    --data "$DEFLARE_GHOSTMASKED_DATA" \
    --output-dir "$NS_BASE" \
    --experiment-name nopathplanning_deflare_ghostmasked \
    --max-num-iterations 30000 \
    --vis tensorboard \
    nerfstudio-data --eval-mode interval --eval-interval 8

echo "Done. Deflare+ghostmasked training complete."
