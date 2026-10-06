#!/bin/bash
#SBATCH --job-name=nopathplan_mask
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/reconstruct_nopathplanning_masked/%j.out
#SBATCH --error=logs/reconstruct_nopathplanning_masked/%j.err

# Lens-flare fix, attempt 2: exclude the sunbeam/veiling-glare region from the
# training loss via a per-frame binary mask (see generate_flare_masks.py),
# instead of trying to model the appearance shift with a bilateral grid
# (nopathplanning_bilateral, evaluated worse than baseline: PSNR 19.89 vs
# 20.94 -- see eval_nopathplanning.sh). Same splatfacto config as
# nopathplanning_baseline (default hyperparameters, 30k iterations,
# use_bilateral_grid=false) so the only variable is the mask.

set -euo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
NS_BASE=$PROJECT_DIR/outputs/reconstruction/mots_nopathplanning_1
SRC_DATA=$NS_BASE/work/ns_data
MASKED_DATA=$NS_BASE/work/ns_data_masked

mkdir -p "$PROJECT_DIR/logs/reconstruct_nopathplanning_masked"
cd "$PROJECT_DIR"

# 1. Build the masked dataset (per-frame flare masks + transforms.json with
#    mask_path wired in) -- cached, cheap, safe to skip on reruns.
if [ -f "$MASKED_DATA/transforms.json" ]; then
    echo "Reusing cached masked dataset at $MASKED_DATA"
else
    uv run --extra recon python src/reconstruction/prepare_masked_dataset.py \
        "$SRC_DATA" "$MASKED_DATA"
fi

# 2. Train splatfacto, identical config to nopathplanning_baseline except for
#    the data dir (masks applied) and experiment name. eval-mode/eval-interval
#    pinned explicitly to match baseline/bilateral's held-out split exactly --
#    the nerfstudio version currently pinned in this repo defaults
#    eval_mode to "fraction" (6 held-out views), while baseline/bilateral
#    were trained under "interval" (8 held-out views, idx % 8 == 0); without
#    this flag the three runs are scored on different held-out frames.
uv run --extra recon-ns ns-train splatfacto \
    --data "$MASKED_DATA" \
    --output-dir "$NS_BASE" \
    --experiment-name nopathplanning_masked \
    --max-num-iterations 30000 \
    --vis tensorboard \
    nerfstudio-data --eval-mode interval --eval-interval 8

echo "Done. Masked training complete."
