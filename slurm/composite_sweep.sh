#!/bin/bash
#SBATCH --job-name=composite_sweep
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/composite_sweep/%j.out
#SBATCH --error=logs/composite_sweep/%j.err

# Sweep elevation (20/35/50 deg) × distance (2m/4m) for the robot composite.
# All variants composited onto eval_04 (best background).
# Genesis is re-initialised per run via subprocess to avoid state issues.

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
GENESIS_VENV="$PROJECT_DIR/venv/.venv-genesis"

mkdir -p "$PROJECT_DIR/logs/composite_sweep" "$PROJECT_DIR/outputs/teaser/sweep"
cd "$PROJECT_DIR"

BG="$PROJECT_DIR/outputs/teaser/teaser_eval_04.png"

run_variant() {
    local EL=$1 DIST=$2
    local TAG="el${EL}_d${DIST}"
    local OUTDIR="$PROJECT_DIR/outputs/teaser/sweep/${TAG}"
    echo ""
    echo "=== el=${EL}° dist=${DIST}m → ${TAG} ==="
    "$GENESIS_VENV/bin/python" src/visualization/composite_robot_gs.py \
        --backgrounds "$BG" \
        --output-dir  "$OUTDIR" \
        --azimuth     80 \
        --elevation   "$EL" \
        --fov         50 \
        --dist        "$DIST" \
        --robot-side  right \
        --robot-scale 0.45 \
        --paste-x     0.72 \
        --paste-y     0.90 \
        --render-res  1280 960
    # Copy the composite to a flat name for easy comparison
    cp "$OUTDIR/composite_teaser_eval_04.png" \
       "$PROJECT_DIR/outputs/teaser/sweep/variant_${TAG}.png"
    echo "Saved: variant_${TAG}.png"
}

# Genesis can't be re-init in the same process; run each variant in its own subprocess.
run_variant  20  2
run_variant  20  4
run_variant  35  2
run_variant  35  4
run_variant  50  2
run_variant  50  4

echo ""
echo "All variants done. Check outputs/teaser/sweep/variant_*.png"
