#!/bin/bash
#SBATCH --job-name=render_teaser_genesis
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --output=logs/render_teaser_genesis/%j.out
#SBATCH --error=logs/render_teaser_genesis/%j.err

# Render 3 Genesis teaser variants: vineyard mesh + Franka Panda arm.
# Mesh is cached (teaser_mesh.obj) so only the Genesis render step runs.
# Produces three 1280×960 PNGs for the paper teaser figure selection.

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
GENESIS_VENV="$PROJECT_DIR/venv/.venv-genesis"

mkdir -p "$PROJECT_DIR/logs/render_teaser_genesis" "$PROJECT_DIR/outputs/teaser"
cd "$PROJECT_DIR"

if [[ ! -x "$GENESIS_VENV/bin/python" ]]; then
    echo "ERROR: Genesis venv not found at $GENESIS_VENV" >&2
    exit 1
fi

MODEL="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1/runs/5355824/model.pt"

render_variant() {
    local AZ=$1 EL=$2 SIDE=$3 SUFFIX=$4
    local OUT="$PROJECT_DIR/outputs/teaser/teaser_genesis_${SUFFIX}.png"
    echo ""
    echo "=== Variant ${SUFFIX}: az=${AZ} el=${EL} robot=${SIDE} → $OUT ==="
    "$GENESIS_VENV/bin/python" src/visualization/teaser_gs_genesis.py \
        --model        "$MODEL" \
        --output       "$OUT" \
        --azimuth      "$AZ" \
        --elevation    "$EL" \
        --robot-side   "$SIDE" \
        --width        1280 \
        --height        960
    echo "Done: $OUT"
}

# Variant A — default diagonal view, robot on right (along vine row)
render_variant -35 28 right "A_right_diag"

# Variant B — opposite diagonal, robot on left (other side of the row)
render_variant 145 25 left  "B_left_diag"

# Variant C — end-of-row view (looking down the row), robot on right
render_variant  55 20 right "C_right_endrow"

echo ""
echo "All 3 variants done. Check outputs/teaser/teaser_genesis_*.png"
