#!/bin/bash
#SBATCH --job-name=composite_robot_gs
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/composite_robot_gs/%j.out
#SBATCH --error=logs/composite_robot_gs/%j.err

# Render Franka Panda against green-screen in Genesis, then composite onto
# the three best GS eval renders (04, 05, 06) at 1280x960.

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
GENESIS_VENV="$PROJECT_DIR/venv/.venv-genesis"

mkdir -p "$PROJECT_DIR/logs/composite_robot_gs" "$PROJECT_DIR/outputs/teaser/composite"
cd "$PROJECT_DIR"

if [[ ! -x "$GENESIS_VENV/bin/python" ]]; then
    echo "ERROR: Genesis venv not found at $GENESIS_VENV" >&2
    exit 1
fi

"$GENESIS_VENV/bin/python" src/visualization/composite_robot_gs.py \
    --backgrounds \
        outputs/teaser/teaser_eval_04.png \
        outputs/teaser/teaser_eval_05.png \
        outputs/teaser/teaser_eval_06.png \
    --output-dir  outputs/teaser/composite \
    --azimuth     80 \
    --elevation   35 \
    --fov         50 \
    --robot-side  right \
    --robot-scale 0.40 \
    --paste-x     0.75 \
    --paste-y     0.92 \
    --render-res  1280 960

echo "Done. Composites: outputs/teaser/composite/"
