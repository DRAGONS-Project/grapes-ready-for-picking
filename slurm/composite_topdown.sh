#!/bin/bash
#SBATCH --job-name=composite_topdown
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/composite_topdown/%j.out
#SBATCH --error=logs/composite_topdown/%j.err

# Top-down robot composite on cropped teaser_az80_el35.png.
# Robot: el=80° (near top-down), arm reaching toward vine row,
# sun-like lighting (single strong directional + cool fill).
# Background: teaser_az80_el35 cropped to [0.08,0.12 → 0.82,0.85].

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
GENESIS_VENV="$PROJECT_DIR/venv/.venv-genesis"

mkdir -p "$PROJECT_DIR/logs/composite_topdown" "$PROJECT_DIR/outputs/teaser/topdown"
cd "$PROJECT_DIR"

"$GENESIS_VENV/bin/python" src/visualization/composite_robot_gs.py \
    --backgrounds   outputs/teaser/teaser_az80_el35.png \
    --output-dir    outputs/teaser/topdown \
    --azimuth       80 \
    --elevation     80 \
    --fov           55 \
    --dist          2.0 \
    --robot-side    right \
    --robot-scale   0.38 \
    --paste-x       0.52 \
    --paste-y       0.72 \
    --crop  0.08 0.12 0.82 0.85 \
    --render-res    1280 960

echo "Done. Output: outputs/teaser/topdown/"
