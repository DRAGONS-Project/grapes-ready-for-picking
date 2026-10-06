#!/bin/bash
#SBATCH --job-name=render_teaser
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/render_teaser/%j.out
#SBATCH --error=logs/render_teaser/%j.err

# Render a teaser PNG: MOTS 3DGS vineyard + Franka Panda arm in Genesis.
#
# Prereqs:
#   1. Reconstruction run completed (model.pt exists).
#   2. Genesis venv reinstalled:
#        cd /home/grid/users/plgmwlodarzc/synthetic-grapes
#        bash slurm/SLURM_script.sh --script slurm/install_genesis.sh \
#            --partition plgrid-lem-gpu-h100 --gpu 1 --cpus 1 --mem 32GB --time 01:00:00
#
# Output: outputs/teaser/teaser.png (1920×1080)
# To try different angles without rebuilding the mesh, add:
#   --azimuth <deg> --elevation <deg> --robot-side left|right|front|back

set -euo pipefail
module load CUDA/12.4.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
GENESIS_VENV=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction/venv/.venv-genesis

mkdir -p "$PROJECT_DIR/logs/render_teaser"
cd "$PROJECT_DIR"

if [[ ! -x "$GENESIS_VENV/bin/python" ]]; then
    echo "ERROR: Genesis venv not found at $GENESIS_VENV" >&2
    echo "       Reinstall via synthetic-grapes/slurm/install_genesis.sh first." >&2
    exit 1
fi

MODEL="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1/runs/5355824/model.pt"
OUTPUT="$PROJECT_DIR/outputs/teaser/teaser.png"

"$GENESIS_VENV/bin/python" src/visualization/teaser_gs_genesis.py \
    --model    "$MODEL" \
    --output   "$OUTPUT" \
    --azimuth  -35 \
    --elevation 28 \
    --robot-side right \
    --width    1920 \
    --height   1080

echo "Done. Output: $OUTPUT"
