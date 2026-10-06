#!/bin/bash
#SBATCH --job-name=sam3_try
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/sam3/%j.out
#SBATCH --error=logs/sam3/%j.err

# One-hour time-boxed SAM 3 attempt on the UK front pass (the agreed time box IS the job limit).
# Frames come from the committed blt_uk_20230726_front extraction; masks white = keep.
set -uo pipefail
module load CUDA/13.0.0
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/sam3_env:$HOME/characterisation_env
export PATH=$PROJECT_DIR/.venv/bin:$PATH
export HF_HOME=/lustre/pd03/plgrid/plgdragons/synthetic-grapes/.cache/huggingface
export HF_HUB_OFFLINE=1
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/sam3
mkdir -p "$WORK" "$PROJECT_DIR/logs/sam3" "$PROJECT_DIR/results/masks/sam3"
cd "$PROJECT_DIR/scripts/characterisation"
# re-extract the committed front frames into the job (frame lists fix the choice)
$PY prepare.py 54 "$WORK" "$WORK/res"
cd "$PROJECT_DIR"
$PY scripts/characterisation/sam3_chassis.py "$WORK/images" \
    "$PROJECT_DIR/results/masks/sam3/blt_uk_20230726_front" \
    "$PROJECT_DIR/results/masks/sam3/blt_uk_20230726_front.report.json" \
    --text "robot" --box "0,560,560,900"
