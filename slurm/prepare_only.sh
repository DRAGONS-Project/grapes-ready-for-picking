#!/bin/bash
#SBATCH --job-name=prep_seq
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_seq/%A_%a.out
#SBATCH --error=logs/prep_seq/%A_%a.err

# Fetch + sampling only (no COLMAP): commits the frame list and meta for a new sequence to
# results/characterisation before any GPU run uses them.
#   sbatch --array=34-44%4 slurm/prepare_only.sh
set -uo pipefail
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env
export OPENBLAS_NUM_THREADS=1
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/prep_$SLURM_ARRAY_TASK_ID
mkdir -p "$WORK" "$PROJECT_DIR/logs/prep_seq"
cd "$PROJECT_DIR/scripts/characterisation"
$PY prepare.py "$SLURM_ARRAY_TASK_ID" "$WORK" "$PROJECT_DIR/results/characterisation"
