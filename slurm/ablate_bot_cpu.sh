#!/bin/bash
#SBATCH --job-name=abl_bot_cpu
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --output=logs/ablate_bot/cpu_%j_%x.out
#SBATCH --error=logs/ablate_bot/cpu_%j_%x.err

# Botrytis ablations, CPU stage for one flight (FLIGHT=45_V1 or 0_V1): download, compose every
# condition, COLMAP gate per condition, undistort, stage for the GPU stage.
set -uo pipefail
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env
export OPENBLAS_NUM_THREADS=1
FLIGHT=${FLIGHT:?set FLIGHT}
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/abl_$FLIGHT
mkdir -p "$WORK" "$PROJECT_DIR/logs/ablate_bot"
cd "$PROJECT_DIR/scripts/characterisation"
$PY ablate_bot.py "$FLIGHT" "$WORK" "$PROJECT_DIR/results/full_program/C/botrytis/$FLIGHT" \
    --threads "$SLURM_CPUS_PER_TASK" ${CONDITIONS:+--conditions "$CONDITIONS"}
