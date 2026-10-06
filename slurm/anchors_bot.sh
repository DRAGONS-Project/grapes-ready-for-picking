#!/bin/bash
#SBATCH --job-name=anchors_bot
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=03:00:00
#SBATCH --output=logs/anchors/%j.out
#SBATCH --error=logs/anchors/%j.err
# D1: per-capture EXIF GPS streamed from the deposit zips (whole members; cached per flight),
# then Sim(3) alignment of each composed-RGB quality row.
set -uo pipefail
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env
cd "$PROJECT_DIR"
for f in 45_V1 0_V1 30_V1 0_V2; do
    id=bot_${f}_rgb
    $PY scripts/characterisation/anchors_eval.py bot \
        "results/full_program/B/$id/$id.csv" "$id" --run "B/$id"
done
