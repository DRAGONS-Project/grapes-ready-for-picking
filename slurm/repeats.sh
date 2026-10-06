#!/bin/bash
#SBATCH --job-name=repeats
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/repeats/%A_%a.out
#SBATCH --error=logs/repeats/%A_%a.err

# Registration repeat study: one condition per array task, N independent runs of the fixed
# COLMAP configuration (seed r, PYTHONHASHSEED r), 4 threads as in the runs being repeated.
set -uo pipefail
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env
export OPENBLAS_NUM_THREADS=1
#            condition            source              mask          runs
SPECS=(
  "uk_front_raw          seq:54              none          5"
  "uk_front_hand         seq:54              fixed_region  5"
  "jul13_front_raw       seq:52              none          5"
  "jul13_front_hand      seq:52              greek         5"
  "sep15_front_raw       seq:53              none          5"
  "sep15_front_hand      seq:53              greek         5"
  "mar23_side_raw        seq:3               none          5"
  "npp1_fps0.5           sweep:0:fps:0.5     none          5"
  "control_npp1_standard seq:0               none          3"
  "control_btg_row7.2_p3 seq:41              none          3"
)
# REPEAT_SPECS="cond src mask runs;..." replaces the table above (later conditions, same instrument)
if [ -n "${REPEAT_SPECS:-}" ]; then IFS=';' read -r -a SPECS <<< "$REPEAT_SPECS"; fi
read -r COND SRC MASK RUNS <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/rep
mkdir -p "$WORK" "$PROJECT_DIR/logs/repeats"
cd "$PROJECT_DIR/scripts/characterisation"
echo "repeat study: $COND ($SRC, mask $MASK, $RUNS runs), $(date)"
$PY repeat_registration.py "$COND" "$SRC" "$MASK" "$RUNS" "$WORK" \
    "$PROJECT_DIR/results/repeats/runs/$COND" --threads "$SLURM_CPUS_PER_TASK"
echo "done $COND $(date)"
