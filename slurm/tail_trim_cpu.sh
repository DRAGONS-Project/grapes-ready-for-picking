#!/bin/bash
#SBATCH --job-name=tail_cpu
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:30:00
#SBATCH --output=logs/tail_trim/%A_%a.cpu.out
#SBATCH --error=logs/tail_trim/%A_%a.cpu.err

# BLT standing-tail ablation, CPU stage: one arm per array task. The repeat-study instrument
# unchanged (COLMAP seed r, PYTHONHASHSEED r, 4 threads), on the frames before the standing
# tail; every run that passes the 50 % gate is undistorted and staged for tail_trim_gpu.sh.
set -uo pipefail
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env
export OPENBLAS_NUM_THREADS=1
RUNS_ROOT=${RUNS_ROOT:-$PROJECT_DIR/results/ablations/blt_tail/runs}
STAGE_ROOT=${STAGE_ROOT:-$HOME/blt_tail_stage}
#   condition              source   mask   runs  keep frames before
DEFAULT_SPECS=(
  "sep15_front_raw_trim   seq:53   none   5     95"
  "sep15_front_hand_trim  seq:53   greek  5     95"
  "jul13_front_raw_trim   seq:52   none   5     131"
  "jul13_front_hand_trim  seq:52   greek  5     131"
)
if [ -n "${TAIL_SPECS:-}" ]; then IFS=';' read -r -a SPECS <<< "$TAIL_SPECS"; else SPECS=("${DEFAULT_SPECS[@]}"); fi
read -r COND SRC MASK RUNS KEEP <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/tail
mkdir -p "$WORK" "$PROJECT_DIR/logs/tail_trim" "$STAGE_ROOT/$COND"
LOSS=""; [ "$MASK" != none ] && LOSS="--loss-masks"
cd "$PROJECT_DIR/scripts/characterisation"
echo "tail trim: $COND ($SRC, mask $MASK, $RUNS runs, frames < $KEEP), $(date)"
$PY repeat_registration.py "$COND" "$SRC" "$MASK" "$RUNS" "$WORK" "$RUNS_ROOT/$COND" \
    --threads "$SLURM_CPUS_PER_TASK" --keep-before "$KEEP" --stage "$STAGE_ROOT/$COND" $LOSS
echo "staged: $(ls "$STAGE_ROOT/$COND" | tr '\n' ' ')"
echo "done $COND $(date)"
