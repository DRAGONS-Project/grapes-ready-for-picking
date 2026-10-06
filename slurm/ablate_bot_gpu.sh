#!/bin/bash
#SBATCH --job-name=abl_bot_gpu
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablate_bot/gpu_%j.out
#SBATCH --error=logs/ablate_bot/gpu_%j.err

# Botrytis ablations, GPU stage for one flight: 3 seeds per staged (gate-passing) condition,
# interp split, renders kept for seed 0 (gt+pred) and seeds 1-2 (pred only). Cleans the stage.
set -uo pipefail
module load CUDA/13.0.0
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1
export PATH=$PROJECT_DIR/.venv/bin:$PATH
FLIGHT=${FLIGHT:?set FLIGHT}
ITERS=${ITERS:-30000}
ROOT=$PROJECT_DIR/results/full_program/C/botrytis/$FLIGHT
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID
HOLDOUT=$WORK/holdout.txt
awk 'NR % 8 == 1 {print $1}' "$PROJECT_DIR/results/characterisation/frame_lists/bot_$FLIGHT.txt" > "$HOLDOUT"
cd "$PROJECT_DIR"
FAILED=0
for STAGE in "$ROOT"/*/_stage; do
    [ -d "$STAGE" ] || continue
    COND_DIR=$(dirname "$STAGE"); COND=$(basename "$COND_DIR")
    echo "== $FLIGHT/$COND $(date)"
    RENDERS=all
    for S in 0 1 2; do
        T=$WORK/train_${COND}_s$S
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$T.gpu_mem.txt" &
        SMI=$!
        ts=$(date +%s)
        $PY src/reconstruction/train_gs.py "$STAGE/sparse" "$STAGE/images" "$T" --iters "$ITERS" \
            --holdout-names "$HOLDOUT" --seed "$S" 2>&1 | grep -v -E '^Step [0-9]+:' > "$T.log"
        te=$(date +%s)
        kill $SMI 2>/dev/null
        tail -3 "$T.log"
        if [ ! -d "$T/eval" ]; then echo "training failed: $COND seed $S"; tail -25 "$T.log"; FAILED=1; continue; fi
        $PY src/reconstruction/compute_metrics.py "$T/eval"
        $PY scripts/characterisation/persist_run.py "$T" "$COND_DIR/seed$S" "$COND_DIR/registration.json" \
            --renders "$RENDERS" --gpu-mem-log "$T.gpu_mem.txt" --set iters="$ITERS" undistorted=1 \
            init_scale='"knn"' seed="$S" condition="\"$COND\"" split='"interp"' \
            split_rule='"every 8th frame of the frame list, from index 0"' \
            t_train_s=$((te - ts)) job="\"$SLURM_JOB_ID\""
        RENDERS=pred
        rm -rf "$T"
    done
    rm -rf "$STAGE"
done
echo "done $FLIGHT $(date)"
exit $FAILED
