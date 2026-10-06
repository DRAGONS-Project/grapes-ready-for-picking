#!/bin/bash
#SBATCH --job-name=abl_mots
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/ablate_mots/%A_%a.out
#SBATCH --error=logs/ablate_mots/%A_%a.err

# MOTS ablations for one sequence (array index = sequence index: 0 = NoPathPlanning_1 backlit,
# 1 = NoPathPlanning_2 control). CPU part via ablate_mots.py, then SEEDS trainings per staged
# condition; metrics full-frame and on the common pixels; renders seed 0 all, later seeds pred.
set -uo pipefail
module load CUDA/13.0.0
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1
export PATH=$PROJECT_DIR/.venv/bin:$PATH
ITERS=${ITERS:-30000}
SEEDS=${SEEDS:-"0 1 2"}
SEQ=$SLURM_ARRAY_TASK_ID
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/m_$SEQ
mkdir -p "$WORK" "$PROJECT_DIR/logs/ablate_mots"
cd "$PROJECT_DIR/scripts/characterisation"
ID=$($PY -c "from sequences import SEQUENCES; print(SEQUENCES[$SEQ]['id'])")
ROOT=$PROJECT_DIR/results/full_program/C/mots/$ID
$PY ablate_mots.py "$SEQ" "$WORK" "$ROOT" --threads "$SLURM_CPUS_PER_TASK" || { echo "cpu stage failed"; exit 1; }
HOLDOUT=$WORK/holdout.txt
awk 'NR % 8 == 1 {print $1}' "$PROJECT_DIR/results/characterisation/frame_lists/$ID.txt" > "$HOLDOUT"
cd "$PROJECT_DIR"
FAILED=0
for STAGE in "$ROOT"/*/_stage; do
    [ -d "$STAGE" ] || continue
    COND_DIR=$(dirname "$STAGE"); COND=$(basename "$COND_DIR")
    MASKARG=""; [ -d "$STAGE/loss_masks" ] && MASKARG="--mask-dir $STAGE/loss_masks"
    echo "== $ID/$COND masks:${MASKARG:-none} $(date)"
    RENDERS=all
    for S in $SEEDS; do
        T=$WORK/train_${COND}_s$S
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$T.gpu_mem.txt" &
        SMI=$!
        ts=$(date +%s)
        $PY src/reconstruction/train_gs.py "$STAGE/sparse" "$STAGE/images" "$T" --iters "$ITERS" \
            --holdout-names "$HOLDOUT" --seed "$S" $MASKARG 2>&1 | grep -v -E '^Step [0-9]+:' > "$T.log"
        te=$(date +%s)
        kill $SMI 2>/dev/null
        tail -3 "$T.log"
        if [ ! -d "$T/eval" ]; then echo "training failed: $COND seed $S"; tail -25 "$T.log"; FAILED=1; continue; fi
        $PY src/reconstruction/compute_metrics.py "$T/eval"
        $PY src/reconstruction/compute_metrics.py "$T/eval" --mask-dir "$STAGE/common_masks" --out metrics_common.json
        $PY scripts/characterisation/persist_run.py "$T" "$COND_DIR/seed$S" "$COND_DIR/registration.json" \
            --renders "$RENDERS" --gpu-mem-log "$T.gpu_mem.txt" --set iters="$ITERS" undistorted=1 \
            init_scale='"knn"' seed="$S" condition="\"$COND\"" split='"interp"' \
            split_rule='"every 8th frame of the frame list, from index 0"' \
            loss_masked=$([ -n "$MASKARG" ] && echo true || echo false) \
            t_train_s=$((te - ts)) job="\"${SLURM_ARRAY_JOB_ID}_$SEQ\""
        RENDERS=pred
        rm -rf "$T"
    done
    rm -rf "$STAGE"
done
echo "done $ID $(date)"
exit $FAILED
