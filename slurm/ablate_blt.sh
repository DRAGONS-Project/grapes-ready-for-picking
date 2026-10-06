#!/bin/bash
#SBATCH --job-name=abl_blt
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:30:00
#SBATCH --output=logs/ablate_blt/%A_%a.out
#SBATCH --error=logs/ablate_blt/%A_%a.err

# One BLT masking condition per array task. BLT_SPECS entries:
#   <seq_index>:<condition>:<mask_kind>:<sfm_mask>:<loss_mask>
# CPU stage via ablate_blt.py; SEEDS trainings when the gate passes; renders seed0=all later=pred.
set -uo pipefail
module load CUDA/13.0.0
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1
export PATH=$PROJECT_DIR/.venv/bin:$PATH
ITERS=${ITERS:-30000}
SEEDS=${SEEDS:-"0 1 2"}
read -r -a ARR <<< "$BLT_SPECS"
IFS=':' read -r SEQ COND KIND SFM LOSS <<< "${ARR[$SLURM_ARRAY_TASK_ID]}"
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/blt
ROOT=${OUT_ROOT:-$PROJECT_DIR/results/full_program/C/blt}
mkdir -p "$WORK" "$PROJECT_DIR/logs/ablate_blt"
cd "$PROJECT_DIR/scripts/characterisation"
ID=$($PY -c "from sequences import SEQUENCES; print(SEQUENCES[$SEQ]['id'])")
echo "BLT condition: $ID / $COND (mask $KIND, sfm $SFM, loss $LOSS), $(date)"
$PY ablate_blt.py "$SEQ" "$COND" "$KIND" "$SFM" "$LOSS" "$WORK" "$ROOT" --threads "$SLURM_CPUS_PER_TASK" || { echo "cpu stage failed"; exit 1; }
COND_DIR=$ROOT/$ID/$COND
STAGE=$COND_DIR/_stage
[ -d "$STAGE" ] || { echo "registration row only; done"; exit 0; }
HOLDOUT=$WORK/holdout.txt
awk 'NR % 8 == 1 {print $1}' "$PROJECT_DIR/results/characterisation/frame_lists/$ID.txt" > "$HOLDOUT"
MASKARG=""; [ -d "$STAGE/loss_masks" ] && MASKARG="--mask-dir $STAGE/loss_masks"
cd "$PROJECT_DIR"
FAILED=0
RENDERS=all
for S in $SEEDS; do
    T=$WORK/train_s$S
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$T.gpu_mem.txt" &
    SMI=$!
    ts=$(date +%s)
    $PY src/reconstruction/train_gs.py "$STAGE/sparse" "$STAGE/images" "$T" --iters "$ITERS" \
        --holdout-names "$HOLDOUT" --seed "$S" $MASKARG 2>&1 | grep -v -E '^Step [0-9]+:' > "$T.log"
    te=$(date +%s)
    kill $SMI 2>/dev/null
    tail -3 "$T.log"
    if [ ! -d "$T/eval" ]; then echo "training failed: seed $S"; tail -25 "$T.log"; FAILED=1; continue; fi
    $PY src/reconstruction/compute_metrics.py "$T/eval"
    $PY scripts/characterisation/persist_run.py "$T" "$COND_DIR/seed$S" "$COND_DIR/registration.json" \
        --renders "$RENDERS" --gpu-mem-log "$T.gpu_mem.txt" --set iters="$ITERS" undistorted=1 \
        init_scale='"knn"' seed="$S" condition="\"$COND\"" split='"interp"' \
        split_rule='"every 8th frame of the frame list, from index 0"' \
        loss_masked=$([ -n "$MASKARG" ] && echo true || echo false) sfm_masked=$([ "$SFM" = 1 ] && echo true || echo false) \
        t_train_s=$((te - ts)) job="\"${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}\""
    RENDERS=pred
    rm -rf "$T"
done
rm -rf "$STAGE"
echo "done $ID/$COND $(date)"
exit $FAILED
