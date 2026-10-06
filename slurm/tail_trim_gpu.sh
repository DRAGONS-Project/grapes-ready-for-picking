#!/bin/bash
#SBATCH --job-name=tail_gpu
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --output=logs/tail_trim/%A_%a.gpu.out
#SBATCH --error=logs/tail_trim/%A_%a.gpu.err

# BLT standing-tail ablation, GPU stage: one arm per array task (same order as
# tail_trim_cpu.sh; submit with --dependency=aftercorr:<cpu job>). Each staged run gets one
# training, trainer seed 0, 30,000 iterations; held-out views = the every-8th names of the
# committed frame list that lie before the trim, so the views are those the untrimmed arms share.
set -uo pipefail
module load CUDA/13.0.0
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1
export PATH=$PROJECT_DIR/.venv/bin:$PATH
ITERS=${ITERS:-30000}
RUNS_ROOT=${RUNS_ROOT:-$PROJECT_DIR/results/ablations/blt_tail/runs}
STAGE_ROOT=${STAGE_ROOT:-$HOME/blt_tail_stage}
OUT_ROOT=${OUT_ROOT:-$PROJECT_DIR/results/ablations/blt_tail/train}
CONDS=(${TAIL_CONDS:-sep15_front_raw_trim sep15_front_hand_trim jul13_front_raw_trim jul13_front_hand_trim})
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/tailgpu
mkdir -p "$WORK" "$PROJECT_DIR/logs/tail_trim"
cd "$PROJECT_DIR"
echo "tail trim training: $COND, $(date)"
FAILED=0
shopt -s nullglob
for STAGE in "$STAGE_ROOT/$COND"/run_*; do
    R=${STAGE##*_}
    RJ=$RUNS_ROOT/$COND/run_$R.json
    read -r ID KEEP SFM <<< "$($PY -c "import json; d=json.load(open('$RJ')); print(d['sequence'], d['keep_before'], int(d['mask_kind'] != 'none'))")"
    HOLDOUT=$WORK/holdout_$R.txt
    $PY - "$PROJECT_DIR/results/characterisation/frame_lists/$ID.txt" "$KEEP" > "$HOLDOUT" <<'EOF'
import re, sys
names = [ln.split("\t")[0] for ln in open(sys.argv[1]) if ln.strip()]
for i, n in enumerate(names):
    if i % 8 == 0 and int(re.search(r"(\d+)\.\w+$", n).group(1)) < int(sys.argv[2]):
        print(n)
EOF
    MASKARG=""; [ -d "$STAGE/loss_masks" ] && MASKARG="--mask-dir $STAGE/loss_masks"
    T=$WORK/train_r$R
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$T.gpu_mem.txt" &
    SMI=$!
    ts=$(date +%s)
    $PY src/reconstruction/train_gs.py "$STAGE/sparse" "$STAGE/images" "$T" --iters "$ITERS" \
        --holdout-names "$HOLDOUT" --seed 0 $MASKARG 2>&1 | grep -v -E '^Step [0-9]+:' > "$T.log"
    te=$(date +%s)
    kill $SMI 2>/dev/null
    tail -3 "$T.log"
    if [ ! -d "$T/eval" ]; then echo "training failed: run $R"; tail -25 "$T.log"; FAILED=1; continue; fi
    $PY src/reconstruction/compute_metrics.py "$T/eval"
    $PY scripts/characterisation/persist_run.py "$T" "$OUT_ROOT/$COND/run_$R" "$STAGE/registration.json" \
        --renders all --gpu-mem-log "$T.gpu_mem.txt" --set iters="$ITERS" undistorted=1 \
        init_scale='"knn"' seed=0 colmap_run="$R" condition="\"$COND\"" keep_before="$KEEP" split='"interp"' \
        split_rule='"every 8th frame of the committed frame list, from index 0, frames before the trim only"' \
        loss_masked=$([ -n "$MASKARG" ] && echo true || echo false) sfm_masked=$([ "$SFM" = 1 ] && echo true || echo false) \
        t_train_s=$((te - ts)) job="\"${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}\""
    cp "$HOLDOUT" "$OUT_ROOT/$COND/run_$R/holdout_names.txt"
    rm -rf "$T" "$STAGE"
done
echo "done $COND $(date)"
exit $FAILED
