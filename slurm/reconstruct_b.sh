#!/bin/bash
#SBATCH --job-name=recon_b
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --output=logs/recon_b/%A_%a.out
#SBATCH --error=logs/recon_b/%A_%a.err

# B-type run: same frames and COLMAP gate as the characterisation run, then train_gs.py (gsplat) for
# 30,000 iterations on the largest model, every 8th frame of the frame list held out, PSNR / SSIM / LPIPS
# (AlexNet).
#   sbatch --array=0 slurm/reconstruct_b.sh      # sequence index from scripts/characterisation/sequences.py
# Keeps metrics, held-out renders (JPEG q90), poses and timings; checkpoints and point clouds are discarded.
# After the COLMAP gate the model and frames are undistorted to a pinhole camera (the trainer has no lens
# model). SEEDS="0 1 2" trains several seeds on the one SfM model; results then go to <out>/seed<k>/ and
# only the first seed keeps its renders. SPLITS="interp block" trains both held-out rules on the one SfM
# model (results in <out>/<split>/): interp = every 8th frame of the frame list from index 0; block = blocks
# of four consecutive frames from frame 16, every 32. Training uses about one core and matching costs the
# same core-hours at any width, so submit with --cpus-per-task=2. Switches for checks only: UNDISTORT=0,
# INIT_SCALE=fixed, OUT_ROOT=dir, TAG=_suffix.

set -uo pipefail
module load CUDA/13.0.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
# gsplat compiles its CUDA extension on first use and needs ninja from the project environment
export PATH=$PROJECT_DIR/.venv/bin:$PATH
ITERS=${ITERS:-30000}
UNDISTORT=${UNDISTORT:-1}
INIT_SCALE=${INIT_SCALE:-knn}
SEEDS=${SEEDS:-${SEED:-0}}
SPLITS=${SPLITS:-interp}
COLMAP_LIMIT=${COLMAP_LIMIT:-10800}
OUT_ROOT=${OUT_ROOT:-$PROJECT_DIR/results/full_program/B}
TAG=${TAG:-}

SEQ=$SLURM_ARRAY_TASK_ID
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/b_$SEQ
RES=$WORK/res
CHAR=$PROJECT_DIR/scripts/characterisation
cd "$CHAR"
ID=$($PY -c "from sequences import SEQUENCES; print(SEQUENCES[$SEQ]['id'])")
OUT=$OUT_ROOT/$ID$TAG
mkdir -p "$WORK" "$RES/frame_lists" "$OUT" "$PROJECT_DIR/logs/recon_b"
# reuse the committed frame list so every experiment on a sequence sees the same frames
cp "$PROJECT_DIR/results/characterisation/frame_lists/$ID.txt" "$RES/frame_lists/" 2>/dev/null
echo "B run: sequence $SEQ = $ID, iters $ITERS, undistort $UNDISTORT, init scale $INIT_SCALE, seeds $SEEDS, $(date)"; nvidia-smi -L
t0=$(date +%s)
$PY prepare.py "$SEQ" "$WORK" "$RES" || { echo "prepare failed"; exit 1; }
t1=$(date +%s)
timeout --signal=KILL "$COLMAP_LIMIT" $PY register.py "$WORK" "$RES" --threads "$SLURM_CPUS_PER_TASK"
t2=$(date +%s)
REG=$RES/per_sequence/$ID.registration.json
cp "$REG" "$RES/poses/$ID.csv" "$OUT/" 2>/dev/null
SPARSE=$($PY -c "import json; print(json.load(open('$REG')).get('best_sparse_dir',''))")
[ -d "$SPARSE" ] || { echo "no sparse model for $ID"; exit 1; }
# GATE_MIN: train only when the largest model holds at least this share of the frames
# (registration row is already persisted above); 0 = always train on the largest model
if [ "${GATE_MIN:-0}" != 0 ]; then
    PASS=$($PY -c "
import json; r = json.load(open('$REG'))
print(1 if max(r.get('model_sizes') or [0]) / r['n_images'] >= ${GATE_MIN} else 0)")
    [ "$PASS" = 1 ] || { echo "gate: largest model under ${GATE_MIN} of frames; registration row only"; exit 0; }
fi
IMAGES=$WORK/images
EXTRA=""
if [ "$UNDISTORT" = 1 ]; then
    SPARSE=$($PY undistort.py "$SPARSE" "$WORK/images" "$WORK/undist") || { echo "undistortion failed"; exit 1; }
    IMAGES=$WORK/undist/images
else
    EXTRA="--allow-distorted"
fi
# held-out names are fixed per sequence from the frame list, whatever registers
$PY - "$RES/frame_lists/$ID.txt" "$WORK" <<'PYEOF'
import sys
names = [ln.split("\t")[0].strip() for ln in open(sys.argv[1]) if ln.strip()]
n, work = len(names), sys.argv[2]
rules = {"interp": (names[0::8], "every 8th frame of the frame list, from index 0")}
if n < 36:
    lo = n // 2 - 2
    rules["block"] = (names[lo:lo + 4], f"middle four frames ({lo}-{lo + 3}) of {n}; sequence under 36 frames")
else:
    # a block that would touch the end of the sequence has training views on one side only: dropped
    starts = [s for s in range(16, n, 32) if s + 4 < n]
    rules["block"] = ([names[i] for s in starts for i in range(s, s + 4)],
                      "blocks of four consecutive frames from frame 16, every 32 (16-19, 48-51, ...); "
                      "a block touching the end of the sequence is dropped")
for split, (held, rule) in rules.items():
    open(f"{work}/holdout_{split}.txt", "w").write("\n".join(held) + "\n")
    open(f"{work}/rule_{split}.txt", "w").write(rule)
    print(f"split {split}: {len(held)} held-out names ({rule})")
PYEOF
SIZE=$($PY -c "import sys; from pathlib import Path; from PIL import Image; print(list(Image.open(sorted(Path(sys.argv[1]).rglob('*.png'))[0]).size))" "$IMAGES")
t3=$(date +%s)

cd "$PROJECT_DIR"
N_SEEDS=$(echo $SEEDS | wc -w)
N_SPLITS=$(echo $SPLITS | wc -w)
FAILED=0
for SPLIT in $SPLITS; do
RENDERS=all
for S in $SEEDS; do
    T=$WORK/train_${SPLIT}_s$S
    RUN_OUT=$OUT
    [ "$N_SPLITS" -gt 1 ] && RUN_OUT=$RUN_OUT/$SPLIT
    [ "$N_SEEDS" -gt 1 ] && RUN_OUT=$RUN_OUT/seed$S
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$T.gpu_mem.txt" &
    SMI=$!
    ts=$(date +%s)
    $PY src/reconstruction/train_gs.py "$SPARSE" "$IMAGES" "$T" --iters "$ITERS" --holdout-names "$WORK/holdout_$SPLIT.txt" \
        --init-scale "$INIT_SCALE" --seed "$S" $EXTRA 2>&1 | grep -v -E '^Step [0-9]+:' > "$T.log"
    te=$(date +%s)
    kill $SMI 2>/dev/null
    grep -m1 "Initial Gaussian size" "$T.log"; tail -4 "$T.log"
    if [ ! -d "$T/eval" ]; then echo "training failed for split $SPLIT seed $S"; tail -30 "$T.log"; FAILED=1; continue; fi
    $PY src/reconstruction/compute_metrics.py "$T/eval"
    tm=$(date +%s)
    $PY "$CHAR/persist_run.py" "$T" "$RUN_OUT" "$REG" --renders "$RENDERS" --gpu-mem-log "$T.gpu_mem.txt" --set \
        iters="$ITERS" undistorted="$UNDISTORT" init_scale="\"$INIT_SCALE\"" seed="$S" train_image_size="$SIZE" \
        cpus="$SLURM_CPUS_PER_TASK" t_prepare_s=$((t1 - t0)) t_colmap_job_s=$((t2 - t1)) t_undistort_s=$((t3 - t2)) \
        t_train_s=$((te - ts)) t_eval_s=$((tm - te)) job="\"${SLURM_ARRAY_JOB_ID}_$SEQ\"" \
        split="\"$SPLIT\"" split_rule="\"$(cat "$WORK/rule_$SPLIT.txt")\""
    RENDERS=pred
    rm -rf "$T"
done
done
echo "done $ID $(date)"
exit $FAILED
