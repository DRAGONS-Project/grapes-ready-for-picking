#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --output=logs/sweep/%A_%a.out
#SBATCH --error=logs/sweep/%A_%a.err

# One requirement-sweep run: SPECS is a space-separated list of <seq_index>/<kind:value>;
# the array index picks one. Gate + one training (seed 0, every 8th held out) + metrics,
# width levels also scored with renders and references resized to 640 px wide.
set -uo pipefail
module load CUDA/13.0.0
PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1
export PATH=$PROJECT_DIR/.venv/bin:$PATH
ITERS=${ITERS:-30000}
read -r -a ARR <<< "$SPECS"
ENTRY=${ARR[$SLURM_ARRAY_TASK_ID]}
SEQ=${ENTRY%%/*}
SPEC=${ENTRY#*/}
WORK=/lustre/tmp/slurm/$SLURM_JOB_ID/sweep
mkdir -p "$WORK" "$PROJECT_DIR/logs/sweep"
cd "$PROJECT_DIR/scripts/characterisation"
ID=$($PY -c "from sequences import SEQUENCES; print(SEQUENCES[$SEQ]['id'])")
OUT=$PROJECT_DIR/results/sweeps/$ID/${SPEC/:/_}
mkdir -p "$OUT"
echo "sweep: $ID $SPEC, $(date)"
$PY sweep_prepare.py "$SEQ" "$SPEC" "$WORK" || { echo "prepare failed"; exit 1; }
cp "$WORK/meta.json" "$WORK/frame_list.txt" "$OUT/"
$PY register.py "$WORK" "$WORK/res" --threads "$SLURM_CPUS_PER_TASK"
SID=$($PY -c "import json; print(json.load(open('$WORK/meta.json'))['id'])")
REG=$WORK/res/per_sequence/$SID.registration.json
cp "$REG" "$OUT/registration.json" 2>/dev/null; cp "$WORK/res/poses/$SID.csv" "$OUT/poses.csv" 2>/dev/null
SHARE=$($PY -c "
import json; r = json.load(open('$REG'))
print(1 if max(r.get('model_sizes') or [0]) / r['n_images'] >= 0.5 else 0)")
[ "$SHARE" = 1 ] || { echo "gate failed; registration row only"; exit 0; }
SPARSE=$($PY -c "import json; print(json.load(open('$REG'))['best_sparse_dir'])")
SPARSE=$($PY undistort.py "$SPARSE" "$WORK/images" "$WORK/undist")
HOLDOUT=$WORK/holdout.txt
awk 'NR % 8 == 1 {print $1}' "$WORK/frame_list.txt" > "$HOLDOUT"
cd "$PROJECT_DIR"
T=$WORK/train
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$T.gpu_mem.txt" &
SMI=$!
ts=$(date +%s)
$PY src/reconstruction/train_gs.py "$SPARSE" "$WORK/undist/images" "$T" --iters "$ITERS" \
    --holdout-names "$HOLDOUT" --seed 0 2>&1 | grep -v -E '^Step [0-9]+:' > "$T.log"
te=$(date +%s)
kill $SMI 2>/dev/null
tail -3 "$T.log"
[ -d "$T/eval" ] || { echo "training failed"; tail -25 "$T.log"; exit 1; }
$PY src/reconstruction/compute_metrics.py "$T/eval"
# the cross-level comparable column: everything resized to 640 px wide first
$PY - "$T/eval" <<'PYEOF'
import sys
from pathlib import Path
from PIL import Image
src = Path(sys.argv[1]); dst = src.parent / "eval640"; dst.mkdir(exist_ok=True)
for p in src.glob("*.png"):
    im = Image.open(p)
    im.resize((640, round(im.height * 640 / im.width)), Image.LANCZOS).save(dst / p.name)
PYEOF
$PY src/reconstruction/compute_metrics.py "$T/eval640" --out metrics_at640.json
cp "$T/eval640/metrics_at640.json" "$T/eval/" 2>/dev/null
$PY scripts/characterisation/persist_run.py "$T" "$OUT" "$OUT/registration.json" \
    --renders all --gpu-mem-log "$T.gpu_mem.txt" --set iters="$ITERS" undistorted=1 \
    init_scale='"knn"' seed=0 split='"interp"' sweep="\"$SPEC\"" \
    split_rule='"every 8th frame of the sweep frame list, from index 0"' \
    t_train_s=$((te - ts)) job="\"${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}\""
cp "$T/eval640/metrics_at640.json" "$OUT/" 2>/dev/null
echo "done $ID $SPEC $(date)"
