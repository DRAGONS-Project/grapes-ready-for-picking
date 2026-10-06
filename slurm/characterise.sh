#!/bin/bash
#SBATCH --job-name=characterise
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=03:30:00
#SBATCH --gres=storage:lustre:1
#SBATCH --output=logs/characterise/%A_%a.out
#SBATCH --error=logs/characterise/%A_%a.err

# Dataset characterisation: one array task per sequence (see scripts/characterisation/sequences.py).
#   sbatch --array=0-24 slurm/characterise.sh            # main pass
#   sbatch --array=25-32 slurm/characterise.sh           # remaining BTG clips
# Optional: MAPPER_THREADS=1 (single-threaded mapping), DET_TEST=1 (map one database three ways: 1 thread twice, then all threads).

set -uo pipefail

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
OUT=$PROJECT_DIR/results/characterisation
PY=$PROJECT_DIR/.venv/bin/python
export PYTHONPATH=$HOME/characterisation_env
export PYTHONHASHSEED=0
# Work around an OpenBLAS thread-pool bug that can segfault pycolmap during incremental mapping.
export OPENBLAS_NUM_THREADS=1

SEQ=$SLURM_ARRAY_TASK_ID
WORK=${TMPDIR_LUSTRE:-/lustre/tmp/slurm/$SLURM_JOB_ID}/char_$SEQ
MAPPER_THREADS=${MAPPER_THREADS:--1}
mkdir -p "$WORK" "$OUT" "$PROJECT_DIR/logs/characterise"
cd "$PROJECT_DIR/scripts/characterisation"

ID=$($PY -c "from sequences import SEQUENCES; print(SEQUENCES[$SEQ]['id'])")
echo "sequence $SEQ = $ID, work dir $WORK, cores $SLURM_CPUS_PER_TASK, mapper threads $MAPPER_THREADS, $(date)"

# 1. fetch + sample + resize (frame lists are never regenerated once written)
$PY prepare.py "$SEQ" "$WORK" "$OUT" || { echo "prepare failed for $ID"; exit 1; }

# 2. frame-level quantities, static mask, contact sheet
$PY frame_quality.py "$WORK" "$OUT" || echo "frame_quality failed for $ID"

# 3. registration gate: 45 min limit, 120 min for the two full stills sets
case "$ID" in esc_photos|emb_stills) LIMIT=7200 ;; *) LIMIT=2700 ;; esac
EXTRA=""
[ "${DET_TEST:-0}" = "1" ] && { EXTRA="--det-test"; LIMIT=$((LIMIT * 2)); }
timeout --signal=KILL "$LIMIT" $PY register.py "$WORK" "$OUT" --threads "$SLURM_CPUS_PER_TASK" --mapper-threads "$MAPPER_THREADS" $EXTRA
rc=$?
if [ $rc -eq 137 ] || [ $rc -eq 124 ]; then
    $PY - "$OUT/per_sequence/$ID.registration.json" "$LIMIT" <<'PYEOF'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); d.update(outcome="timeout", wall_s=float(sys.argv[2]), n_registered=None)
json.dump(d, open(p, "w"), indent=1)
PYEOF
    echo "register timed out after $LIMIT s"
elif [ $rc -ne 0 ]; then
    echo "register exited with code $rc"
fi
echo "done $ID $(date)"
