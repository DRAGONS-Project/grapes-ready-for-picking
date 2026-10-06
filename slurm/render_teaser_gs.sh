#!/bin/bash
#SBATCH --job-name=render_teaser_gs
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/render_teaser_gs/%j.out
#SBATCH --error=logs/render_teaser_gs/%j.err

# Render three close-angle candidates at 1280×960 for final teaser pick.
# Best angle from grid: ~35° elevation, East side.
# Candidates: az 80° / 90° / 110° at el 35° — covers the E-to-SE arc.

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
mkdir -p "$PROJECT_DIR/logs/render_teaser_gs" "$PROJECT_DIR/outputs/teaser"
cd "$PROJECT_DIR"

MODEL="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/mots_pathplanning_1/splatfacto/2026-06-16_001706/nerfstudio_models/step-000029999.ckpt"
TRANSFORMS="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/work/ns_data/transforms.json"

# Render at two opacity thresholds to compare floater suppression.
# Best angle: az=80° el=35° (diagonal row, two blue crates, dirt path visible).

render_final() {
    local MINOP=$1
    local OUT="$PROJECT_DIR/outputs/teaser/teaser_gs_op${MINOP//./}.png"
    echo "=== min-opacity=${MINOP} → $OUT ==="
    uv run --extra recon python src/visualization/render_gs_views.py \
        --model           "$MODEL" \
        --transforms-json "$TRANSFORMS" \
        --output          "$OUT" \
        --width      1280 \
        --height      960 \
        --fov          48 \
        --dist-scale  1.0 \
        --min-opacity "$MINOP" \
        --az  80 \
        --el  35
}

render_final 0.10
render_final 0.20

# Also produce a center crop of the 0.10 render to remove edge floaters:
python3 - <<'PYEOF'
from PIL import Image
img = Image.open("/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction/outputs/teaser/teaser_gs_op010.png")
w, h = img.size   # 1280×960
# Crop out 12% on each side horizontally, 8% top/bottom — keeps main vine rows
left   = int(w * 0.12)
right  = int(w * 0.88)
top    = int(h * 0.05)
bottom = int(h * 0.88)
crop = img.crop((left, top, right, bottom)).resize((1280, 960), Image.LANCZOS)
out = "/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction/outputs/teaser/teaser_gs_cropped.png"
crop.save(out)
print(f"Crop saved: {out}  (from {left},{top} → {right},{bottom}, upscaled to 1280×960)")
PYEOF

echo "Done."
