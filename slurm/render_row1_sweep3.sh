#!/bin/bash
#SBATCH --job-name=render_row1_sw3
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/render_row1_sweep/%j.out
#SBATCH --error=logs/render_row1_sweep/%j.err

# Third sweep: target the green-blob floater below the first row.
# Best candidate so far: fov28, shift=[0.135,-0.770,-1.107].
# Try higher min-opacity (0.20 / 0.30) to prune the low-density floater cloud.
# Also try a slightly lighter shift (s100) with fov25 as a middle-ground option.

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
mkdir -p "$PROJECT_DIR/logs/render_row1_sweep" "$PROJECT_DIR/outputs/teaser/row1"
cd "$PROJECT_DIR"

MODEL="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/mots_pathplanning_1/splatfacto/2026-06-16_001706/nerfstudio_models/step-000029999.ckpt"
TRANSFORMS="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/work/ns_data/transforms.json"

render() {
    local FOV=$1 SX=$2 SY=$3 SZ=$4 MINOP=$5 TAG=$6
    echo "=== FOV=${FOV} shift=[${SX},${SY},${SZ}] op>=${MINOP} → row1_${TAG}.png ==="
    uv run --extra recon python src/visualization/render_gs_views.py \
        --model           "$MODEL" \
        --transforms-json "$TRANSFORMS" \
        --output          "$PROJECT_DIR/outputs/teaser/row1/row1_${TAG}.png" \
        --width  1280 --height 960 \
        --fov    "$FOV" \
        --dist-scale 1.0 \
        --min-opacity "$MINOP" \
        --az 80 --el 35 \
        --lookat-shift "$SX" "$SY" "$SZ"
}

# Best angle from sweep2: fov28, s135 — now vary opacity threshold
render 28  0.135 -0.770 -1.107  0.20  fov28_s135_op020
render 28  0.135 -0.770 -1.107  0.30  fov28_s135_op030
# Also try fov22 with op=0.20 (tighter zoom, maybe cleaner)
render 22  0.135 -0.770 -1.107  0.20  fov22_s135_op020
# Middle-ground: fov25, lighter shift, op=0.15
render 25  0.100 -0.570 -0.820  0.15  fov25_s100_op015

echo "Done. Check outputs/teaser/row1/"
