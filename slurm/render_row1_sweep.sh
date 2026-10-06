#!/bin/bash
#SBATCH --job-name=render_row1_sweep
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/render_row1_sweep/%j.out
#SBATCH --error=logs/render_row1_sweep/%j.err

# Sweep lookat shift + FOV to isolate the first vine row.
# Camera Y direction at az=80 el=35 (nerfstudio space): [+0.10, -0.57, -0.82].
# Positive shift in this direction moves the first row toward image centre.
#
# We also try FOV=28° (zoom ~1.7×) vs FOV=35° and two shift magnitudes.

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
mkdir -p "$PROJECT_DIR/logs/render_row1_sweep" "$PROJECT_DIR/outputs/teaser/row1"
cd "$PROJECT_DIR"

MODEL="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/mots_pathplanning_1/splatfacto/2026-06-16_001706/nerfstudio_models/step-000029999.ckpt"
TRANSFORMS="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/work/ns_data/transforms.json"

render() {
    local FOV=$1 SX=$2 SY=$3 SZ=$4 TAG=$5
    echo "=== FOV=${FOV} shift=[${SX},${SY},${SZ}] → row1_${TAG}.png ==="
    uv run --extra recon python src/visualization/render_gs_views.py \
        --model           "$MODEL" \
        --transforms-json "$TRANSFORMS" \
        --output          "$PROJECT_DIR/outputs/teaser/row1/row1_${TAG}.png" \
        --width  1280 --height 960 \
        --fov    "$FOV" \
        --dist-scale 1.0 \
        --min-opacity 0.10 \
        --az 80 --el 35 \
        --lookat-shift "$SX" "$SY" "$SZ"
}

# Direction [0.10, -0.57, -0.82] normalised; scale by 0.15 and 0.30 of scene radius.
# Scene typical radius ~0.7 units in nerfstudio space → shifts of ~0.11 and 0.21.
render 35   0.015 -0.085 -0.123  fov35_shift015   # mild zoom, mild shift
render 28   0.015 -0.085 -0.123  fov28_shift015   # stronger zoom, mild shift
render 28   0.030 -0.171 -0.246  fov28_shift030   # stronger zoom, stronger shift
render 22   0.030 -0.171 -0.246  fov22_shift030   # very tight zoom

echo "Done. Check outputs/teaser/row1/"
