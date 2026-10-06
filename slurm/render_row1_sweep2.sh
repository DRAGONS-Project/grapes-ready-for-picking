#!/bin/bash
#SBATCH --job-name=render_row1_sw2
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

# Second sweep with correctly-scaled lookat shifts.
# Camera-Y direction (down in image) at az=80 el=35: [0.10, -0.57, -0.82]
# Scene P5-P95 extent [7.13, 4.64, 3.19]; dist ≈ 7.13 units.
# First row is ~30% below centre in fov35 render → need ~1.35 units shift.

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

# Shift direction (camera-Y): [0.10, -0.57, -0.82], magnitude 1.35 to center first row
render 35   0.135 -0.770 -1.107  fov35_s135   # FOV=35 + centred on first row
render 28   0.135 -0.770 -1.107  fov28_s135   # FOV=28 + centred
render 22   0.135 -0.770 -1.107  fov22_s135   # FOV=22 + centred (tight)
render 22   0.200 -1.140 -1.640  fov22_s200   # FOV=22 + pushed further into first row

echo "Done. Check outputs/teaser/row1/"
