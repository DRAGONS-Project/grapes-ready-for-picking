#!/bin/bash
#SBATCH --job-name=render_row1_sw4
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

# Sweep4: clip-bbox ON, much smaller lookat shifts.
# Previous shift=1.35 was too large — the vine canopy got pushed out of frame.
# Now try no-shift, shift=0.3, shift=0.6, with bbox_pad=0.10 for scene boundary.

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
mkdir -p "$PROJECT_DIR/logs/render_row1_sweep" "$PROJECT_DIR/outputs/teaser/row1"
cd "$PROJECT_DIR"

MODEL="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/mots_pathplanning_1/splatfacto/2026-06-16_001706/nerfstudio_models/step-000029999.ckpt"
TRANSFORMS="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/work/ns_data/transforms.json"

render() {
    local FOV=$1 SX=$2 SY=$3 SZ=$4 PAD=$5 TAG=$6
    echo "=== FOV=${FOV} shift=[${SX},${SY},${SZ}] pad=${PAD} → ${TAG}.png ==="
    uv run --extra recon python src/visualization/render_gs_views.py \
        --model           "$MODEL" \
        --transforms-json "$TRANSFORMS" \
        --output          "$PROJECT_DIR/outputs/teaser/row1/${TAG}.png" \
        --width  1280 --height 960 \
        --fov    "$FOV" \
        --dist-scale 1.0 \
        --min-opacity 0.10 \
        --az 80 --el 35 \
        --lookat-shift "$SX" "$SY" "$SZ" \
        --clip-bbox --bbox-pad "$PAD"
}

# No shift — bbox clip removes the green blob, first row stays at bottom of frame
render 35   0.0   0.0   0.0   0.10  row1_fov35_s0_clip
render 28   0.0   0.0   0.0   0.10  row1_fov28_s0_clip

# Shift = 0.3 units in cam-Y direction: slight downward pull toward first row
render 28   0.030 -0.171 -0.246  0.10  row1_fov28_s030_clip
render 22   0.030 -0.171 -0.246  0.10  row1_fov22_s030_clip

# Shift = 0.6 units: moderate pull
render 28   0.060 -0.342 -0.492  0.10  row1_fov28_s060_clip

echo "Done. Check outputs/teaser/row1/"
