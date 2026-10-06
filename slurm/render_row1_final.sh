#!/bin/bash
#SBATCH --job-name=render_row1_final
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

# Final sweep: add --clip-bbox to eliminate the below-ground green floater.
# Best angle from sweep2/3: fov28, shift=[0.135,-0.770,-1.107], op=0.10.
# Also try fov22 for a tighter crop on the vine canopy.

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
mkdir -p "$PROJECT_DIR/logs/render_row1_sweep" "$PROJECT_DIR/outputs/teaser/row1"
cd "$PROJECT_DIR"

MODEL="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/mots_pathplanning_1/splatfacto/2026-06-16_001706/nerfstudio_models/step-000029999.ckpt"
TRANSFORMS="$PROJECT_DIR/outputs/reconstruction/mots_pathplanning_1_splatfacto/work/ns_data/transforms.json"

render() {
    local FOV=$1 SX=$2 SY=$3 SZ=$4 MINOP=$5 PAD=$6 TAG=$7
    echo "=== FOV=${FOV} shift=[${SX},${SY},${SZ}] op>=${MINOP} pad=${PAD} → ${TAG}.png ==="
    uv run --extra recon python src/visualization/render_gs_views.py \
        --model           "$MODEL" \
        --transforms-json "$TRANSFORMS" \
        --output          "$PROJECT_DIR/outputs/teaser/row1/${TAG}.png" \
        --width  1280 --height 960 \
        --fov    "$FOV" \
        --dist-scale 1.0 \
        --min-opacity "$MINOP" \
        --az 80 --el 35 \
        --lookat-shift "$SX" "$SY" "$SZ" \
        --clip-bbox --bbox-pad "$PAD"
}

render 28  0.135 -0.770 -1.107  0.10  0.0   row1_fov28_s135_clip
render 28  0.135 -0.770 -1.107  0.10  0.1   row1_fov28_s135_clip_pad10
render 22  0.135 -0.770 -1.107  0.10  0.0   row1_fov22_s135_clip
render 25  0.100 -0.570 -0.820  0.10  0.0   row1_fov25_s100_clip

echo "Done. Check outputs/teaser/row1/"
