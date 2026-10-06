#!/bin/bash
#SBATCH --job-name=composite_dist_sweep
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:12:00
#SBATCH --output=logs/composite_dist_sweep/%j.out
#SBATCH --error=logs/composite_dist_sweep/%j.err

# Strategy: render robot ONCE via Genesis (the expensive step), save robot_rgba.png,
# then do 4 distance/scale composites with pure PIL — no Genesis re-init.

set -euo pipefail
module load CUDA/12.6.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
GENESIS_VENV="$PROJECT_DIR/venv/.venv-genesis"
BG="$PROJECT_DIR/outputs/teaser/row1/row1_fov28_s030_clip.png"
OUTDIR="$PROJECT_DIR/outputs/teaser/dist_sweep"

mkdir -p "$PROJECT_DIR/logs/composite_dist_sweep" "$OUTDIR"
cd "$PROJECT_DIR"

# ── Step 1: render robot once against green screen (Genesis) ──────────────────
echo "=== Rendering robot green-screen (Genesis, once) ==="
"$GENESIS_VENV/bin/python" src/visualization/composite_robot_gs.py \
    --backgrounds   "$BG" \
    --output-dir    "$OUTDIR/tmp_render" \
    --azimuth       80 \
    --elevation     50 \
    --fov           50 \
    --dist          2.5 \
    --robot-side    right \
    --robot-scale   0.27 \
    --paste-x       0.58 \
    --paste-y       0.84 \
    --render-res    1280 960

# robot_rgba.png is saved next to the script's repo root
ROBOT_RGBA="$PROJECT_DIR/outputs/teaser/robot_rgba.png"
echo "Robot RGBA: $ROBOT_RGBA"

# ── Step 2: 4 composites at different scales (pure PIL, fast) ─────────────────
echo "=== Compositing 4 distance variants ==="
"$GENESIS_VENV/bin/python" - << PYEOF
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

robot_rgba = np.array(Image.open("$ROBOT_RGBA").convert("RGBA"))
bg_path    = Path("$BG")
outdir     = Path("$OUTDIR")
outdir.mkdir(parents=True, exist_ok=True)

variants = [
    ("far",    0.18),
    ("mid",    0.27),
    ("close",  0.36),
    ("vclose", 0.44),
]
paste_x, paste_y = 0.58, 0.84

def composite_one(bg_path, robot_rgba, paste_x, paste_y, robot_scale, out_path):
    bg_img = Image.open(bg_path).convert("RGB")
    bg = np.array(bg_img)
    H, W = bg.shape[:2]

    # Tight-crop robot to its bounding box
    alpha = robot_rgba[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        print(f"WARNING: empty robot mask, skipping {out_path}")
        return
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    robot_crop = robot_rgba[r0:r1+1, c0:c1+1]
    rh, rw = robot_crop.shape[:2]

    target_h = int(H * robot_scale)
    target_w = int(rw * target_h / rh)
    robot_img = Image.fromarray(robot_crop).resize((target_w, target_h), Image.LANCZOS)
    robot_arr = np.array(robot_img)

    x_center = int(W * paste_x)
    y_bottom  = int(H * paste_y)
    x0 = x_center - target_w // 2
    y0 = y_bottom  - target_h

    rx0 = max(0, -x0); x0 = max(0, x0)
    ry0 = max(0, -y0); y0 = max(0, y0)
    x1  = min(W, x0 + target_w - rx0)
    y1  = min(H, y0 + target_h - ry0)
    rx1 = rx0 + (x1 - x0)
    ry1 = ry0 + (y1 - y0)

    if x1 <= x0 or y1 <= y0:
        print(f"WARNING: paste out of bounds for {out_path}")
        return

    patch = robot_arr[ry0:ry1, rx0:rx1]
    a = patch[:, :, 3:4].astype(np.float32) / 255.0
    bg[y0:y1, x0:x1] = (a * patch[:, :, :3] + (1-a) * bg[y0:y1, x0:x1]).clip(0,255).astype(np.uint8)

    Image.fromarray(bg).save(str(out_path))
    print(f"Saved: {out_path}")

saved = []
for tag, scale in variants:
    out = outdir / f"dist_{tag}.png"
    composite_one(bg_path, robot_rgba, paste_x, paste_y, scale, out)
    saved.append((tag, scale, out))

# ── Step 3: stitch 2x2 comparison ─────────────────────────────────────────────
W, H = 1280, 960
thumb_w, thumb_h = W // 2, H // 2
gap = 8
grid_w = thumb_w * 2 + gap * 3
grid_h = thumb_h * 2 + gap * 3
grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))

try:
    font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 22)
except Exception:
    font = ImageFont.load_default()

labels = {"far": "Far (scale 0.18)", "mid": "Mid (scale 0.27)",
          "close": "Close (scale 0.36)", "vclose": "V.Close (scale 0.44)"}

for i, (tag, scale, path) in enumerate(saved):
    row, col = divmod(i, 2)
    img = Image.open(path).resize((thumb_w, thumb_h), Image.LANCZOS)
    x = gap + col * (thumb_w + gap)
    y = gap + row * (thumb_h + gap)
    grid.paste(img, (x, y))
    draw = ImageDraw.Draw(grid)
    draw.rectangle([x, y, x + thumb_w - 1, y + 26], fill=(0, 0, 0))
    draw.text((x + 6, y + 3), labels[tag], fill=(255, 255, 200), font=font)

out_grid = outdir / "comparison_dist_sweep.png"
grid.save(str(out_grid))
print(f"Grid saved: {out_grid}")
PYEOF

echo "Done. Check outputs/teaser/dist_sweep/"
