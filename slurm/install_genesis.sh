#!/bin/bash
#SBATCH --job-name=install_genesis
#SBATCH --account=plgdragons
#SBATCH --qos=plgdragons
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/install_genesis/%j.out
#SBATCH --error=logs/install_genesis/%j.err

# Install Genesis (genesis-world 0.4.7) into venv/.venv-genesis on Lustre.
# Stored on Lustre (/lustre/pd03) to avoid home-directory inode/quota limits.

set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

PROJECT_DIR=/lustre/pd03/plgrid/plgdragons/vineyard-scene-reconstruction
VENV_STORE=$PROJECT_DIR/venv
GENESIS_VENV=$VENV_STORE/.venv-genesis

# uv cache on Lustre too — home quota can't absorb it.
export UV_CACHE_DIR=$VENV_STORE/.uv-cache
export UV_LINK_MODE=copy

mkdir -p "$VENV_STORE"

echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Store: $VENV_STORE"
echo ""

echo "=== [1/3] Creating Python 3.11 venv ==="
uv venv --python 3.11 --clear "$GENESIS_VENV"

echo "=== [2/3] Installing PyTorch 2.12.0+cu126 ==="
uv pip install --python "$GENESIS_VENV" \
    "torch==2.12.0" \
    "torchvision==0.27.0" \
    --index-url https://download.pytorch.org/whl/cu126 \
    2>&1 | grep -E "Installed|error" | head -10

echo "=== [3/3] Installing genesis-world + dependencies ==="
uv pip install --python "$GENESIS_VENV" \
    "genesis-world==0.4.7" \
    imageio \
    imageio-ffmpeg \
    open3d \
    pillow \
    2>&1 | grep -E "Installed|error" | head -20

"$GENESIS_VENV/bin/python" -c "
import genesis as gs
print('genesis', gs.__version__, 'OK')
print('CUDA backend available:', hasattr(gs, 'cuda'))
"

echo "Done. Venv: $GENESIS_VENV"
