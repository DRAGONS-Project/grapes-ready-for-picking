"""Thin wrapper around `ns-eval` that patches torch.load for PyTorch 2.6+.

Same PyTorch 2.6 weights_only issue as ns_train_resume.py, hit by
eval_utils.eval_load_checkpoint instead of the trainer's own loader.

Usage (mirrors ns-eval exactly):
    uv run --extra recon-ns python src/reconstruction/ns_eval_patched.py \
        --load-config <config.yml> --output-path <metrics.json>
"""

import sys
import torch
import numpy._core.multiarray

# Allowlist the numpy scalar type present in nerfstudio checkpoints
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

# Patch torch.load to fall back to weights_only=False if still needed
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

# Hand off to nerfstudio's own entrypoint
from nerfstudio.scripts.eval import entrypoint
sys.exit(entrypoint())
