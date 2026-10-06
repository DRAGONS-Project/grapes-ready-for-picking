"""Thin wrapper around `ns-train` that patches torch.load for PyTorch 2.6+.

PyTorch 2.6 changed the default of `weights_only` from False to True.
Nerfstudio checkpoints contain numpy globals (numpy._core.multiarray.scalar)
that are rejected under weights_only=True.  This wrapper registers those
globals as safe before nerfstudio imports anything.

Usage (mirrors ns-train exactly):
    uv run --extra recon-ns python src/reconstruction/ns_train_resume.py \
        splatfacto --load-config <config.yml> [other ns-train flags...]
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
from nerfstudio.scripts.train import entrypoint
sys.exit(entrypoint())
