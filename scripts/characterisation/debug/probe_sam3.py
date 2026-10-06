import os, sys
from pathlib import Path
SG = Path("/lustre/pd03/plgrid/plgdragons/synthetic-grapes")
sys.path.insert(0, str(SG / "vendor/sam3"))
os.environ.setdefault("HF_HOME", str(SG / ".cache/huggingface"))
frames = sys.argv[1]
def describe(x, depth=0, name="root"):
    import torch
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{name}: dict keys={list(x.keys())}")
        if depth < 3:
            for k, v in x.items(): describe(v, depth + 1, str(k))
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{name}: {type(x).__name__} len={len(x)}")
        if x and depth < 3: describe(x[0], depth + 1, "[0]")
    elif isinstance(x, torch.Tensor):
        print(f"{pad}{name}: Tensor {tuple(x.shape)} {x.dtype}")
    else:
        print(f"{pad}{name}: {type(x).__name__} {str(x)[:80]}")
from sam3.model_builder import build_sam3_video_predictor
p = build_sam3_video_predictor()
for prompt in (dict(text="grape"), dict(text="robot"),
               dict(bounding_boxes=[[0.0, 0.62, 0.35, 0.38]], bounding_box_labels=[1])):
    r = p.handle_request(request=dict(type="start_session", resource_path=frames))
    sid = r["session_id"]
    resp = p.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, **prompt))
    print("=" * 30, prompt)
    describe(resp, name="add_prompt response")
    for i, item in enumerate(p.handle_stream_request(request=dict(
            type="propagate_in_video", session_id=sid, propagation_direction="forward", start_frame_index=0))):
        if i == 0: describe(item, name="first propagate item")
        if i > 1: break
    p.handle_request(request=dict(type="close_session", session_id=sid))
