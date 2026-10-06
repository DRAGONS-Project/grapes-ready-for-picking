"""Time-boxed SAM 3 attempt: segment the robot chassis in the BLT front-pass frames.

Tries, in order: a concept (text) prompt on the video, then a box prompt from the
static-median map. Every failure is recorded in the report; the job's time limit is the
time box. Masks are written white = keep (chassis black), named like the frames.

usage: sam3_chassis.py <frames_dir> <out_masks_dir> <report.json>
       [--text robot] [--box x0,y0,x1,y1] [--iou-min 0.9]
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

SG = Path("/lustre/pd03/plgrid/plgdragons/synthetic-grapes")
sys.path.insert(0, str(SG / "vendor/sam3"))

report = {"steps": [], "outcome": None}


def log(step, **kw):
    report["steps"].append(dict(step=step, t=round(time.time() - T0, 1), **kw))
    print(json.dumps(report["steps"][-1]), flush=True)


T0 = time.time()


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("report", type=Path)
    ap.add_argument("--text", default="robot")
    ap.add_argument("--box", default=None, help="x0,y0,x1,y1 fallback box prompt")
    ap.add_argument("--iou-min", type=float, default=0.9)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in a.frames.glob("*.png")) or sorted(
        p.name for p in a.frames.glob("*.jpg"))
    log("frames", n=len(names), first=names[0] if names else None)

    import os
    os.environ.setdefault("HF_HOME", str(SG / ".cache/huggingface"))
    try:
        import torch
        log("torch", version=torch.__version__, cuda=torch.cuda.is_available())
        from sam3.model_builder import build_sam3_video_predictor
        predictor = build_sam3_video_predictor()
        log("model_built")
    except Exception as e:
        log("build_failed", error=f"{type(e).__name__}: {e}")
        report["outcome"] = "failed: model could not be built"
        a.report.write_text(json.dumps(report, indent=1))
        raise SystemExit(1) from None

    def masks_from_outputs(outputs):
        """union of the per-object binary masks -> bool HxW or None; outputs carry numpy
        arrays under out_binary_masks (probed structure; empty arrays = no objects)"""
        if not isinstance(outputs, dict):
            return None, type(outputs).__name__
        m = outputs.get("out_binary_masks")
        keys = list(outputs.keys())
        if m is None:
            return None, keys
        m = np.asarray(m)
        if m.size == 0:
            return None, keys
        if m.ndim > 2:
            m = m.reshape((-1, *m.shape[-2:])).any(axis=0)
        return m.astype(bool), keys

    def run_prompt(kind, **prompt):
        resp = predictor.handle_request(request=dict(type="start_session",
                                                     resource_path=str(a.frames)))
        sid = resp["session_id"]
        r1 = predictor.handle_request(request=dict(type="add_prompt", session_id=sid,
                                                   frame_index=0, **prompt))
        m0, keys = masks_from_outputs(r1.get("outputs"))
        log("add_prompt", kind=kind, outputs_keys=str(keys)[:200],
            frame0_share=None if m0 is None else round(float(m0.mean()), 4))
        per = {}
        if m0 is not None:
            per[0] = m0
        for item in predictor.handle_stream_request(
                request=dict(type="propagate_in_video", session_id=sid,
                             propagation_direction="forward", start_frame_index=0)):
            m, _ = masks_from_outputs(item.get("outputs"))
            if m is not None:
                per[int(item["frame_index"])] = m
        if len(per) < len(names) * 0.95 and kind == "box":
            # the tracker does not carry the box-prompted object; the chassis is rigid in
            # the frame, so prompt every frame with the same box instead of tracking
            log("propagation_sparse", n=len(per), fallback="per-frame box prompts")
            for j in range(len(names)):
                r = predictor.handle_request(request=dict(type="add_prompt", session_id=sid,
                                                          frame_index=j, **prompt))
                m, _ = masks_from_outputs(r.get("outputs"))
                if m is not None:
                    per[int(r.get("frame_index", j))] = m
        predictor.handle_request(request=dict(type="close_session", session_id=sid))
        return per

    first = cv2.imread(str(a.frames / names[0]))
    H, W = first.shape[:2]
    attempts = [("text", dict(text=a.text))]
    if a.box:
        x0, y0, x1, y1 = [int(v) for v in a.box.split(",")]
        # the API takes relative xywh boxes
        attempts.append(("box", dict(bounding_boxes=[[x0 / W, y0 / H,
                                                      (x1 - x0) / W, (y1 - y0) / H]],
                                     bounding_box_labels=[1])))
    chosen = None
    for kind, prompt in attempts:
        try:
            per = run_prompt(kind, **prompt)
            if not per:
                log("prompt_empty", kind=kind, prompt=str(prompt))
                continue
            idx = sorted(per)
            ious, shares = [], []
            for p, q in itertools.pairwise(idx):
                A, B = per[p], per[q]
                inter, union = (A & B).sum(), (A | B).sum()
                ious.append(inter / union if union else 1.0)
            shares = [float(per[i].mean()) for i in idx]
            log("prompt_result", kind=kind, prompt=str(prompt), n=len(per),
                iou_consec_min=round(float(np.min(ious)), 4) if ious else None,
                iou_consec_median=round(float(np.median(ious)), 4) if ious else None,
                area_share_median=round(float(np.median(shares)), 4))
            ok = len(per) >= len(names) * 0.95 and ious and float(np.min(ious)) >= a.iou_min
            if ok and chosen is None:
                chosen = (kind, prompt, per)
        except Exception as e:
            log("prompt_failed", kind=kind, error=f"{type(e).__name__}: {e}")
    if chosen is None:
        report["outcome"] = "failed: no prompt met the stability check"
        a.report.write_text(json.dumps(report, indent=1))
        raise SystemExit(1)
    kind, prompt, per = chosen
    for j, name in enumerate(names):
        m = per.get(j)
        if m is None:
            m = np.zeros_like(next(iter(per.values())))
        keep = (~m).astype(np.uint8) * 255  # white = keep, chassis excluded
        cv2.imwrite(str(a.out / name), keep)
    report["outcome"] = f"ok: {kind} prompt {prompt}"
    report["n_masks"] = len(names)
    a.report.write_text(json.dumps(report, indent=1))
    log("done", outcome=report["outcome"])


if __name__ == "__main__":
    main()
