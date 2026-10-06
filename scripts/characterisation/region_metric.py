"""Task-relevant region metric on MOTS: score held-out renders inside the dataset's grape
cluster masks and outside them, per view.

The cluster masks live in the MOTS annotation zips (default/instances/frame_<global>.png,
one per source video frame); single members are read over HTTP range, cached under
results/regions/_mask_cache/. Each run's mask is resized to the frame geometry, undistorted
with the camera stored in that run's registration.json (replicates the training-time warp;
verified to the pixel on mots_NoPathPlanning_1), and the per-pixel SSIM and LPIPS maps are
averaged over each region. Scores come from the persisted q90 JPEG renders, which adds a
small recompression bias; it applies to both regions alike.

usage: region_metric.py <run_dir ...> [--out results/regions/mots_regions.csv]
A run dir holds eval_renders/ and metrics_full.json; registration.json is found in the run
dir or its parents.
"""

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent.parent
ZIP_URL = "https://zenodo.org/api/records/10625595/files/{key}/content"


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = (off if whence == 0 else self.pos + off if whence == 1
                    else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:  # Zenodo rate-limits bursts of range requests
                if e.code != 429 or attempt == 5:
                    raise
                time.sleep(20 * (attempt + 1))
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


_zips: dict[str, tuple[zipfile.ZipFile, str, int]] = {}


def _open_zip(key: str):
    """(zip, instances prefix, index offset): the three NoPathPlanning zips index their
    frames continuously across the videos (NPP_1 from 0, NPP_2 from 1050, NPP_3 from 1950,
    the last under 'default-2/'), so member index = video frame + the zip's minimum index."""
    if key not in _zips:
        z = zipfile.ZipFile(io.BufferedReader(
            HttpFile(ZIP_URL.format(key=key)), buffer_size=2 << 20))
        inst = [n for n in z.namelist() if "/instances/" in n and n.endswith(".png")]
        prefix = inst[0].rsplit("/", 1)[0]
        offset = min(int(re.search(r"frame_(\d+)", n).group(1)) for n in inst)
        _zips[key] = (z, prefix, offset)
    return _zips[key]


def cluster_mask(base_id: str, global_idx: int, cache: Path) -> np.ndarray | None:
    """Union of instance masks for one source frame, native size, bool."""
    out = cache / base_id / f"frame_{global_idx:06d}.png"
    if not out.exists():
        key = base_id.removeprefix("mots_") + ".zip"
        z, prefix, offset = _open_zip(key)
        member = f"{prefix}/frame_{global_idx + offset:06d}.png"
        try:
            data = z.read(member)
        except KeyError:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    arr = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    # KITTI-MOTS ids: class * 1000 + instance; labels.txt: 1 = grape, 2 = trunk, 3 = pole
    return (arr >= 1000) & (arr < 2000)


def run_camera(run: Path):
    """The run's original camera and the undistorted geometry, from registration.json."""
    import pycolmap
    for d in (run, run.parent, run.parent.parent):
        found = list(d.glob("*registration.json")) + list(d.glob("registration.json"))
        if found:
            reg = json.loads(found[0].read_text())
            cam_info = next(iter(reg["cameras"].values()))
            cam = pycolmap.Camera(model="OPENCV", width=cam_info["width"],
                                  height=cam_info["height"], params=cam_info["params"],
                                  camera_id=1)
            return reg["id"], cam
    raise FileNotFoundError(f"no registration.json near {run}")


def undistort_mask(mask_u8: np.ndarray, cam) -> np.ndarray:
    import pycolmap
    bmp = pycolmap.Bitmap.from_array(np.dstack([mask_u8] * 3))
    ub, _ = pycolmap.undistort_image(pycolmap.UndistortCameraOptions(), bmp, cam)
    return ub.to_array()[..., 0] > 127


def region_scores(gt: np.ndarray, pred: np.ndarray, keep: np.ndarray, ssim_map, lpips_map):
    mse = float(((gt - pred) ** 2)[keep].mean())
    inner = np.zeros_like(keep)
    inner[3:-3, 3:-3] = keep[3:-3, 3:-3]
    sel = inner if inner.any() else keep
    return dict(psnr=float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
                ssim=float(ssim_map[sel].mean()), lpips=float(lpips_map[keep].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=REPO / "results/regions/mots_regions.csv")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import lpips
    import torch
    from skimage.metrics import structural_similarity
    loss_fn = lpips.LPIPS(net="alex", spatial=True).to(a.device)
    cache = REPO / "results/regions/_mask_cache"

    rows = []
    for run in [r.resolve() for r in a.runs]:
        mf = run / "metrics_full.json"
        if not mf.exists():
            print(f"skip {run}: no metrics_full.json")
            continue
        rid, cam = run_camera(run)
        base_id = re.sub(r"_c_.*$", "", rid)
        labels = {}
        flist = REPO / "results/characterisation/frame_lists" / f"{base_id}.txt"
        for ln in flist.read_text().splitlines():
            if ln.strip():
                name, lab = ln.split("\t")
                g = re.search(r"global (\d+)", lab)
                labels[name] = int(g.group(1)) if g else None
        for pv in json.loads(mf.read_text())["per_view"]:
            img_name = re.sub(r"\.png$", "", pv["name"])  # eval names carry an extra .png
            gidx = labels.get(img_name)
            if gidx is None:
                print(f"skip {run} {img_name}: not in frame list")
                continue
            m = cluster_mask(base_id, gidx, cache)
            if m is None:
                print(f"skip {run} {img_name}: no instance mask for global {gidx}")
                continue
            gt_p = run / "eval_renders" / f"gt_{Path(img_name).stem}.jpg"
            pr_p = run / "eval_renders" / f"pred_{Path(img_name).stem}.jpg"
            if not gt_p.exists():
                print(f"skip {run} {img_name}: no kept render")
                continue
            gt = cv2.imread(str(gt_p))[..., ::-1].astype(np.float32) / 255.0
            pred = cv2.imread(str(pr_p))[..., ::-1].astype(np.float32) / 255.0
            mu8 = cv2.resize((m * 255).astype(np.uint8), (cam.width, cam.height),
                             interpolation=cv2.INTER_NEAREST)
            keep = undistort_mask(mu8, cam)
            if keep.shape != gt.shape[:2]:
                keep = cv2.resize(keep.astype(np.uint8), gt.shape[1::-1],
                                  interpolation=cv2.INTER_NEAREST) > 0
            _, ssim_map = structural_similarity(gt, pred, data_range=1.0, channel_axis=2,
                                                full=True)
            ssim_map = ssim_map.mean(axis=2)
            t = (torch.from_numpy(np.stack([gt, pred])).permute(0, 3, 1, 2) * 2 - 1).to(a.device)
            with torch.no_grad():
                lpips_map = loss_fn(t[:1], t[1:])[0, 0].cpu().numpy()
            row = dict(run=str(run.relative_to(REPO)), id=rid, view=img_name,
                       global_index=gidx, cluster_share=float(keep.mean()))
            if keep.any():
                row.update({f"{k}_cluster": v for k, v in
                            region_scores(gt, pred, keep, ssim_map, lpips_map).items()})
            if (~keep).any():
                row.update({f"{k}_canopy": v for k, v in
                            region_scores(gt, pred, ~keep, ssim_map, lpips_map).items()})
            rows.append(row)
            print(f"{rid} {img_name} cluster {row['cluster_share']:.3f} "
                  f"psnr_c {row.get('psnr_cluster', float('nan')):.2f} "
                  f"psnr_k {row.get('psnr_canopy', float('nan')):.2f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run", "id", "view", "global_index", "cluster_share",
            "psnr_cluster", "ssim_cluster", "lpips_cluster",
            "psnr_canopy", "ssim_canopy", "lpips_canopy"]
    exists = a.out.exists()
    old = []
    if exists:
        old = [r for r in csv.DictReader(a.out.open())
               if r["run"] not in {str(x["run"]) for x in rows}]
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(old + rows)
    print(f"{len(rows)} new view rows -> {a.out}")


if __name__ == "__main__":
    main()
