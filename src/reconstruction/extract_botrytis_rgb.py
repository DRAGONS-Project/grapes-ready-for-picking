"""Build RGB frames from a Micasense RedEdge-MX multispectral capture zip (Botrytis dataset).

Each capture is a set of 5 single-band 16-bit TIFFs (band 1=Blue, 2=Green, 3=Red,
4=NIR, 5=Red Edge) under ``<flight>/SET/IMG_<id>_<band>.tif``. We compose bands
3/2/1 into an 8-bit RGB image.

The RedEdge-MX is 5 physically separate lenses on the camera body (not one
sensor behind a Bayer filter), each with a slightly different optical axis;
merging the raw bands directly produces severe per-edge chromatic
ghosting/fringing from the inter-lens parallax (confirmed against the
captures' own XMP metadata: no MicaSense RigRelatives calibration is embedded
in this dataset's TIFFs, so alignment has to come from image registration, not
a calibration lookup). We correct this with a single ORB-feature-matched
homography per band (Blue/Red -> Green), estimated once from a robust sample
of captures and reused for the whole flight -- the offset is a fixed property
of the rig at a given altitude, not scene-dependent, and per-frame estimation
was measured to fail outright (garbage homographies) on ~15% of captures
tested, typically low-texture ones with too few ORB inliers.

A second, independent problem: composing each band with its own independent
percentile stretch (as an earlier version of this script did) distorts
relative RGB ratios inconsistently across a frame, producing a false
magenta/pink cast over soil and shadowed canopy even after alignment is
fixed. A joint percentile stretch (one shared low/high computed across all
three bands together) plus a gray-world white-balance pass corrects most of
this -- see docs/datasets/multispectral_botrytis.md for the investigation.
"""

import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np

from config import load_config

# Micasense RedEdge-MX band order.
BLUE, GREEN, RED = 1, 2, 3


def _stretch_u8(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(band, [low, high])
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((band.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _read_band(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        raw = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_ANYDEPTH)


def estimate_band_alignment(
    zf: zipfile.ZipFile, ids: list[str], calib_n: int = 25, min_inliers: int = 80
) -> dict[int, np.ndarray]:
    """Robust one-time per-band homography (Blue/Red -> Green), the median over
    many captures, each individually filtered by RANSAC inlier count so a
    handful of low-texture failures can't corrupt the estimate."""
    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    rng = np.random.default_rng(0)
    if len(ids) <= calib_n:
        sample_ids = ids
    else:
        idx = sorted(rng.choice(len(ids), calib_n, replace=False))
        sample_ids = [ids[i] for i in idx]

    homographies: dict[int, list[np.ndarray]] = {BLUE: [], RED: []}
    for stem in sample_ids:
        ref8 = _stretch_u8(_read_band(zf, f"{stem}_{GREEN}.tif"))
        kp_ref, des_ref = orb.detectAndCompute(ref8, None)
        if des_ref is None or len(des_ref) < 10:
            continue
        for band in (BLUE, RED):
            src8 = _stretch_u8(_read_band(zf, f"{stem}_{band}.tif"))
            kp, des = orb.detectAndCompute(src8, None)
            if des is None or len(des) < 10:
                continue
            matches = sorted(bf.match(des, des_ref), key=lambda m: m.distance)[:500]
            if len(matches) < 10:
                continue
            src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if H is None or mask is None or int(mask.sum()) < min_inliers:
                continue
            homographies[band].append(H)

    ref_H = {}
    for band, hs in homographies.items():
        if not hs:
            raise RuntimeError(
                f"Could not calibrate band {band} alignment: no sampled capture passed the "
                f"{min_inliers}-inlier threshold. Try a larger --calib-n."
            )
        ref_H[band] = np.median(np.array(hs), axis=0)
        print(f"  band {band}: {len(hs)}/{len(sample_ids)} captures used, "
              f"shift=({ref_H[band][0, 2]:.1f}, {ref_H[band][1, 2]:.1f})")
    return ref_H


def compose_rgb(zf: zipfile.ZipFile, stem: str, ref_H: dict[int, np.ndarray], crop: int) -> np.ndarray:
    bands = {b: _read_band(zf, f"{stem}_{b}.tif").astype(np.float32) for b in (BLUE, GREEN, RED)}
    for band in (BLUE, RED):
        h, w = bands[band].shape
        bands[band] = cv2.warpPerspective(bands[band], ref_H[band], (w, h), flags=cv2.INTER_LINEAR)

    stacked = np.concatenate([bands[b].ravel() for b in (BLUE, GREEN, RED)])
    lo, hi = np.percentile(stacked, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1
    channels = [np.clip((bands[b] - lo) / (hi - lo), 0, 1) for b in (BLUE, GREEN, RED)]

    means = [c.mean() for c in channels]
    target = sum(means) / len(means)
    channels = [np.clip(c * (target / m), 0, 1) if m > 0 else c for c, m in zip(channels, means)]

    img = cv2.merge([(c * 255).astype(np.uint8) for c in channels])
    h, w = img.shape[:2]
    return img[crop: h - crop, crop: w - crop]


def extract_botrytis_rgb(
    zip_path: Path,
    output_dir: Path,
    max_dim: int | None = None,
    max_frames: int | None = None,
    crop: int = 30,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        ids = sorted(
            n[: -len(f"_{RED}.tif")]
            for n in zf.namelist()
            if "/SET/" in n and n.endswith(f"_{RED}.tif")
        )
        if max_frames and len(ids) > max_frames:
            step = len(ids) / max_frames
            ids = [ids[round(i * step)] for i in range(max_frames)]

        print(f"Calibrating band alignment from a sample of {len(ids)} captures...")
        ref_H = estimate_band_alignment(zf, ids)

        for i, stem in enumerate(ids):
            img = compose_rgb(zf, stem, ref_H, crop)
            if max_dim:
                h, w = img.shape[:2]
                scale = max_dim / max(h, w)
                if scale < 1:
                    img = cv2.resize(
                        img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
                    )
            cv2.imwrite(str(output_dir / f"frame_{i:06d}.jpg"), img)

    return len(ids)


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="Micasense capture zip (e.g. 45_V1.zip)")
    parser.add_argument("output_dir", type=Path, help="Directory to write composed RGB frames")
    parser.add_argument(
        "--max-dim", type=int, default=config.frames.max_dim, help="Resize longest image side to this many pixels"
    )
    parser.add_argument(
        "--max-frames", type=int, default=config.frames.max_frames, help="Cap on the number of extracted frames"
    )
    parser.add_argument(
        "--crop", type=int, default=30, help="Pixels cropped from each border to remove warp-induced edge artifacts"
    )
    args = parser.parse_args()

    n = extract_botrytis_rgb(args.zip_path, args.output_dir, args.max_dim, args.max_frames, args.crop)
    print(f"Composed {n} RGB frames to {args.output_dir}")


if __name__ == "__main__":
    main()
