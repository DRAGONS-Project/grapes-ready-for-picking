"""Render side-by-side SfM point cloud comparison plots for the BLT dataset.

Produces a 2-column × 2-row figure:
  col 0 = raw (unmasked robot body), col 1 = preprocessed (masked)
  row 0 = top-down view (X–Z plane), row 1 = lateral view (X–Y plane)

Points are coloured with their per-point RGB from the COLMAP PLY export.
Outliers (beyond 3 σ) are clipped so a single stray point does not
dominate the axis limits.

Usage:
    uv run --extra recon python src/visualization/plot_blt_sfm.py \
        --raw      outputs/reconstruction/blt_sfm_comparison/blt_raw_sparse.ply \
        --prep     outputs/reconstruction/blt_sfm_comparison/blt_preprocessed_sparse.ply \
        --stats    outputs/reconstruction/blt_sfm_comparison/sfm_comparison.json \
        --output   outputs/figures/blt_sfm_comparison.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import open3d as o3d


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cloud(path: Path):
    pc = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pc.points, dtype=np.float32)
    if pc.has_colors():
        rgb = np.asarray(pc.colors, dtype=np.float32)   # already [0,1]
    else:
        rgb = np.full((len(pts), 3), 0.6, dtype=np.float32)
    return pts, rgb


def robust_limits(vals: np.ndarray, n_sigma: float = 3.0):
    """Return (lo, hi) clipped to ±n_sigma from median."""
    med = np.median(vals)
    std = np.std(vals)
    return med - n_sigma * std, med + n_sigma * std


def scatter_view(ax, x, y, rgb, title, xlabel, ylabel):
    # Sort by a proxy depth so darker/closer points draw on top:
    # use the 3rd omitted axis implicitly captured via rgb brightness.
    brightness = rgb.mean(axis=1)
    order = np.argsort(brightness)[::-1]   # dim points on top (paint background first)
    ax.scatter(x[order], y[order], c=rgb[order], s=1.2, linewidths=0,
               rasterized=True)
    xl = robust_limits(x)
    yl = robust_limits(y)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_facecolor("#1a1a1a")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    repo = Path(__file__).resolve().parents[2]
    base = repo / "outputs" / "reconstruction" / "blt_sfm_comparison"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",    default=base / "blt_raw_sparse.ply",    type=Path)
    p.add_argument("--prep",   default=base / "blt_preprocessed_sparse.ply", type=Path)
    p.add_argument("--stats",  default=base / "sfm_comparison.json",   type=Path)
    p.add_argument("--output", default=repo / "outputs" / "figures" / "blt_sfm_comparison.pdf",
                   type=Path)
    p.add_argument("--dpi",    default=200, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading point clouds …")
    raw_pts,  raw_rgb  = load_cloud(args.raw)
    prep_pts, prep_rgb = load_cloud(args.prep)

    stats = {}
    if args.stats.exists():
        with open(args.stats) as f:
            stats = json.load(f)

    def fmt_stats(key):
        if key not in stats:
            return ""
        s = stats[key]
        return (f"{s['num_reg_images']} images · {s['num_points3D']:,} pts · "
                f"reproj err {s['mean_reprojection_error']:.3f} px")

    clouds = [
        ("Raw (unmasked)",       raw_pts,  raw_rgb,  "raw"),
        ("Preprocessed (masked)", prep_pts, prep_rgb, "preprocessed"),
    ]

    # View definitions: (row_label, xi, yi, xlabel, ylabel)
    views = [
        ("Top (X–Z)",      0, 2, "X  [m]", "Z  [m]"),
        ("Side (X–Y)",     0, 1, "X  [m]", "Y  [m]"),
    ]

    fig = plt.figure(figsize=(8, 6), facecolor="white")
    fig.suptitle("BLT SfM sparse point clouds — raw vs. preprocessed",
                 fontsize=11, y=0.98)

    gs = gridspec.GridSpec(
        len(views), len(clouds),
        figure=fig,
        hspace=0.35, wspace=0.3,
        top=0.92, bottom=0.08, left=0.08, right=0.97,
    )

    for ci, (col_title, pts, rgb, stat_key) in enumerate(clouds):
        for ri, (row_label, xi, yi, xlabel, ylabel) in enumerate(views):
            ax = fig.add_subplot(gs[ri, ci])
            title = col_title if ri == 0 else ""
            scatter_view(ax, pts[:, xi], pts[:, yi], rgb,
                         title=title, xlabel=xlabel, ylabel=ylabel)
            if ri == 0:
                # Add stats as a subtitle below the column title
                sub = fmt_stats(stat_key)
                if sub:
                    ax.set_title(f"{col_title}\n{sub}", fontsize=7.5, fontweight="bold",
                                 linespacing=1.4)

    # Row labels on the left edge
    for ri, (row_label, *_) in enumerate(views):
        fig.text(0.01, 0.79 - ri * 0.46, row_label,
                 va="center", ha="left", fontsize=8,
                 rotation=90, color="#444")

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # Also save a PNG for quick preview
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
