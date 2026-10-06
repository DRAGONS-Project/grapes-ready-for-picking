# Vineyard Scene Reconstruction

> Repurposing existing precision-viticulture UAV/ground-robot footage (of which none was collected with 3D reconstruction in mind) into 3D Gaussian Splatting scenes of real grapevine rows. Repurposing scenes for downstream goals of RL-based UAV/robot grape-sampling simulation. Stack: **pycolmap** (SfM) · **gsplat** / **Nerfstudio splatfacto** (3DGS) · **uv** · **SLURM**.

## 🍇 What this is

Public vineyard datasets (UAV video, multispectral UAV imagery, ground-robot RGB-D) were collected for detection, tracking, or disease mapping, not for radiance-field reconstruction. This project asks whether they can be reconstructed anyway, catalogues what breaks when they can't, and fixes what's fixable with low-effort preprocessing rather than new captures.

Provided here is a standardized reconstruction pipeline ([`src/reconstruction/`](src/reconstruction/)) (video/imagees → frames → COLMAP SfM → 3DGS).

## 🛠️ Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Sync the reconstruction pipeline's dependencies:
   ```sh
   uv sync --extra recon        # pycolmap, gsplat, opencv, lpips/ssim metrics
   uv sync --extra recon-ns     # + Nerfstudio, for the splatfacto training path
   uv run pre-commit install
   ```
   Both extras can be installed together (`uv sync --extra recon --extra recon-ns`). `recon-ns` is only needed if you're training with `ns-train splatfacto` rather than the custom `train_gs.py` loop.
3. On a SLURM adjusted cluster, `module load CUDA/13.0.0` (or whatever CUDA module your cluster exposes) before any GPU step — `gsplat`/`nerfstudio` silently fall back to CPU-disabled builds otherwise.

## 🚀 Quickstart: Reconstruct a Scene

The fastest way to see the pipeline work end-to-end is the smoke test, one small UAV clip, short training budget, no dataset download required beyond the clip itself:

```sh
sbatch slurm/smoke_test_reconstruction.sh
```

That runs, in order, the same three stages any scene goes through:

```sh
# 1. Sample frames from source video at a fixed fps/resolution (configs/reconstruction.yaml)
uv run --extra recon python src/reconstruction/extract_frames.py <video.mp4> <frames_dir>

# 2. COLMAP structure-from-motion (feature extraction, exhaustive matching, incremental mapping)
uv run --extra recon python src/reconstruction/sfm.py <frames_dir> <colmap_dir>

# 3a. Train with the custom gsplat loop (holds out every Nth view, reports PSNR/SSIM)
uv run --extra recon python src/reconstruction/train_gs.py \
    "$(cat <colmap_dir>/best_sparse_dir.txt)" <frames_dir> <output_dir> --iters 30000

# 3b. ...or convert to a Nerfstudio dataset and train splatfacto instead
uv run --extra recon python src/reconstruction/to_nerfstudio.py <sparse_dir> <frames_dir> <ns_data_dir>
uv run --extra recon-ns ns-train splatfacto --data <ns_data_dir> \
    --max-num-iterations 30000 nerfstudio-data --eval-mode interval --eval-interval 8
```

Sampling rate, resize, frame cap, COLMAP camera model, and training/holdout settings are all pulled from [`configs/reconstruction.yaml`](configs/reconstruction.yaml) by default, so a scene only deviates from the standard protocol when a dataset-specific fix requires it (documented per-dataset when that happens).

To go beyond the smoke test, pull one of the datasets the pipeline has been run against:

- [MOTS-Annotated UAV Vineyard](https://zenodo.org/records/10625595)
- [Bodegas Terras Gauda UAV RGB](https://zenodo.org/records/7330951)
- [Multispectral Botrytis](https://zenodo.org/records/7383601)
- [BLT (Bacchus Long-Term)](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/blt/)

`slurm/download_datasets.sh` bulk-downloads them.

## 🔧 Known failure modes and fixes

Not every dataset reconstructs cleanly out of the box. Three failure modes have been identified so far:

- **Lens flare, veiling glare and lens ghost** (MOTS UAV footage, backlit sun) — `src/reconstruction/deflare_dark_channel.py` (dark-channel-prior deglare) + `src/reconstruction/generate_ghost_masks.py` (ghost region mask).
- **Multispectral band misalignment** (Botrytis, 5 physically separate lenses, no embedded calibration) — `src/reconstruction/extract_botrytis_rgb.py` (ORB-matched per-band alignment + joint color stretch).
- **Robot in frame and low frame overlap** (BLT ground-robot sequences) — COLMAP registers only ~half the input frames; ruled out robot-body occlusion as the cause by re-running SfM on robot-body-masked frames.

## 📁 Repo map

```
configs/reconstruction.yaml   Standardized frame-sampling / COLMAP / training protocol
src/reconstruction/   The actual pipeline: extract → SfM → train/eval → per-dataset fixes
slurm/                One SLURM script per pipeline stage / dataset / fix arm
docs/                 The catalog website (GitHub Pages)
```

`src/` also carries a generic Hydra + Weights & Biases experiment harness (`main.py`, `model/`, `dataset/`, `metric/`, `logger/`, `configs/{model,dataset,metric,logger,hydra}/`) inherited from the project template, not used by the reconstruction pipeline above, which runs as standalone scripts.

## 📚 Citation

See [How to cite](README.md#how-to-cite) in the README.
