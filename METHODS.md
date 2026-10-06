# Methods, as run (reproducibility appendix source)

This file records the protocol **as it ran** for the revision experiments (October 2026),
not as planned. Code paths are relative to the repository root. The `results/` paths and the
per-section commit hashes (`results/RUNS.md`) refer to
[vineyard-scene-reconstruction](https://github.com/DRAGONS-Project/vineyard-scene-reconstruction), branch `full-program`, which holds the full
history; this repository is a snapshot of its code.

## Sampling and frame extraction

- Sampling rule for every sequence: frames `0, k, 2k, …` with
  `k = max(floor(N/200), round(fps/2), 1)`, at most 200 frames (`scripts/characterisation/common.py`;
  `MAX_FRAMES=200`, `TARGET_FPS=2.0`). Videos are decoded with OpenCV; ROS bags are read over
  HTTP range from the LCAS share (`common.RemoteBag`); Botrytis captures come from the Zenodo
  deposit zips.
- Frames are resized to 1600 px width (`TARGET_WIDTH`), never upscaled, and written as PNG.
- Frame lists are fixed once written (`results/characterisation/frame_lists/<id>.txt`); every
  later experiment on a sequence reuses the committed list. A rerun that would select
  different frames writes a `.rerun.txt` next to it instead of replacing it.
- BLT sequences use one corridor pass chosen from odometry (longest run with speed > 0.4 m/s
  and heading change < 15°); front-camera sequences reuse the committed side-pass window
  verbatim (`prepare.py: prep_blt, prep_blt_front`). Black frames (mean luminance < 8) are
  excluded before sampling.
- The pass was selected on bag record time. In the bags of 13 Jul and 15 Sep 2022 odometry and
  RTK messages are recorded 7 to 12 s (up to 29 s) after the images with the same header stamp,
  so those cuts start late and end with the robot standing (17 and 21 frames). The corrected
  cuts (`blt_gr_20220713_hdr`, `blt_gr_20220915_hdr` and their `_front` passes) select on header
  stamps (`clock="header"`) and keep the original row (`same_pass_as`): the candidate rows are
  near-tied in length, and on the corrected clock the rule alone would pick a different row.
  The other four BLT bags have no recording lag, so both clocks give the same cut. The
  original-cut sequences are kept for the standing-tail ablation (`results/ablations/blt_tail/`);
  `results/ablations/blt_recut/replacement_rows.csv` maps each original-cut row to its
  corrected-cut row.
- Terras Gauda sequences are part-level: a part is one source `.mp4` (the clips concatenate
  separate recordings); the quality rows use the largest part of each clip
  (`scripts/characterisation/sequences.py`).
- Botrytis composed-RGB sequences use exactly the captures of the committed green-band lists;
  one 30_V1 capture lacking band files is skipped with its index-bound name kept.

## Structure from motion (fixed configuration; `scripts/characterisation/register.py`)

- pycolmap 4.0.4 (bundled COLMAP 4.0.4), CPU only, `pycolmap.set_random_seed(0)`.
- Features: SIFT, `max_image_size=1600`, `max_num_features=8192`, camera model OPENCV,
  camera mode SINGLE for same-size sequences (PER_FOLDER dimension groups for mixed-size
  stills). Matching: exhaustive. Mapper: `IncrementalPipelineOptions` defaults.
- Optional masks for feature extraction follow COLMAP's convention
  (`<mask_dir>/<image name>.png`, zero = no features), via `--sfm-masks`.
- Gate for training: the largest reconstructed model must hold ≥ 50 % of the sequence's
  frames; every run keeps its registration row regardless (`outcome`, `model_sizes`).
  For the quality rows the gate was decided on the characterisation run's registration;
  the quality job registers again and trains on what that run gives (`GATE_MIN` was not
  set). One quality row, `blt_gr_20220915`, registered 57 of 109 in the characterisation
  run and 47 of 109 (below the gate) in the quality run, and was trained on the 47.
  Ablation and sweep jobs apply the gate to their own registration.
- Undistortion: after the gate, `pycolmap.undistort_images` (default `UndistortCameraOptions`)
  converts the best model and its frames to PINHOLE (`scripts/characterisation/undistort.py`);
  the trainer refuses distorted camera models otherwise (`--allow-distorted` exists only to
  reproduce the superseded pre-revision runs). Loss masks are undistorted with the same model.

## Trainer (`src/reconstruction/train_gs.py`; gsplat 1.4.0)

- 30,000 iterations for every quality and ablation row; one view per iteration, train views
  shuffled per epoch; seed 0 unless stated (`--seed` sets `random` and `torch`).
- Initialization from the SfM points: per-point isotropic size = RMS distance to the 3
  nearest neighbours (`knn_scales`, clamp ≥ 1e-7; `--init-scale fixed` reproduces the old
  0.01 constant); opacity 0.5; colours RGB (no spherical harmonics); identity rotations.
- Densification: gsplat `DefaultStrategy` defaults (grow_grad2d 0.0002, grow_scale3d 0.01,
  grow_scale2d 0.05, prune_opa 0.005, prune_scale3d 0.1, refine every 100 from iteration 500,
  opacity reset every 3000), `refine_stop_iter = min(iters/2, 15000)`; no cap on the number
  of Gaussians.
- Optimizers (Adam, eps 1e-15): means lr 1.6e-4 × scene scale with exponential decay to 1 %
  over the run; scales 5e-3; quats 1e-3; opacities 5e-2; colours 2.5e-3. Scene scale = 95th
  percentile distance of the SfM points from their centroid.
- Loss: 0.8 · L1 + 0.2 · (1 − SSIM); rasterization `packed=True`. Images train at the
  undistorted resolution (1600 px width inputs → ≈1602 px after undistortion).
- Optional loss masks (`--mask-dir`, white = used): excluded pixels take the target's value
  before the loss, carrying no gradient.

## Evaluation

- Held-out splits, fixed by image NAME per sequence from the committed frame list and reused
  across all conditions (`--holdout-names`): **interp** = every 8th name from index 0;
  **block** = blocks of four consecutive names from index 16, every 32, a block touching the
  sequence end dropped; sequences under 36 frames hold out their middle four. Listed names
  that did not register are skipped and counted.
- Metrics: PSNR and SSIM from scikit-image (`data_range=1`), LPIPS (lpips 0.1.4) with the
  AlexNet backbone. Per-view values are persisted (`metrics_full.json`).
- Overlap columns per row: for each held-out view, distance from its camera centre to the
  nearest training camera as a fraction of the camera-path length (`nn_dist_frac`) and the
  angle between their optical axes (`nn_angle_deg`).
- Masked scoring (`compute_metrics.py --mask-dir`): PSNR from the masked MSE; SSIM and LPIPS
  as their per-pixel maps averaged over the mask (SSIM over the mask eroded by the 3-pixel
  window border, so a full mask reproduces the unmasked value exactly). Used for the
  common-pixel ablation column and the region metric.
- Own-reference caveat: conditions that change the images (dehazing, band composition) are
  scored against their own processed frames; registration metrics lead every ablation table.
- Region metric (MOTS only; `scripts/characterisation/region_metric.py`): the deposits'
  instance maps (KITTI-MOTS ids; classes grape/trunk/pole; **grape only**) are resized
  (nearest) and undistorted with the row's stored camera, and each kept held-out render is
  scored inside and outside the mask. Scores come from the kept q90 JPEG renders — a small
  recompression bias that applies to both regions alike. The three NoPathPlanning zips index
  frames continuously (offsets 0 / 1050 / 1950; the last under `default-2/`), verified
  visually.

## Preprocessing methods (ablation conditions, applied before SfM)

- Dehazing (`src/reconstruction/deflare_dark_channel.py`): dark-channel prior, patch radius 7,
  omega 0.90, t0 0.15; veil colour = mean of the top 0.1 % dark-channel candidates
  (`--atmo brightest` = the single brightest candidate, the ablation variant); transmission
  refined by a guided filter (radius 40, eps 1e-3; `--no-guided` skips it).
- Ghost masks (`src/reconstruction/generate_ghost_masks.py`): local excess-green above a
  45-px-radius baseline AND low 5-px-radius V-channel texture, V floor 0.45, blobs ≥ 0.2 %
  of the frame, dilated 9 px; output white = keep. Used in SfM, in the loss and (as the
  union over conditions) for common-pixel scoring.
- Botrytis band composition (`scripts/characterisation/botrytis_compose.py`): modes are
  single named deviations from the full method — alignment none / median-of-25 ORB+RANSAC
  homographies (≥ 80 inliers each) / per-frame (identity fallback on failure, share
  recorded); stretch per-band or joint 2–98 percentile; gray-world on or off; border crop
  0 or 30 px; bands B/G/R or Red/RedEdge/NIR (false colour, reference band = the middle one).
  The full method = median alignment + joint stretch + gray-world + 30 px crop.

## Scale anchors (`scripts/characterisation/anchors_eval.py`)

- Similarity (Sim(3)) alignment by Umeyama between registered camera centres and the anchor
  positions; the residual is the camera-position RMSE after alignment (validates scale and
  trajectory shape, not georeferencing).
- BLT: `/gps/fix` (RTK, all fixes status 2) interpolated to each frame's header stamp in a
  local ENU frame. Botrytis: per-capture EXIF GPS streamed from the deposit zips (the GPS
  IFD sits at the end of each TIFF, so whole members are read; cached per flight). RTK video
  (Tomiño): the shipped DJI flight log's `isVideo` rows at 10 Hz, frame time = global index /
  30 fps + a time offset searched over −4…+2 s (interior optimum reported). The log's own
  columns report P-GPS at gpslevel 5; "RTK" is the deposit's claim. This one row
  (`kind = uav_rtk_log` in `results/anchors/geometric.csv`) was computed once in an
  interactive session; its alignment code is not in the repository.

## Requirement sweeps

One property varied, everything else at the standard protocol; one seed; interp split on the
sweep's own frame list; the shared level (2 fps / 1600 px / whole span) is the sequence's
quality row. Width levels above 1600 px change training and scoring only (SIFT stays capped);
every width level is additionally scored with renders and references resized to 640 px
(`psnr_at640`, the cross-level comparable column; the shared level's value comes from its
kept q90 JPEGs). Both sweep sequences are UAV row passes over the same estate:
MOTS NoPathPlanning_1 (frontal, backlit, at harvest) and Terras Gauda Row7.2_p3 (3 m AGL,
60° tilt, dense pre-harvest canopy).

## Registration repeat study (`scripts/characterisation/repeat_registration.py`)

The fixed COLMAP configuration is run N times on identical frames (and masks) per condition;
run r sets `pycolmap.set_random_seed(r)` (`register.py --seed r`) and `PYTHONHASHSEED=r`,
4 threads, nothing else changes (seed 0 is the instrument's own). Ten conditions, 46 runs:
two controls at N = 3 (MOTS NoPathPlanning_1 standard level, Terras Gauda Row7.2_p3) and
eight at N = 5 (NoPathPlanning_1 at 0.5 fps; UK front raw / fixed-region mask; 13 Jul front
raw / Greek mask; 15 Sep front raw / Greek mask; 23 Mar side raw). Success = largest model
≥ 50 % of frames (the training gate). Each run records model sizes and, parsed from the
mapper log, every initial pair tried with its frame gap, verified inlier count (from
`two_view_geometries`) and the number of images registered after it (a count of mapper
log lines, which can exceed the frame count). The "pair behind the largest model" is the
attempt followed by most such lines, a proxy read from the log, not from the model; in
`summary.csv` a distinct pair is its two frame names in either order. Outputs:
`results/repeats/runs/<condition>/run_<r>.json`, `repeats.csv`, `summary.csv`
(`collect_repeats.py`); `registration_stability[_masked]` in `results/joined.csv` (the front
passes and the sweep level appear there as stub rows carrying only these columns);
`registration_success_rate` / `pending_repeats` in `results/ablations/blt.csv`
(`pending_repeats = true` marks SfM configurations outside the study). The COLMAP
configuration was deliberately not altered to stabilise it. SLURM job 6024964 (array 0-9).

## Hardware, software, cost, noise floor

- Nodes: PLGrid WCSS "Lem". GPU jobs: 1× NVIDIA H100, 2–4 CPU cores, 32–48 GB RAM.
  CPU-only stages: 1–8 cores.
- Software: Python 3.11, PyTorch 2.12.0+cu130, gsplat 1.4.0, pycolmap 4.0.4, lpips 0.1.4,
  OpenCV 4.13, scikit-image; CUDA 13.0.
- Measured times (60-frame sequence at 1600 px): COLMAP ≈ 96 s at 8 cores / 184 s at 4 /
  493 s at 2 (matching core-seconds are constant; mapping partly serial); undistortion ≈ 10 s;
  training 30,000 iterations ≈ 660–720 s (≈ 7–14 M Gaussians, ≤ 7 GB GPU memory);
  evaluation ≈ 8 s + LPIPS.
- Noise floor (debug study, MOTS NoPathPlanning_1): same seed repeated differs by 0.02 dB
  mean PSNR; another seed by 0.1–0.6 dB; single held-out views by up to 1.5 dB (floaters
  differ between runs). Differences below ≈ 0.5 dB between single runs are not evidence;
  ablation conditions therefore run 3 seeds.
- Undistortion matters: training on distorted OPENCV frames cost ≈ 2.5 dB in the debug study
  (18.46 → 21.12); every pre-revision number from this trainer is void for that reason (the
  old SfM also estimated one camera per frame).
