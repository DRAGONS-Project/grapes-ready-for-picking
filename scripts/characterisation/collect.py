"""Assemble the per-sequence JSON files into the CSVs the paper reads.

usage: collect.py <results_dir>
"""

import csv
import json
import sys
from pathlib import Path

from sequences import SEQUENCES

R = Path(sys.argv[1])
PS = R / "per_sequence"


def load(name):
    p = PS / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=4):
    if v is None:
        return "undetermined"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def write(name, cols, rows):
    with open(R / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


# documentation-reliability findings that apply to every sequence of a dataset
DOC_NOTES = {
    "btg": ["documentation: the parts of a clip (Row4.1_1, Row4.1_2, ...) are separate recordings - the view jumps at every join and each "
            "part registers as its own COLMAP model; the data paper describes them as contiguous"],
    "bot": ["documentation: the paper states a 30 m flight height; EXIF GPS altitude is about 99-101 m in flight against 79-84 m for the "
            "0_V2 panel captures taken on the ground, i.e. roughly 16-20 m above ground"],
}
seq_rows, fq_rows, reg_rows = [], [], []
for s in SEQUENCES:
    m, q, g = load(f"{s['id']}.meta.json"), load(f"{s['id']}.quality.json"), load(f"{s['id']}.registration.json")
    if m is None:
        seq_rows.append(dict(dataset=s["dataset"], sequence_id=s["sequence_id"], population=s["population"], notes="not run"))
        continue
    seq_rows.append(dict(dataset=s["dataset"], sequence_id=s["sequence_id"], population=s["population"], source=m["source"],
                         n_frames_total=m["n_frames_total"], frame_list=f"frame_lists/{s['id']}.txt", width_used=m["width_used"],
                         k=m["k"], frame_spacing_s=fmt(m["frame_spacing_s"]), fps_source=fmt(m["fps_source"]), n_images=m["n_images"],
                         camera_mode=m["camera_mode"], notes=" | ".join(m["notes"] + DOC_NOTES.get(s["dataset"], []))))
    if q:
        fq_rows.append(dict(dataset=s["dataset"], sequence_id=s["sequence_id"], population=s["population"], n_frames=q["n_frames"],
                            **{k: fmt(q[k]) for k in ("blur_median", "blur_share_below", "blur_p10", "blur_p50", "blur_p90", "clip_share_median",
                                                      "clip_frames_gt5", "lum_mean_sigma", "static_share", "black_frames_share")},
                            black_share_basis=q.get("black_share_basis", ""),
                            static_share_top_third=fmt(q.get("static_share_top_third")) if "static_share_top_third" in q else "n/a",
                            static_share_bottom_third=fmt(q.get("static_share_bottom_third")) if "static_share_bottom_third" in q else "n/a"))
    if g:
        notes = []
        if g.get("focal_doc_source"):
            notes.append("focal_px_doc: " + g["focal_doc_source"])
        if g.get("focal_px_est_min") is not None and g.get("focal_px_est_min") != g.get("focal_px_est_max"):
            notes.append(f"focal_px_est range {g['focal_px_est_min']:.1f}-{g['focal_px_est_max']:.1f} over {len(g.get('cameras', {}))} camera groups")
        if g.get("focal_px_doc_by_group"):
            notes.append("focal_px_doc by group: " + json.dumps(g["focal_px_doc_by_group"]))
        if s["population"] == "stills":
            notes.append("consecutive cameras = file-name order")
        geo = ("tri_angle_median", "baseline_frac", "view_coverage_sr", "view_spread_p90", "focal_px_est", "focal_px_doc", "focal_ratio", "reproj_err", "track_len")
        reg_rows.append(dict(dataset=s["dataset"], sequence_id=s["sequence_id"], population=s["population"], n_images=g["n_images"],
                             n_registered=g.get("n_registered", "undetermined"), reg_rate=fmt(g.get("reg_rate")), n_models=g.get("n_models", "undetermined"),
                             model_sizes=fmt(g.get("model_sizes")), n_points=g.get("n_points", "undetermined"),
                             **{k: fmt(g.get(k)) for k in geo}, outcome=g["outcome"], wall_s=fmt(g.get("wall_s")), t_extract_s=fmt(g.get("t_extract")),
                             t_match_s=fmt(g.get("t_match")), t_map_s=fmt(g.get("t_map")), cores=g.get("threads"), mapper_threads=g.get("mapper_threads"),
                             camera_mode=g.get("camera_mode"), notes=" | ".join(notes)))
vld = load("vld.json") or []
for v in vld:
    seq_rows.append(dict(dataset="vld", sequence_id=v["file"], population="point_cloud", source=f"zenodo 8113105 {v['file']} (header read remotely)",
                         n_frames_total=0, width_used="n/a", n_images=0,
                         notes=f"point cloud, n_points={v['n_points']}, bbox_min={[round(x, 2) for x in v['bbox_min']]}, bbox_max={[round(x, 2) for x in v['bbox_max']]} "
                               f"({v['bbox_source']}), crs={(v['crs'] or {}).get('epsg') or 'none declared'}, LAS {v['las_version']}"))
write("sequences.csv", ["dataset", "sequence_id", "population", "source", "n_frames_total", "frame_list", "width_used", "k", "frame_spacing_s",
                        "fps_source", "n_images", "camera_mode", "notes"], seq_rows)
write("frame_quality.csv", ["dataset", "sequence_id", "population", "n_frames", "blur_median", "blur_share_below", "blur_p10", "blur_p50", "blur_p90",
                            "clip_share_median", "clip_frames_gt5", "lum_mean_sigma", "static_share", "static_share_top_third",
                            "static_share_bottom_third", "black_frames_share", "black_share_basis"], fq_rows)
write("registration.csv", ["dataset", "sequence_id", "population", "n_images", "n_registered", "reg_rate", "n_models", "model_sizes", "reproj_err",
                           "track_len", "n_points", "tri_angle_median", "baseline_frac", "view_coverage_sr", "view_spread_p90", "focal_px_est",
                           "focal_px_doc", "focal_ratio", "outcome", "wall_s", "t_extract_s", "t_match_s", "t_map_s", "cores", "mapper_threads",
                           "camera_mode", "notes"], reg_rows)

# artefacts.csv: keep existing annotations, add missing rows
acols = ["dataset", "sequence_id", "glare", "lens_ghost", "vignetting", "chromatic_fringing", "rolling_shutter", "dirt_or_rain", "other", "confidence", "notes"]
existing = {}
if (R / "artefacts.csv").exists():
    existing = {(r["dataset"], r["sequence_id"]): r for r in csv.DictReader(open(R / "artefacts.csv"))}
arows = [existing.get((s["dataset"], s["sequence_id"]), dict(dataset=s["dataset"], sequence_id=s["sequence_id"]))
         for s in SEQUENCES if (R / "contact_sheets" / f"{s['id']}.jpg").exists()]
write("artefacts.csv", acols, arows)

# anchors.csv: one row per dataset
AP = R / "anchors_parts"


def part(name):
    p = AP / name
    return json.loads(p.read_text()) if p.exists() else None


an = []
blt = [(s["sequence_id"], part(f"{s['id']}.json")) for s in SEQUENCES if s["dataset"] == "blt"]
blt = [(k, v) for k, v in blt if v]
if blt:
    ok = sum(1 for _, v in blt if (v["share_gap_lt_0p2_header"] or 0) > 0.95)
    an.append(dict(dataset="blt", anchor_type="RTK-GNSS fix topic /gps/fix", anchor_present="yes", anchor_resolvable="yes",
                   check=f"share of sampled frames with a fix within 0.2 s (header stamps) exceeds 95 % in {ok} of {len(blt)} passes",
                   notes="; ".join(f"{k}: {v['share_gap_lt_0p2_header']:.3f} by header stamp, {v['share_gap_lt_0p2_bagtime']:.3f} by bag record time, "
                                   f"{v['n_fix_in_window']} fixes, status {v['fix_status_counts']}" for k, v in blt)))
bot = [(s["sequence_id"], part(f"{s['id']}.json")) for s in SEQUENCES if s["dataset"] == "bot"]
bot = [(k, v) for k, v in bot if v]
if bot:
    an.append(dict(dataset="bot", anchor_type="EXIF GPS per capture; 7 GCPs in GCPs.zip (ETRS89 / UTM 29N)", anchor_present="yes", anchor_resolvable="yes",
                   check="all sampled TIFFs carry EXIF GPS: " + ("yes" if all(v["sampled_with_gps_share"] == 1.0 for _, v in bot) else "no")
                         + "; GCPs inside the convex hull of image positions: " + ", ".join(f"{k} {v['gcp_inside_hull']}/{v['n_gcp']}" for k, v in bot),
                   notes="hull built from the GPS positions of all kept captures of the flight"))
if vld:
    declared = [v for v in vld if v["crs"] and v["crs"].get("epsg")]
    local = [v["file"] for v in vld if not (v["crs"] and v["crs"].get("epsg"))]
    an.append(dict(dataset="vld", anchor_type="CRS declared in the LAS header", anchor_present=f"{len(declared)} of {len(vld)} files",
                   anchor_resolvable="yes for the 8 files of 2022; no for the 2 files of 2021-09-16",
                   check=f"EPSG {sorted({v['crs']['epsg'] for v in declared})} declared in {len(declared)} files; {len(local)} files declare no CRS and hold local "
                         "coordinates, so overlap with the Botrytis GCPs cannot be tested for them; the data paper states EPSG:32629 georeferencing "
                         "for the dataset",
                   notes="GCPs inside the XY bounding box: " + ", ".join(f"{v['file'][:8]}_{v['file'].split('_')[-2]}_{v['file'].split('_')[-1][:2]} "
                                                                         f"{v.get('gcp_inside_xy_bbox', 'n/a')}/{v.get('n_gcp', 'n/a')}" for v in vld)
                         + "; no CRS: " + ", ".join(local)))
for ds in ("mots", "btg"):
    an.append(dict(dataset=ds, anchor_type="RTK UAV (per data paper)", anchor_present="no", anchor_resolvable="no", check="no trajectory in deposit",
                   notes="videos carry no GPS or pose track"))
esc = part("esc_photos.json")
if esc:
    an.append(dict(dataset="esc", anchor_type="EXIF GPS per photo; trunk shapefile (vineyard B7)", anchor_present="yes", anchor_resolvable="partly",
                   check=f"{esc['n_with_gps']} of {esc['n_photos']} photos carry GPS; {esc['share_within_50m']:.3f} lie within 50 m of a trunk",
                   notes=f"{esc['n_trunks']} trunks; distance to nearest trunk: median {esc['dist_m_p50']:.1f} m, 95th percentile {esc['dist_m_p95']:.0f} m, "
                         f"max {esc['dist_m_max']:.0f} m"))
for ds in ("emb", "gcs", "gst"):
    an.append(dict(dataset=ds, anchor_type="none", anchor_present="no", anchor_resolvable="no", check="n/a", notes=""))
write("anchors.csv", ["dataset", "anchor_type", "anchor_present", "anchor_resolvable", "check", "notes"], an)
