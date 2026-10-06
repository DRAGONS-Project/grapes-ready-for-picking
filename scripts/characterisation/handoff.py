"""Hand-off for the paper.

    handoff.py collect            supplementary tables, collected from kept artefacts only:
                                  results/characterisation/recording.csv   (per_sequence/*.meta.json)
                                  results/ablations/masks/quality.csv      (mask reports, provenance)
    handoff.py tables <out_dir>   the value tables B1..B10 of the Claude Doc "Characterization
                                  values for the paper", as Markdown and as tables.json
    handoff.py pack <dir> <tables_dir> <data_dictionary.md> <disk_checks_dir> [doc_url]
                                  the hand-off directory, its README.md manifest, the zip, md5

Every value in the tables is read from a CSV under results/. Rounding is half-up on the CSV
string: shares as percentages with one decimal, angles one decimal, PSNR and other medians
two decimals, SSIM / LPIPS / ATE / scale three decimals, counts as integers.
"""

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
MISSING = {"", "undetermined", "n/a", "None", "nan"}
DASH = "—"
NA = "n/a"


# ---------------------------------------------------------------- reading and formatting

def read(rel):
    with open(RES / rel) as f:
        return list(csv.DictReader(f))


def miss(v):
    return v is None or str(v).strip() in MISSING


def dec(v):
    return Decimal(str(v).strip())


def rnd(v, nd):
    return str(Decimal(v).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def num(v, nd):
    return DASH if miss(v) else rnd(dec(v), nd)


def pct(v):
    return DASH if miss(v) else rnd(dec(v) * 100, 1) + " %"


def integer(v):
    return DASH if miss(v) else f"{int(dec(v)):,}"


def plain(v):
    """A stored value in plain notation (no added precision)."""
    if miss(v):
        return DASH
    d = dec(v)
    return f"{d:f}"


def sizes(v):
    return DASH if miss(v) else ", ".join(str(x) for x in json.loads(v))


def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mean(vals):
    return sum(vals) / len(vals)


def agg(rows, col, kind):
    """'median [min–max]' over the rows whose value is determined; '(n = k)' when fewer."""
    vals = [dec(r[col]) for r in rows if not miss(r.get(col))]
    if not vals:
        return DASH
    if kind == "pct":
        f = lambda d: rnd(d * 100, 1)
        unit = " %"
    elif kind == "deg":
        f = lambda d: rnd(d, 1)
        unit = ""
    elif kind == "count":
        f = lambda d: rnd(d, 0) if d == d.to_integral_value() else rnd(d, 1)
        unit = ""
    else:
        f = lambda d: rnd(d, 2)
        unit = ""
    out = f(median(vals)) + unit
    if len(vals) > 1 and f(min(vals)) != f(max(vals)):
        out += f" [{f(min(vals))}–{f(max(vals))}]"
    if len(vals) < len(rows):
        out += f" (n = {len(vals)})"
    return out


def seed_stat(rows, col, nd):
    """Seed mean with seed range."""
    vals = [dec(r[col]) for r in rows if not miss(r.get(col))]
    if not vals:
        return DASH
    out = rnd(mean(vals), nd)
    if len(vals) > 1:
        out += f" [{rnd(min(vals), nd)}–{rnd(max(vals), nd)}]"
    return out


def table(headers, rows):
    for r in rows:
        assert len(r) == len(headers), (headers, r)
        assert all("|" not in str(c) and "\n" not in str(c) for c in r), r
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- supplementary tables

def fmt4(v):
    if v is None:
        return "undetermined"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def collect():
    ps = RES / "characterisation/per_sequence"
    rows = []
    for f in sorted(ps.glob("*.meta.json")):
        m = json.loads(f.read_text())
        ns = m.get("native_size")
        if isinstance(ns, dict):
            ns = "; ".join(f"{k} ({v})" for k, v in ns.items())
        alt = ("", "", "")
        for n in m.get("notes", []):
            g = re.search(r"GPS altitude median ([\d.]+) m, min ([\d.]+) m, max ([\d.]+) m", n)
            if g:
                alt = g.groups()
        pw = m.get("pass_window_s")
        rows.append(dict(
            id=m["id"], dataset=m["dataset"], sequence_id=m["sequence_id"],
            population=m["population"], source=m["source"], n_frames_total=m["n_frames_total"],
            fps_source=fmt4(m.get("fps_source")), k=m["k"],
            frame_spacing_s=fmt4(m.get("frame_spacing_s")), native_size=ns,
            width_used=m["width_used"], n_images=m["n_images"], camera_mode=m["camera_mode"],
            exif_gps_alt_median_m=alt[0], exif_gps_alt_min_m=alt[1], exif_gps_alt_max_m=alt[2],
            pass_window_s=f"{pw[0]}-{pw[1]}" if pw else "",
            meta_file=str(f.relative_to(REPO))))
    cols = list(rows[0].keys())
    with open(RES / "characterisation/recording.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # the 34 characterised sequences must agree with sequences.csv
    seq = {f"{r['dataset']}_{r['sequence_id']}": r for r in read("characterisation/sequences.csv")}
    for r in rows:
        s = seq.get(r["id"])
        if s:
            for c in ("n_frames_total", "n_images", "k", "width_used", "frame_spacing_s",
                      "fps_source", "source"):
                assert str(r[c]) == s[c], (r["id"], c, r[c], s[c])
    print(f"recording.csv: {len(rows)} rows")

    mrows = []
    for f in sorted((RES / "full_program/C/blt").glob("*/*/mask_report.json")):
        d = json.loads(f.read_text())
        mrows.append(dict(kind="condition", sequence=f.parts[-3], condition=f.parts[-2],
                          mask_kind=d["mask_kind"], n_masks=d["n_masks"],
                          masked_share_mean=d["masked_share_mean"],
                          masked_share_max=d["masked_share_max"], sfm_mask=d["sfm_mask"],
                          loss_mask=d["loss_mask"], source_file=str(f.relative_to(REPO))))
    for name, kind in (("uk_front", "fixed_region"), ("greek_front", "greek")):
        f = RES / f"masks/hand/{name}.provenance.json"
        p = json.loads(f.read_text())
        mrows.append(dict(kind="hand_mask", mask_kind=kind, mask_file=f"results/masks/hand/{name}.png",
                          masked_share_mean=p.get("masked_share", ""),
                          width=p["size"][0], height=p["size"][1],
                          provenance=p.get("source") or p.get("status"),
                          source_file=str(f.relative_to(REPO))))
    f = RES / "masks/sam3/blt_uk_20230726_front.report.json"
    rep = json.loads(f.read_text())
    for st in rep["steps"]:
        if st["step"] == "prompt_empty":
            mrows.append(dict(kind="sam3_attempt", sequence="blt_uk_20230726_front",
                              mask_kind="sam3", prompt=f"{st['kind']}: {st['prompt']}",
                              outcome="no object found", source_file=str(f.relative_to(REPO))))
        if st["step"] == "prompt_result":
            mrows.append(dict(kind="sam3_attempt", sequence="blt_uk_20230726_front",
                              mask_kind="sam3", prompt=f"{st['kind']}: per-frame box prompts",
                              n_masks=st["n"], iou_consec_median=st["iou_consec_median"],
                              iou_consec_min=st["iou_consec_min"],
                              area_share_median=st["area_share_median"], outcome=rep["outcome"],
                              source_file=str(f.relative_to(REPO))))
    cols = ["kind", "sequence", "condition", "mask_kind", "mask_file", "width", "height",
            "n_masks", "masked_share_mean", "masked_share_max", "sfm_mask", "loss_mask", "prompt",
            "iou_consec_median", "iou_consec_min", "area_share_median", "outcome", "provenance",
            "source_file"]
    out = RES / "ablations/masks"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "quality.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(mrows)
    print(f"ablations/masks/quality.csv: {len(mrows)} rows")


# ---------------------------------------------------------------- the Doc tables

DS_NAME = {"mots": "GrapeMOTS", "blt": "BLT", "btg": "Terras Gauda", "bot": "Botrytis",
           "emb": "Embrapa WGISD", "esc": "EscaYard", "gcs": "GrapeCS-ML", "gst": "GrapeSet",
           "slam": "GrapeSLAM", "vld": "VineLiDAR"}
FLAGS = ["glare", "lens_ghost", "vignetting", "chromatic_fringing", "rolling_shutter",
         "dirt_or_rain", "other"]


def rid(r):
    return r.get("id") or f"{r['dataset']}_{r['sequence_id']}"


def load():
    T = dict(
        seq=read("characterisation/sequences.csv"), reg=read("characterisation/registration.csv"),
        fq=read("characterisation/frame_quality.csv"), art=read("characterisation/artefacts.csv"),
        anc=read("characterisation/anchors.csv"), rec=read("characterisation/recording.csv"),
        doc=read("characterisation/documentation_checks.csv"), q=read("quality/quality.csv"),
        mask=read("ablations/masks/quality.csv"), sw=read("sweeps/sweeps.csv"),
        geo=read("anchors/geometric.csv"), rep=read("repeats/summary.csv"),
        regions=read("regions/mots_regions.csv"))
    T["abl"] = {s: read(f"ablations/{s}.csv") for s in ("botrytis", "mots", "blt")}
    blt_raw, seen = [], set()
    for r in T["abl"]["blt"]:
        if r["condition"] == "raw" and r["sequence"] not in seen:
            seen.add(r["sequence"])
            blt_raw.append(dict(r, id=r["sequence"]))
    T["blt_raw"] = blt_raw
    T["rep_raw"] = {r["sequence"]: r for r in T["rep"] if r["mask_kind"] == "none"}
    return T


def populations(T):
    reg, q = T["reg"], T["q"]
    by = lambda rows, ds, pop=None: [r for r in rows if r["dataset"] == ds
                                     and (pop is None or r.get("population") == pop)]
    C, QC, BC = "characterisation/registration.csv", "quality/quality.csv", "ablations/blt.csv"
    return [
        dict(code="mots", pop="UAV row passes (NoPathPlanning_1 to 3)", rows=by(reg, "mots", "row_pass"), src=C),
        dict(code="mots", pop="UAV orbits (PathPlanning_1, 2, 8)", rows=by(reg, "mots", "orbit"), src=C),
        dict(code="blt", pop="Greek side-camera passes", rows=by(reg, "blt", "gr_session"), src=C),
        dict(code="blt", pop="UK side-camera pass", rows=by(reg, "blt", "uk_session"), src=C),
        dict(code="blt", pop="front-camera passes, unmasked (5 Greek, 1 UK)", rows=T["blt_raw"], src=BC,
             content="not measured"),
        dict(code="btg", pop="parts run (largest part of 11 clips, and the one-part clip Row6.3)",
             rows=by(q, "btg"), src=QC, content="see whole clips"),
        dict(code="btg", pop="whole clips, parts concatenated", rows=by(reg, "btg"), src=C),
        dict(code="bot", pop="multispectral flights, green band", rows=by(reg, "bot"), src=C),
        dict(code="emb", pop="video frames", rows=by(reg, "emb", "video_frames"), src=C),
        dict(code="emb", pop="stills", rows=by(reg, "emb", "stills"), src=C, stills=True),
        dict(code="esc", pop="smartphone stills", rows=by(reg, "esc"), src=C, stills=True),
        dict(code="gcs", pop="stills (Set 5)", rows=by(reg, "gcs"), src=C, stills=True),
        dict(code="gst", pop="stills (one setup)", rows=by(reg, "gst"), src=C, stills=True),
        dict(code="slam", pop="UAV video with flight log", rows=by(reg, "slam"), src=C),
        dict(code="vld", pop="LiDAR point clouds (10 files)", rows=[], src="characterisation/sequences.csv"),
    ]


def short(i):
    return i.split("_", 1)[1]


def stability(T, ids):
    parts = []
    for i in ids:
        r = T["rep_raw"].get(i)
        if r:
            parts.append(f"{short(i)}: {r['success_count']} of {r['n']} ({pct(r['success_rate'])})")
    if not parts:
        return "single run"
    return "; ".join(parts) + ("; others single run" if len(parts) < len(ids) else "")


def art_flags(rows):
    if not rows or all(all(miss(r[f]) for f in FLAGS) for r in rows):
        return "not annotated"
    out = []
    for f in FLAGS:
        for val, n in Counter(r[f] for r in rows if not miss(r[f]) and r[f] != "no").items():
            out.append(f"{f} {val} ({n} of {len(rows)})" if len(rows) > 1 else f"{f} {val}")
    return ", ".join(out) or "none annotated"


def doc_phrase(T, code):
    ph = [r["short_phrase"] for r in T["doc"]
          if r["dataset_code"] == code and r["short_phrase"] and r["verdict"] != "consistent"]
    return "; ".join(ph) if ph else "no disagreement found"


def b1(T):
    fq = {rid(r): r for r in T["fq"]}
    art = {rid(r): r for r in T["art"]}
    a, b = [], []
    for p in populations(T):
        rows, ids = p["rows"], [rid(r) for r in p["rows"]]
        name = DS_NAME[p["code"]]
        if p["code"] == "vld":
            a.append([name, p["pop"], "10", NA, NA, NA, NA, NA, NA, p["src"]])
            b.append([name, p["pop"], NA, NA, NA, NA, doc_phrase(T, "vld")])
            continue
        st = p.get("stills")
        a.append([name, p["pop"], str(len(rows)), agg(rows, "reg_rate", "pct"), stability(T, ids),
                  agg(rows, "n_models", "count"),
                  NA if st else agg(rows, "track_len", "f2"),
                  NA if st else agg(rows, "tri_angle_median", "deg"),
                  NA if st else agg(rows, "view_coverage_sr", "f2"), p["src"]])
        if p.get("content"):
            b.append([name, p["pop"], p["content"], p["content"], p["content"], p["content"],
                      doc_phrase(T, p["code"])])
        else:
            f = [fq[i] for i in ids]
            b.append([name, p["pop"], agg(f, "blur_share_below", "pct"),
                      agg(f, "clip_share_median", "pct"),
                      NA if st else agg(f, "static_share", "pct"),
                      art_flags([art[i] for i in ids if i in art]), doc_phrase(T, p["code"])])
    ha = ["Dataset", "Population", "Sequences", "Registered share (`reg_rate`), median [min–max]",
          "Registration stability (repeat study)", "Models (`n_models`), median [min–max]",
          "Track length (`track_len`), median [min–max]",
          "Triangulation angle in degrees (`tri_angle_median`), median [min–max]",
          "Viewpoint coverage in sr (`view_coverage_sr`), median [min–max]", "Registration values from"]
    hb = ["Dataset", "Population", "Blurred-frame share (`blur_share_below`), median [min–max]",
          "Clipped-pixel share (`clip_share_median`), median [min–max]",
          "Static-pixel share (`static_share`), median [min–max]", "Lens artifacts, annotated by eye",
          "Documentation reliability"]
    return [dict(title="B1a. Registration yield per dataset", headers=ha, rows=a),
            dict(title="B1b. Image content and documentation per dataset", headers=hb, rows=b)]


def b2_ids(T):
    """(id, registration row, source) for the 34 characterised sequences, the parts, the fronts."""
    reg = {rid(r): r for r in T["reg"]}
    q = {r["id"]: r for r in T["q"]}
    order = []
    for code, pop in (("mots", None), ("blt", None)):
        order += [(rid(r), r, "characterisation/registration.csv") for r in T["reg"] if r["dataset"] == code]
    order += [(r["id"], r, "ablations/blt.csv (raw)") for r in T["blt_raw"]]
    order += [(i, q[i], "quality/quality.csv") for i in sorted(q) if i.startswith("btg_") and "_p" in i]
    for code in ("btg", "bot", "emb", "esc", "gcs", "gst", "slam"):
        order += [(rid(r), r, "characterisation/registration.csv") for r in T["reg"] if r["dataset"] == code]
    assert len(order) == 51 and len({o[0] for o in order}) == 51, len(order)
    return order


def b2(T):
    rec = {r["id"]: r for r in T["rec"]}
    fq = {rid(r): r for r in T["fq"]}
    art = {rid(r): r for r in T["art"]}
    order = b2_ids(T)
    a, b, c = [], [], []
    for i, r, src in order:
        m = rec[i]
        a.append([i, m["source"], integer(m["n_frames_total"]), integer(m["n_images"]), m["k"],
                  num(m["frame_spacing_s"], 2), m["width_used"], m["native_size"]])
        st = T["rep_raw"].get(i)
        b.append([i, f"{integer(r['n_registered'])} / {integer(r['n_images'])}", pct(r["reg_rate"]),
                  f"{st['success_count']} of {st['n']}" if st else "single run",
                  f"{r['n_models']} ({sizes(r['model_sizes'])})", num(r["reproj_err"], 2),
                  num(r["track_len"], 2), num(r["tri_angle_median"], 1),
                  num(r["view_coverage_sr"], 2), num(r.get("wall_s"), 0), r["outcome"], src])
        if i in fq:
            f, x = fq[i], art.get(i)
            flags = art_flags([x]) if x else "not annotated"
            c.append([i, integer(f["n_frames"]), plain(f["blur_median"]), pct(f["blur_share_below"]),
                      pct(f["clip_share_median"]), pct(f["clip_frames_gt5"]),
                      plain(f["lum_mean_sigma"]),
                      NA if f["static_share"] == "n/a" else pct(f["static_share"]),
                      pct(f["black_frames_share"]), flags,
                      (x or {}).get("confidence") or DASH, (x or {}).get("notes") or DASH])
    ha = ["Sequence", "Source file(s) (`source`)", "Frames available (`n_frames_total`)",
          "Frames used (`n_images`)", "k", "Frame spacing in s (`frame_spacing_s`)",
          "Width used in px (`width_used`)", "Native size in px (`native_size`)"]
    hb = ["Sequence", "Registered / given", "Registered share (`reg_rate`)", "Stability (repeat study)",
          "Models (sizes)", "Reprojection error in px (`reproj_err`)", "Track length (`track_len`)",
          "Triangulation angle in degrees (`tri_angle_median`)",
          "Viewpoint coverage in sr (`view_coverage_sr`)", "COLMAP wall time in s (`wall_s`)",
          "Outcome (`outcome`)", "Values from"]
    hc = ["Sequence", "Frames measured (`n_frames`)", "Blur, median Laplacian variance (`blur_median`)",
          "Blurred-frame share (`blur_share_below`)", "Clipped-pixel share (`clip_share_median`)",
          "Frames with more than 5 % clipped (`clip_frames_gt5`)",
          "Exposure variation (`lum_mean_sigma`)", "Static-pixel share (`static_share`)",
          "Black-frame share (`black_frames_share`)", "Lens artifacts, annotated by eye",
          "Annotation confidence", "Annotator note (`notes`)"]
    return [dict(title="B2a. Sampling per sequence", headers=ha, rows=a),
            dict(title="B2b. Registration per sequence", headers=hb, rows=b),
            dict(title="B2c. Image content per sequence", headers=hc, rows=c)]


def distinct(vals, f=lambda v: v):
    out = []
    for v in vals:
        v = f(v)
        if v not in out:
            out.append(v)
    return out


def b3(T):
    rec = {r["id"]: r for r in T["rec"]}
    anc = {r["dataset"]: r for r in T["anc"]}
    slam = [g for g in T["geo"] if g["kind"] == "uav_rtk_log"][0]
    rows = []
    for p in populations(T):
        name, code = DS_NAME[p["code"]], p["code"]
        an = anc.get(code)
        if an:
            atype, ares = an["anchor_type"], f"present: {an['anchor_present']}; resolvable: {an['anchor_resolvable']}"
        else:
            atype = "DJI flight log shipped with the video"
            ares = f"present: yes; resolved: {slam['n_matched']} of {slam['n_cameras']} cameras aligned (B9)"
        if code == "vld":
            rows.append([name, p["pop"], "10", NA, NA, NA, NA, NA, atype, ares])
            continue
        ms = [rec[rid(r)] for r in p["rows"]]
        fps = distinct([m["fps_source"] for m in ms], lambda v: DASH if miss(v) else plain(v))
        nat = distinct([m["native_size"] for m in ms])
        alt = "; ".join(f"{m['sequence_id']}: {num(m['exif_gps_alt_median_m'], 1)} "
                        f"[{num(m['exif_gps_alt_min_m'], 1)}–{num(m['exif_gps_alt_max_m'], 1)}]"
                        for m in ms if m["exif_gps_alt_median_m"]) or DASH
        rows.append([name, p["pop"], str(len(ms)), agg(ms, "n_frames_total", "count"),
                     agg(ms, "n_images", "count"), ", ".join(fps), "; ".join(nat), alt, atype, ares])
    h = ["Dataset", "Population", "Sequences", "Frames available (`n_frames_total`), median [min–max]",
         "Frames used (`n_images`), median [min–max]", "Frame rate measured in fps (`fps_source`)",
         "Native image size in px (`native_size`)", "EXIF GPS altitude in m, median [min–max] per flight",
         "Anchor type (`anchor_type`)", "Anchor present and resolvable"]
    return [dict(title="B3. Observation and recording values", headers=h, rows=rows)]


def b4(T):
    dis, con = [], []
    for r in T["doc"]:
        row = [r["dataset"], r["item"], f"{r['documented_value']}. Stated in: {r['documented_where']}",
               f"{r['measured_value']}. How: {r['measured_how']}", r["note"] or DASH, r["verdict"],
               r["source"]]
        (con if r["verdict"] == "consistent" else dis).append(row)
    h = ["Dataset", "Item", "Documented value and where it is stated", "Measured value and how", "Note",
         "Kind", "Source of the check"]
    return [dict(title="B4a. Disagreements between documentation and data", headers=h, rows=dis),
            dict(title="B4b. Checks where documentation and data agree", headers=h, rows=con)]


STD_BLOCK = ("blocks of four consecutive frames from frame 16, every 32 (16-19, 48-51, ...); "
             "a block touching the end of the sequence is dropped")


def b5(T):
    a, b = [], []
    for r in T["q"]:
        a.append([DS_NAME[r["dataset"]], r["id"], r["scene_group"],
                  f"{integer(r['n_registered'])} / {integer(r['n_images'])}", r["interp_n_views"],
                  f"{num(r['interp_psnr_mean'], 2)} ± {num(r['interp_psnr_std'], 2)}",
                  num(r["interp_ssim_mean"], 3), num(r["interp_lpips_mean"], 3),
                  pct(r["interp_nn_dist_frac"]), num(r["interp_nn_angle_deg"], 1),
                  integer(r["interp_n_gaussians"]), integer(r["interp_t_train_s"])])
        rule = "standard" if r["block_split_rule"] == STD_BLOCK else r["block_split_rule"]
        b.append([r["id"], rule, r["block_n_views"],
                  f"{num(r['block_psnr_mean'], 2)} ± {num(r['block_psnr_std'], 2)}",
                  num(r["block_ssim_mean"], 3), num(r["block_lpips_mean"], 3),
                  pct(r["block_nn_dist_frac"]), num(r["block_nn_angle_deg"], 1),
                  integer(r["block_n_gaussians"]), integer(r["block_t_train_s"])])
    ha = ["Dataset", "Sequence", "Scene group", "Registered / given", "Test views",
          "PSNR in dB, mean ± SD over views", "SSIM", "LPIPS", "`nn_dist_frac`", "`nn_angle_deg` in degrees",
          "Gaussians", "Training time in s"]
    hb = ["Sequence", "Block rule", "Test views", "PSNR in dB, mean ± SD over views", "SSIM", "LPIPS",
          "`nn_dist_frac`", "`nn_angle_deg` in degrees", "Gaussians", "Training time in s"]
    return [dict(title="B5a. Every-8th-frame split (interpolation)", headers=ha, rows=a),
            dict(title="B5b. Block split", headers=hb, rows=b)]


def conditions(rows):
    out = {}
    for r in rows:
        out.setdefault((r["sequence"], r["condition"]), []).append(r)
    return out


def region_means(T):
    by = {}
    for r in T["regions"]:
        by.setdefault(r["run"], []).append(r)
    return by


def b6(T):
    geo = {g["run"]: g for g in T["geo"]}
    regions = region_means(T)
    out = []
    # Botrytis
    a, b = [], []
    for (seq, cond), rs in conditions(T["abl"]["botrytis"]).items():
        r = rs[0]
        g = geo.get(f"C/botrytis/{seq}/{cond}")
        a.append([seq, cond, f"{integer(r['n_registered'])} / {integer(r['n_images'])}",
                  f"{r['n_models']} ({sizes(r['model_sizes'])})", num(r["reproj_err"], 2),
                  num(r["track_len"], 2), num(r["tri_angle_median"], 1), pct(r["perframe_failure_share"]),
                  num(g["ate_rmse_m"], 3) if g else DASH, num(g["scale_m_per_unit"], 3) if g else DASH,
                  f"{g['n_matched']} / {g['n_cameras']}" if g else DASH])
        b.append([seq, cond, str(len(rs)), r["n_views"], seed_stat(rs, "psnr_mean", 2),
                  seed_stat(rs, "ssim_mean", 3), seed_stat(rs, "lpips_mean", 3)])
    out.append(dict(title="B6a. Botrytis band composition: registration and geometric anchor",
                    headers=["Flight", "Condition", "Registered / given", "Models (sizes)",
                             "Reprojection error in px", "Track length", "Triangulation angle in degrees",
                             "Per-frame alignment failure share", "ATE against EXIF GPS in m",
                             "Scale in m per unit", "Cameras matched / registered"], rows=a))
    out.append(dict(title="B6b. Botrytis band composition: held-out quality",
                    headers=["Flight", "Condition", "Seeds", "Test views", "PSNR in dB, seed mean [range]",
                             "SSIM, seed mean [range]", "LPIPS, seed mean [range]"], rows=b))
    # MOTS
    a, b = [], []
    for (seq, cond), rs in conditions(T["abl"]["mots"]).items():
        r = rs[0]
        a.append([seq, cond, f"{integer(r['n_registered'])} / {integer(r['n_images'])}",
                  f"{r['n_models']} ({sizes(r['model_sizes'])})", num(r["reproj_err"], 2),
                  num(r["track_len"], 2), num(r["tri_angle_median"], 1), r["loss_masked"]])
        reg = regions[f"results/full_program/C/mots/{seq}/{cond}/seed0"]
        b.append([seq, cond, str(len(rs)), r["n_views"], seed_stat(rs, "psnr_mean", 2),
                  seed_stat(rs, "psnr_common", 2), seed_stat(rs, "ssim_mean", 3),
                  seed_stat(rs, "ssim_common", 3), seed_stat(rs, "lpips_mean", 3),
                  seed_stat(rs, "lpips_common", 3), pct(r["common_mask_share"]),
                  rnd(mean([dec(x["psnr_cluster"]) for x in reg]), 2),
                  rnd(mean([dec(x["psnr_canopy"]) for x in reg]), 2)])
    out.append(dict(title="B6c. MOTS preprocessing: registration",
                    headers=["Sequence", "Condition", "Registered / given", "Models (sizes)",
                             "Reprojection error in px", "Track length", "Triangulation angle in degrees",
                             "Loss masked (`loss_masked`)"], rows=a))
    out.append(dict(title="B6d. MOTS preprocessing: held-out quality",
                    headers=["Sequence", "Condition", "Seeds", "Test views",
                             "PSNR full frame in dB, seed mean [range]",
                             "PSNR common pixels in dB, seed mean [range]", "SSIM full frame",
                             "SSIM common pixels", "LPIPS full frame", "LPIPS common pixels",
                             "Common-pixel share", "PSNR on clusters in dB, seed 0",
                             "PSNR on canopy in dB, seed 0"], rows=b))
    # BLT
    mask = {(m["sequence"], m["condition"]): m for m in T["mask"] if m["kind"] == "condition"}
    a, b = [], []
    for (seq, cond), rs in conditions(T["abl"]["blt"]).items():
        r, m = rs[0], mask.get((seq, cond))
        rate = r["registration_success_rate"]
        a.append([seq, cond, m["mask_kind"] if m else "none", pct(m["masked_share_mean"]) if m else DASH,
                  (m["sfm_mask"] if m else "False"), (m["loss_mask"] if m else "False"),
                  f"{integer(r['n_registered'])} / {integer(r['n_images'])}",
                  f"{r['n_models']} ({sizes(r['model_sizes'])})", num(r["reproj_err"], 2),
                  num(r["track_len"], 2), num(r["tri_angle_median"], 1),
                  pct(rate) if not miss(rate) else "not repeated", r["pending_repeats"]])
        trained = [x for x in rs if not miss(x["psnr_mean"])]
        if trained:
            b.append([seq, cond, str(len(trained)), r["n_views"], seed_stat(trained, "psnr_mean", 2),
                      seed_stat(trained, "ssim_mean", 3), seed_stat(trained, "lpips_mean", 3)])
        else:
            b.append([seq, cond, "0", DASH, "not trained", "not trained", "not trained"])
    out.append(dict(title="B6e. BLT front passes and masking: registration",
                    headers=["Pass", "Condition", "Mask kind", "Masked share", "Mask in SfM", "Mask in loss",
                             "Registered / given, the run that fed training", "Models (sizes)",
                             "Reprojection error in px", "Track length", "Triangulation angle in degrees",
                             "Repeat success rate (`registration_success_rate`)", "`pending_repeats`"],
                    rows=a))
    out.append(dict(title="B6f. BLT front passes and masking: held-out quality",
                    headers=["Pass", "Condition", "Seeds", "Test views", "PSNR in dB, seed mean [range]",
                             "SSIM, seed mean [range]", "LPIPS, seed mean [range]"], rows=b))
    return out


def b7(T):
    rows = []
    for r in T["rep"]:
        rows.append([r["condition"], r["sequence"], r["mask_kind"], r["n"], integer(r["n_images"]),
                     r["success_count"], pct(r["success_rate"]),
                     plain(r["registered_median"]) + ("" if r["registered_min"] == r["registered_max"]
                                                      else f" [{r['registered_min']}–{r['registered_max']}]"),
                     sizes(r["registered_all"]), r["distinct_first_init_pairs"],
                     r["distinct_largest_init_pairs"], plain(r["init_attempts_median"])])
    h = ["Condition", "Sequence", "Mask in SfM (`mask_kind`)", "Runs", "Frames", "Successes",
         "Success rate", "Registered, median [min–max]", "Registered per run",
         "Distinct first initial pairs", "Distinct pairs of the attempt that registered most images (log proxy)",
         "Initialisation attempts, median"]
    return [dict(title="B7. Registration repeat study", headers=h, rows=rows)]


def b8(T):
    order = {"fps": 0, "width": 1, "count": 2}
    rows = []
    for r in sorted(T["sw"], key=lambda r: (r["sequence"], order[r["sweep"]], dec(r["value"]))):
        level = r["value"] + (" (shared level)" if r["note"].startswith("shared level") else "")
        rate = dec(r["n_registered"]) / dec(r["n_frames"])
        rows.append([r["sequence"], r["sweep"], level, r["sfm_width"], r["train_width"],
                     f"{r['n_registered']} / {r['n_frames']}", rnd(rate * 100, 1) + " %",
                     f"{r['n_models']} ({sizes(r['model_sizes'])})", num(r["tri_angle_median"], 1),
                     pct(r["baseline_frac"]), num(r["psnr"], 2), num(r["psnr_at640"], 2),
                     num(r["ssim"], 3), num(r["lpips"], 3), pct(r["nn_dist_frac"]), num(r["span_s"], 2)])
    h = ["Sequence", "Sweep", "Level (`value`)", "SfM width in px", "Training width in px",
         "Registered / frames", "Registration rate", "Models (sizes)", "Triangulation angle in degrees",
         "Baseline fraction (`baseline_frac`)", "PSNR native in dB", "PSNR at 640 px in dB", "SSIM", "LPIPS",
         "`nn_dist_frac`", "Span in s"]
    return [dict(title="B8. Requirement sweeps", headers=h, rows=rows)]


def b9(T):
    rows = []
    for g in T["geo"]:
        kind = g["kind"]
        if kind == "blt":
            st = json.loads(g["status_counts"])
            atype = "RTK-GNSS fixes, topic /gps/fix"
            acc = "; ".join(f"{n} fixes with status {s}" for s, n in st.items())
            off = "none: fixes interpolated to frame header stamps"
        elif kind == "bot":
            atype = "EXIF GPS per capture"
            acc = f"none stated ({g['n_gps']} captures with GPS)"
            off = "none: one position per capture"
        else:
            atype = "DJI flight log shipped with the video"
            acc = "log reports P-GPS, gpslevel 5"
            off = "searched over -4 to 2 s, interior optimum"
        rows.append([DS_NAME[g["id"].split("_")[0]], g["run"], atype, acc,
                     f"{g['n_matched']} / {g['n_cameras']}", num(g["ate_rmse_m"], 3),
                     num(g["scale_m_per_unit"], 3), off, g["note"] or DASH])
    h = ["Dataset", "Run (sequence or condition)", "Anchor type", "Anchor's own stated accuracy class",
         "Cameras matched / registered", "ATE RMSE in m (`ate_rmse_m`)",
         "Scale in m per unit (`scale_m_per_unit`)", "Time offset", "Note in the CSV (`note`)"]
    return [dict(title="B9. Scale anchors", headers=h, rows=rows)]


def b10(T):
    rows = []
    for run, rs in region_means(T).items():
        m = lambda c, nd: rnd(mean([dec(x[c]) for x in rs]), nd)
        rel = run.replace("results/full_program/", "")
        parts = rel.split("/")
        what = f"quality row, {parts[2]} split" if parts[0] == "B" else f"condition {parts[3]}, seed 0"
        rows.append([parts[1] if parts[0] == "B" else parts[2], what, str(len(rs)),
                     rnd(mean([dec(x["cluster_share"]) for x in rs]) * 100, 1) + " %",
                     m("psnr_cluster", 2), m("ssim_cluster", 3), m("lpips_cluster", 3),
                     m("psnr_canopy", 2), m("ssim_canopy", 3), m("lpips_canopy", 3), rel])
    h = ["Sequence", "Run", "Views", "Cluster mask coverage (`cluster_share`), mean",
         "PSNR on clusters in dB", "SSIM on clusters", "LPIPS on clusters", "PSNR on canopy in dB",
         "SSIM on canopy", "LPIPS on canopy", "Run directory under results/full_program/"]
    return [dict(title="B10. Grape-region metric (MOTS), mean over held-out views", headers=h, rows=rows)]


def tables(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = load()
    sections = dict(B1=b1(T), B2=b2(T), B3=b3(T), B4=b4(T), B5=b5(T), B6=b6(T), B7=b7(T), B8=b8(T),
                    B9=b9(T), B10=b10(T))
    for key, tabs in sections.items():
        md = "\n\n".join(f"### {t['title']}\n\n{table(t['headers'], t['rows'])}" for t in tabs)
        (out / f"{key}.md").write_text(md + "\n")
        print(key, [(t["title"].split(".")[0], len(t["rows"]), len(t["headers"])) for t in tabs], len(md))
    (out / "tables.json").write_text(json.dumps(sections, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- the package

DESCR = [
    (r"^README\.md$", "this file: contents, one line per file, source commit"),
    (r"^METHODS\.md$", "the protocol as run (as in the repository)"),
    (r"^RUNS\.md$", "provenance index: every table row traced to run directory, SLURM job and date (results/RUNS.md in the repository)"),
    (r"^INVENTORY\.md$", "what exists on the cluster: storage, earlier results, data locations, candidate datasets"),
    (r"^data_dictionary\.md$", "every CSV: column, meaning, unit, missing-value encoding"),
    (r"^results/characterisation/sequences\.csv$", "one row per sampled sequence or LiDAR file: source, frames, sampling, notes"),
    (r"^results/characterisation/frame_quality\.csv$", "per sequence: blur, clipping, exposure variation, static and black shares"),
    (r"^results/characterisation/registration\.csv$", "per sequence: fixed-COLMAP registration of the characterisation run"),
    (r"^results/characterisation/artefacts\.csv$", "per sequence: lens artifacts annotated by eye from contact sheets"),
    (r"^results/characterisation/anchors\.csv$", "per dataset: anchor type, presence, resolvability"),
    (r"^results/characterisation/recording\.csv$", "per sequence incl. parts and front passes: native size, frame rate, sampling, EXIF GPS altitude (collected from per_sequence/*.meta.json)"),
    (r"^results/characterisation/documentation_checks\.csv$", "documentation-versus-data checks with a source per row (transcribed from the check outputs; table B4)"),
    (r"^results/characterisation/frame_lists/", "frame list of the sequence: the sampled frames, in order"),
    (r"^results/characterisation/poses/", "camera poses of the registered frames of the sequence (characterisation run)"),
    (r"^results/characterisation/static_masks/", "static-pixel mask of the sequence (white = static)"),
    (r"^results/characterisation/per_sequence/.*\.meta\.json$", "sampling record of the sequence (source, frames, k, native size, notes)"),
    (r"^results/characterisation/per_sequence/.*\.quality\.json$", "frame-quality record of the sequence"),
    (r"^results/characterisation/per_sequence/.*\.registration\.json$", "registration record of the sequence (characterisation run)"),
    (r"^results/characterisation/per_sequence/", "per-dataset record of the characterisation run"),
    (r"^results/quality/quality\.csv$", "reconstruction quality, 28 gated sequences, both held-out splits"),
    (r"^results/ablations/botrytis\.csv$", "Botrytis band-composition ablation, one row per flight x condition x seed"),
    (r"^results/ablations/mots\.csv$", "MOTS preprocessing ablation, one row per sequence x condition x seed"),
    (r"^results/ablations/blt\.csv$", "BLT front passes and masking, one row per pass x condition x seed, with repeat success rates"),
    (r"^results/ablations/masks/quality\.csv$", "mask table: masked share per BLT condition, hand-mask provenance, SAM 3 stability figures (collected from mask reports)"),
    (r"^results/sweeps/sweeps\.csv$", "requirement sweeps: one row per level"),
    (r"^results/anchors/geometric\.csv$", "scale anchors: trajectory error and scale per aligned reconstruction"),
    (r"^results/repeats/repeats\.csv$", "registration repeat study: one row per COLMAP run"),
    (r"^results/repeats/summary\.csv$", "registration repeat study: one row per condition"),
    (r"^results/repeats/runs/", "repeat-study run record with every initial pair tried"),
    (r"^results/regions/mots_regions\.csv$", "grape-region metric: one row per held-out view"),
    (r"^results/joined\.csv$", "every characterisation property beside every quality column, one row per sequence"),
    (r"^results/debug/.*/summary\.json$", "trainer-check run: flat record of metrics, counts, timings"),
    (r"^results/debug/.*/metrics_full\.json$", "trainer-check run: PSNR, SSIM, LPIPS per held-out view"),
    (r"^results/debug/.*/metrics\.json$", "trainer-check run: trainer's own PSNR/SSIM and view overlap"),
    (r"^results/debug/.*\.registration\.json$", "trainer-check run: registration record"),
    (r"^results/masks/hand/.*\.png$", "hand mask (white = used, black = excluded)"),
    (r"^results/masks/hand/.*\.json$", "hand-mask provenance: region coordinates and source"),
    (r"^results/masks/reference_frames/", "reference frame for the masks (single frame, temporal median or static-pixel map)"),
    (r"^results/masks/sam3/", "SAM 3 attempt: step log and stability figures"),
    (r"^figures/[^/]+/meta\.json$", "per-panel record: run, selection rule, frame, PSNR, SSIM, LPIPS"),
    (r"^figures/[^/]+/sheet\.jpg$", "contact sheet of the set"),
    (r"^figures/.*_pred\.png$", "render of the held-out view"),
    (r"^figures/.*_gt\.png$", "ground truth of the held-out view"),
    (r"^figures/.*_clustermask\.png$", "grape-cluster mask of the view"),
    (r"^figures/static_masks/", "static-pixel map or its reference frame (chassis against sky)"),
    (r"^doc_tables/B\d+\.md$", "the Doc's value tables of this section, as generated from the CSVs"),
    (r"^doc_tables/tables\.json$", "all Doc value tables as JSON (headers and rows)"),
    (r"^doc_tables/handoff\.py$", "the script that collected the supplementary tables and generated the Doc tables and this package"),
    (r"^doc_tables/handoff_spotcheck\.py$", "independent spot-check: five values per Doc table recomputed from the CSVs (run from this directory)"),
    (r"^sources/disk_checks/disk_checks_report\.md$", "dataset checks of 30 Sep 2026: methods and findings (source of several B4 rows)"),
    (r"^sources/disk_checks/disk_checks\.json$", "dataset checks, machine-readable"),
    (r"^sources/disk_checks/video_stats\.csv$", "resolution, frame rate and frame count of all 51 Terras Gauda and MOTS videos"),
]


def describe(rel):
    for pat, d in DESCR:
        if re.search(pat, rel):
            return d
    raise SystemExit(f"no description for {rel}")


def pack(dst, tables_dir, dictionary, disk_checks, doc_url=""):
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    def cp(src, rel):
        t = dst / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, t)

    for name in ("METHODS.md", "INVENTORY.md"):
        cp(REPO / name, name)
    cp(RES / "RUNS.md", "RUNS.md")
    cp(dictionary, "data_dictionary.md")
    ch = RES / "characterisation"
    for f in sorted(ch.glob("*.csv")):
        cp(f, f"results/characterisation/{f.name}")
    for sub in ("frame_lists", "poses", "static_masks", "per_sequence"):
        for f in sorted((ch / sub).iterdir()):
            if f.is_file():
                cp(f, f"results/characterisation/{sub}/{f.name}")
    for rel in ("quality/quality.csv", "ablations/botrytis.csv", "ablations/mots.csv", "ablations/blt.csv",
                "ablations/masks/quality.csv", "sweeps/sweeps.csv", "anchors/geometric.csv",
                "repeats/repeats.csv", "repeats/summary.csv", "regions/mots_regions.csv", "joined.csv"):
        cp(RES / rel, f"results/{rel}")
    for f in sorted((RES / "repeats/runs").glob("*/*.json")):
        cp(f, f"results/repeats/runs/{f.parent.name}/{f.name}")
    for d in sorted((RES / "debug/trainer_check").iterdir()):
        for f in sorted(d.glob("*.json")):
            cp(f, f"results/debug/{d.name}/{f.name}")
    for sub in ("hand", "reference_frames"):
        for f in sorted((RES / "masks" / sub).iterdir()):
            if f.is_file() and "PROPOSED" not in f.name:
                cp(f, f"results/masks/{sub}/{f.name}")
    cp(RES / "masks/sam3/blt_uk_20230726_front.report.json", "results/masks/sam3/blt_uk_20230726_front.report.json")
    for f in sorted((RES / "figures").rglob("*")):
        if f.is_file():
            cp(f, f"figures/{f.relative_to(RES / 'figures')}")
    for f in sorted(Path(tables_dir).iterdir()):
        cp(f, f"doc_tables/{f.name}")
    cp(Path(__file__), "doc_tables/handoff.py")
    cp(Path(__file__).with_name("handoff_spotcheck.py"), "doc_tables/handoff_spotcheck.py")
    for name in ("disk_checks_report.md", "disk_checks.json", "video_stats.csv"):
        cp(Path(disk_checks) / name, f"sources/disk_checks/{name}")

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    files = sorted(str(f.relative_to(dst)) for f in dst.rglob("*") if f.is_file()) + ["README.md"]
    files = sorted(set(files))
    lines = [
        f"# Hand-off package for the paper ({dst.name})",
        "",
        "Everything the paper quotes from the cluster side, in one directory. Nothing in here was",
        "computed for the hand-off: tables were collected from kept run files, figures are the",
        "figure pack as produced.",
        "",
        f"- Source: repository `vineyard-scene-reconstruction`, branch `full-program`, commit `{head}`"
        + (" (tracked files modified at pack time: see below)." if dirty else " (working tree clean at pack time)."),
        f"- Companion Doc with the values to quote: {doc_url}" if doc_url else "",
        "- Mask convention everywhere: white = used, black = excluded.",
        "- Not included by design: raw renders beyond the figure pack, COLMAP databases, checkpoints,",
        "  and the contact sheets of the characterisation run (deliberately unpublished).",
        "",
        "Three tables are not measurement outputs but collections made for this hand-off, each with",
        "a source column: `results/characterisation/recording.csv` (from the per-sequence records),",
        "`results/characterisation/documentation_checks.csv` (transcribed from the check outputs)",
        "and `results/ablations/masks/quality.csv` (from the mask reports). The requested",
        "`debug/` holds all eight trainer-check runs: the five-run distortion and initial-scale",
        "study, its two follow-ups (a second seed and an exact repeat), and the superseded",
        "distorted debug run 2.",
        "",
        "## Files",
        "",
        "| File | Size (bytes) | Content |",
        "| --- | --- | --- |",
    ]
    if dirty:
        lines[6:6] = ["", "Tracked files modified at pack time:", "", "```", dirty, "```"]
    body = [ln for ln in lines if ln is not None]
    rows = []
    for rel in files:
        if rel == "README.md":
            continue
        rows.append(f"| `{rel}` | {(dst / rel).stat().st_size} | {describe(rel)} |")
    readme = "\n".join(body + [f"| `README.md` | (this file) | {describe('README.md')} |"] + rows) + "\n"
    (dst / "README.md").write_text(readme)

    zpath = dst.parent / f"{dst.name}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(dst.rglob("*")):
            if f.is_file():
                z.write(f, f"{dst.name}/{f.relative_to(dst)}")
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        names = {n[len(dst.name) + 1:] for n in z.namelist()}
    listed = set(re.findall(r"^\| `([^`]+)` \|", readme, flags=re.M))
    assert names == listed, (names ^ listed)
    md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
    print(json.dumps(dict(zip=str(zpath), size_bytes=zpath.stat().st_size, md5=md5, files=len(names),
                          commit=head, dirty=bool(dirty)), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        collect()
    elif cmd == "tables":
        tables(sys.argv[2])
    elif cmd == "pack":
        pack(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
