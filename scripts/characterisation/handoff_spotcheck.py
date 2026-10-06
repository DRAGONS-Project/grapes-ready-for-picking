"""Independent spot-check of the hand-off value tables: five values per table recomputed
straight from the CSVs with plain floats (no code shared with handoff.py) and compared with
the table cells. Run from the repository root or from the hand-off directory:

    python3 <this file> <tables.json>

One known difference: a tie (Botrytis triangulation-angle minimum, 19.65) that the tables round
half-up on the CSV string (19.7) and float formatting rounds down (19.6).
"""
import csv, json, statistics as st, sys
R = "results/"
def rd(p): return list(csv.DictReader(open(R + p)))
T = json.load(open(sys.argv[1]))
tabs = {t["title"].split(".")[0]: t for sec in T.values() for t in sec}
def cell(tab, key, col):
    t = tabs[tab]
    j = [i for i, h in enumerate(t["headers"]) if col in h]
    assert len(j) == 1, (tab, col, j)
    rows = [r for r in t["rows"] if all(k in r for k in key)]
    assert len(rows) == 1, (tab, key, len(rows))
    return rows[0][j[0]]
f = lambda x, nd: f"{x:.{nd}f}"
P = lambda x: f"{100 * x:.1f}"
ok = bad = 0
per = {}
def chk(tab, key, col, expect):
    global ok, bad
    got = cell(tab, key, col)
    per[tab] = per.get(tab, 0) + 1
    if got == expect: ok += 1
    else:
        bad += 1; print(f"DIFF {tab} {key} {col}: table {got!r} vs recomputed {expect!r}")
reg = rd("characterisation/registration.csv"); fq = rd("characterisation/frame_quality.csv")
rec = {r["id"]: r for r in rd("characterisation/recording.csv")}; seq = rd("characterisation/sequences.csv")
q = {r["id"]: r for r in rd("quality/quality.csv")}; geo = {r["run"]: r for r in rd("anchors/geometric.csv")}
blt = rd("ablations/blt.csv"); bot = rd("ablations/botrytis.csv"); mots = rd("ablations/mots.csv")
sw = rd("sweeps/sweeps.csv"); summ = {r["condition"]: r for r in rd("repeats/summary.csv")}; runs = rd("repeats/repeats.csv")
regions = rd("regions/mots_regions.csv"); masks = rd("ablations/masks/quality.csv")
sel = lambda rows, ds, pop=None: [r for r in rows if r["dataset"] == ds and (pop is None or r["population"] == pop)]
fl = lambda rows, c: [float(r[c]) for r in rows if r[c] not in ("", "undetermined", "n/a")]
def mmm(v, g): return f"{g(st.median(v))} [{g(min(v))}–{g(max(v))}]"
# B1a
v = fl(sel(reg, "mots", "row_pass"), "track_len"); chk("B1a", ["GrapeMOTS", "UAV row passes (NoPathPlanning_1 to 3)"], "track_len", mmm(v, lambda x: f(x, 2)))
v = fl(sel(reg, "blt", "gr_session"), "reg_rate"); chk("B1a", ["BLT", "Greek side-camera passes"], "reg_rate", f"{P(st.median(v))} % [{P(min(v))}–{P(max(v))}]")
v = fl(sel(reg, "btg"), "n_models"); chk("B1a", ["Terras Gauda", "whole clips, parts concatenated"], "n_models", f"{st.median(v):g} [{min(v):g}–{max(v):g}]")
v = fl(sel(reg, "bot"), "tri_angle_median"); chk("B1a", ["Botrytis"], "tri_angle_median", mmm(v, lambda x: f(x, 1)))
raw = {}; [raw.setdefault(r["sequence"], r) for r in blt if r["condition"] == "raw"]
v = [float(r["reg_rate"]) for r in raw.values()]; chk("B1a", ["BLT", "front-camera passes, unmasked (5 Greek, 1 UK)"], "reg_rate", f"{P(st.median(v))} % [{P(min(v))}–{P(max(v))}]")
# B1b
v = fl(sel(fq, "blt", "gr_session"), "static_share"); chk("B1b", ["BLT", "Greek side-camera passes"], "static_share", f"{P(st.median(v))} % [{P(min(v))}–{P(max(v))}]")
chk("B1b", ["BLT", "UK side-camera pass"], "clip_share_median", P(fl(sel(fq, "blt", "uk_session"), "clip_share_median")[0]) + " %")
chk("B1b", ["GrapeCS-ML"], "blur_share_below", P(fl(sel(fq, "gcs"), "blur_share_below")[0]) + " %")
v = fl(sel(fq, "mots", "row_pass"), "clip_share_median"); chk("B1b", ["GrapeMOTS", "UAV row passes (NoPathPlanning_1 to 3)"], "clip_share_median", f"{P(st.median(v))} % [{P(min(v))}–{P(max(v))}]")
chk("B1b", ["Embrapa WGISD", "stills"], "blur_share_below", P(fl(sel(fq, "emb", "stills"), "blur_share_below")[0]) + " %")
# B2a (recording.csv, cross-checked with sequences.csv where the sequence is there)
sq = {f"{r['dataset']}_{r['sequence_id']}": r for r in seq}
chk("B2a", ["mots_PathPlanning_2"], "n_frames_total", f"{int(sq['mots_PathPlanning_2']['n_frames_total']):,}")
chk("B2a", ["blt_uk_20230726_front"], "frame_spacing_s", f(float(rec["blt_uk_20230726_front"]["frame_spacing_s"]), 2))
chk("B2a", ["btg_Row7.2_p3"], "n_frames_total", f"{int(rec['btg_Row7.2_p3']['n_frames_total']):,}")
chk("B2a", ["bot_0_V1"], "n_images", sq["bot_0_V1"]["n_images"])
chk("B2a", ["esc_photos"], "native_size", rec["esc_photos"]["native_size"])
# B2b
rg = {f"{r['dataset']}_{r['sequence_id']}": r for r in reg}
chk("B2b", ["mots_NoPathPlanning_1"], "reproj_err", f(float(rg["mots_NoPathPlanning_1"]["reproj_err"]), 2))
chk("B2b", ["blt_gr_20220915"], "Registered / given", f"{rg['blt_gr_20220915']['n_registered']} / {rg['blt_gr_20220915']['n_images']}")
chk("B2b", ["btg_Row7.1"], "Models (sizes)", "8 (" + ", ".join(map(str, json.loads(rg["btg_Row7.1"]["model_sizes"]))) + ")")
chk("B2b", ["bot_30_V1"], "view_coverage_sr", f(float(rg["bot_30_V1"]["view_coverage_sr"]), 2))
chk("B2b", ["blt_gr_20220713_front"], "Registered / given", f"{raw['blt_gr_20220713_front']['n_registered']} / {raw['blt_gr_20220713_front']['n_images']}")
# B2c
fqd = {f"{r['dataset']}_{r['sequence_id']}": r for r in fq}
chk("B2c", ["blt_gr_20220323"], "static_share", P(float(fqd["blt_gr_20220323"]["static_share"])) + " %")
chk("B2c", ["blt_gr_20220608"], "clip_frames_gt5", P(float(fqd["blt_gr_20220608"]["clip_frames_gt5"])) + " %")
chk("B2c", ["gcs_set5"], "blur_share_below", P(float(fqd["gcs_set5"]["blur_share_below"])) + " %")
chk("B2c", ["btg_Row7.4"], "lum_mean_sigma", fqd["btg_Row7.4"]["lum_mean_sigma"])
chk("B2c", ["mots_NoPathPlanning_3"], "blur_median", fqd["mots_NoPathPlanning_3"]["blur_median"])
# B3
g = lambda x: f"{x:g}"
v = [float(rec[i]["n_frames_total"]) for i in rg if i.startswith("blt_gr")]; chk("B3", ["BLT", "Greek side-camera passes"], "n_frames_total", mmm(v, g))
v = [float(rec[i]["n_frames_total"]) for i in q if i.startswith("btg")]; chk("B3", ["Terras Gauda", "parts run (largest part of 11 clips, and the one-part clip Row6.3)"], "n_frames_total", mmm(v, g))
note = sq["bot_30_V1"]["notes"]; assert "GPS altitude median 101.0 m, min 90.6 m, max 106.4 m" in note
assert "30_V1: 101.0 [90.6–106.4]" in cell("B3", ["Botrytis"], "EXIF GPS altitude"); per["B3"] = per.get("B3", 0) + 1; ok += 1
v = [float(sq[i]["n_images"]) for i in ("mots_PathPlanning_1", "mots_PathPlanning_2", "mots_PathPlanning_8")]; chk("B3", ["GrapeMOTS", "UAV orbits (PathPlanning_1, 2, 8)"], "n_images", mmm(v, g))
v = [float(rec[i]["n_frames_total"]) for i in raw]; chk("B3", ["BLT", "front-camera passes, unmasked (5 Greek, 1 UK)"], "n_frames_total", mmm(v, g))
# B4a / B4b: the quoted values must be present in the CSVs they cite
anc = {r["dataset"]: r for r in rd("characterisation/anchors.csv")}
tests = [("B4a", "13 reflectance-panel captures excluded" in sq["bot_0_V1"]["notes"] and "the data paper gives 65" in sq["bot_0_V1"]["notes"]),
         ("B4a", "8 of 10 files" in anc["vld"]["anchor_present"] and "2 files declare no CRS" in anc["vld"]["check"]),
         ("B4a", "0.526 lie within 50 m" in anc["esc"]["check"] and "median 6.3 m, 95th percentile 304 m, max 318 m" in anc["esc"]["notes"]),
         ("B4a", "decoded 3224 frames, container says 3227" in sq["btg_Row4.1"]["notes"]),
         ("B4a", "P-GPS gpslevel 5" in geo["B/slam_Side_Short_Light"]["note"] and geo["B/slam_Side_Short_Light"]["ate_rmse_m"] == "0.056" and geo["B/slam_Side_Short_Light"]["n_matched"] == "49"),
         ("B4b", sorted(float(rg[i]["focal_ratio"]) for i in rg if i.startswith("bot_"))[0] == 0.9656),
         ("B4b", max(float(rg[i]["focal_ratio"]) for i in rg if i.startswith("bot_")) == 1.031),
         ("B4b", {rg[i]["focal_px_doc"] for i in rg if i.startswith("bot_")} == {"1467"}),
         ("B4b", sorted(float(rg[i]["focal_px_doc"]) for i in rg if i.startswith("blt_gr") and rg[i]["focal_px_doc"] != "undetermined")[::3] == [888.4, 889.2]),
         ("B4b", rg["blt_uk_20230726"]["focal_px_doc"] == "942.9" and rg["blt_uk_20230726"]["focal_px_est"] == "966")]
for tab, t in tests:
    per[tab] = per.get(tab, 0) + 1
    if t: ok += 1
    else: bad += 1; print("DIFF", tab, "source text check failed")
# B5a / B5b
chk("B5a", ["mots_PathPlanning_8"], "PSNR", f"{f(float(q['mots_PathPlanning_8']['interp_psnr_mean']),2)} ± {f(float(q['mots_PathPlanning_8']['interp_psnr_std']),2)}")
chk("B5a", ["bot_45_V1_rgb"], "SSIM", f(float(q["bot_45_V1_rgb"]["interp_ssim_mean"]), 3))
chk("B5a", ["blt_gr_20220608"], "LPIPS", f(float(q["blt_gr_20220608"]["interp_lpips_mean"]), 3))
chk("B5a", ["btg_Row7.2_p3"], "Gaussians", f"{int(q['btg_Row7.2_p3']['interp_n_gaussians']):,}")
chk("B5a", ["slam_Side_Short_Light"], "nn_dist_frac", P(float(q["slam_Side_Short_Light"]["interp_nn_dist_frac"])) + " %")
chk("B5b", ["mots_NoPathPlanning_1"], "PSNR", f"{f(float(q['mots_NoPathPlanning_1']['block_psnr_mean']),2)} ± {f(float(q['mots_NoPathPlanning_1']['block_psnr_std']),2)}")
chk("B5b", ["btg_Row4.1_p2"], "SSIM", f(float(q["btg_Row4.1_p2"]["block_ssim_mean"]), 3))
chk("B5b", ["blt_gr_20220915"], "Test views", q["blt_gr_20220915"]["block_n_views"])
chk("B5b", ["bot_30_V1_rgb"], "nn_angle_deg", f(float(q["bot_30_V1_rgb"]["block_nn_angle_deg"]), 1))
chk("B5b", ["emb_video_frames"], "Training time", q["emb_video_frames"]["block_t_train_s"])
# B6
def cond(rows, s, c): return [r for r in rows if r["sequence"] == s and r["condition"] == c]
def sm(rows, col, nd):
    v = [float(r[col]) for r in rows]; return f"{f(sum(v)/len(v), nd)} [{f(min(v), nd)}–{f(max(v), nd)}]"
c = cond(bot, "0_V1", "radiometry_only")[0]; chk("B6a", ["0_V1", "radiometry_only"], "Registered / given", f"{c['n_registered']} / {c['n_images']}")
chk("B6a", ["45_V1", "naive"], "ATE", f(float(geo["C/botrytis/45_V1/naive"]["ate_rmse_m"]), 3))
chk("B6a", ["0_V1", "align_perframe"], "failure share", P(float(cond(bot, "0_V1", "align_perframe")[0]["perframe_failure_share"])) + " %")
chk("B6a", ["45_V1", "green"], "Scale", f(float(geo["C/botrytis/45_V1/green"]["scale_m_per_unit"]), 3))
chk("B6a", ["0_V1", "naive"], "Track length", f(float(cond(bot, "0_V1", "naive")[0]["track_len"]), 2))
for s, cn, col, nd, hdr in [("0_V1", "green", "psnr_mean", 2, "PSNR"), ("45_V1", "full", "psnr_mean", 2, "PSNR"), ("45_V1", "naive", "ssim_mean", 3, "SSIM"), ("0_V1", "full", "lpips_mean", 3, "LPIPS"), ("45_V1", "falsecolor", "psnr_mean", 2, "PSNR")]:
    chk("B6b", [s, cn], hdr, sm(cond(bot, s, cn), col, nd))
for s, cn, col, hdr in [("mots_NoPathPlanning_1", "ghost", "model_sizes", "Models"), ("mots_NoPathPlanning_2", "dehaze", "reproj_err", "Reprojection"), ("mots_NoPathPlanning_1", "dehaze_noguided", "track_len", "Track"), ("mots_NoPathPlanning_2", "dehaze_bright", "tri_angle_median", "Triangulation"), ("mots_NoPathPlanning_1", "dehaze_ghost", "loss_masked", "loss_masked")]:
    r = cond(mots, s, cn)[0]
    e = {"model_sizes": f"{r['n_models']} (" + ", ".join(map(str, json.loads(r["model_sizes"]))) + ")", "reproj_err": f(float(r["reproj_err"]), 2), "track_len": f(float(r["track_len"]), 2), "tri_angle_median": f(float(r["tri_angle_median"]), 1), "loss_masked": r["loss_masked"]}[col]
    chk("B6c", [s, cn], hdr, e)
chk("B6d", ["mots_NoPathPlanning_1", "raw"], "PSNR full frame", sm(cond(mots, "mots_NoPathPlanning_1", "raw"), "psnr_mean", 2))
chk("B6d", ["mots_NoPathPlanning_1", "dehaze_bright"], "PSNR common", sm(cond(mots, "mots_NoPathPlanning_1", "dehaze_bright"), "psnr_common", 2))
chk("B6d", ["mots_NoPathPlanning_2", "ghost"], "SSIM full", sm(cond(mots, "mots_NoPathPlanning_2", "ghost"), "ssim_mean", 3))
rr = [x for x in regions if x["run"].endswith("C/mots/mots_NoPathPlanning_1/dehaze_bright/seed0")]
chk("B6d", ["mots_NoPathPlanning_1", "dehaze_bright"], "PSNR on clusters", f(sum(float(x["psnr_cluster"]) for x in rr) / len(rr), 2))
rr = [x for x in regions if x["run"].endswith("C/mots/mots_NoPathPlanning_1/raw/seed0")]
chk("B6d", ["mots_NoPathPlanning_1", "raw"], "PSNR on canopy", f(sum(float(x["psnr_canopy"]) for x in rr) / len(rr), 2))
mk = {(m["sequence"], m["condition"]): m for m in masks if m["kind"] == "condition"}
chk("B6e", ["blt_uk_20230726_front", "hsv_both"], "Masked share", P(float(mk[("blt_uk_20230726_front", "hsv_both")]["masked_share_mean"])) + " %")
c = cond(blt, "blt_gr_20220915_front", "hand_sfm_only")[0]; chk("B6e", ["blt_gr_20220915_front", "hand_sfm_only"], "Registered / given", f"{c['n_registered']} / {c['n_images']}")
chk("B6e", ["blt_gr_20220713_front", "hand_both"], "registration_success_rate", P(float(cond(blt, "blt_gr_20220713_front", "hand_both")[0]["registration_success_rate"])) + " %")
chk("B6e", ["blt_gr_20220323_front", "raw"], "Track length", f(float(cond(blt, "blt_gr_20220323_front", "raw")[0]["track_len"]), 2))
chk("B6e", ["blt_uk_20230726_front", "raw"], "pending_repeats", cond(blt, "blt_uk_20230726_front", "raw")[0]["pending_repeats"])
for s, cn, col, nd, hdr in [("blt_gr_20220323_front", "raw", "psnr_mean", 2, "PSNR"), ("blt_gr_20220608_front", "raw", "psnr_mean", 2, "PSNR"), ("blt_gr_20220915_front", "hand_train_only", "psnr_mean", 2, "PSNR"), ("blt_uk_20230726_front", "hsv_both", "ssim_mean", 3, "SSIM"), ("blt_gr_20220713_front", "hand_both", "lpips_mean", 3, "LPIPS")]:
    chk("B6f", [s, cn], hdr, sm(cond(blt, s, cn), col, nd))
# B7: recomputed from the per-run table
for cn in ("jul13_front_raw", "jul13_front_hand", "uk_front_hand", "mar23_side_raw", "sep15_front_raw"):
    rs = [r for r in runs if r["condition"] == cn]
    succ = sum(r["success"] == "True" for r in rs)
    assert str(succ) == summ[cn]["success_count"]
    chk("B7", [cn], "Success rate", P(succ / len(rs)) + " %")
# B8
sd = {(r["sequence"], r["sweep"], r["value"]): r for r in sw}
r = sd[("mots_NoPathPlanning_1", "fps", "1")]; chk("B8", ["mots_NoPathPlanning_1", "fps", "24 / 30"], "Registration rate", P(int(r["n_registered"]) / int(r["n_frames"])) + " %")
r = sd[("btg_Row7.2_p3", "fps", "4")]; chk("B8", ["btg_Row7.2_p3", "fps", "131 / 131"], "PSNR native", f(float(r["psnr"]), 2))
r = sd[("btg_Row7.2_p3", "width", "4096")]; chk("B8", ["btg_Row7.2_p3", "width", "4096"], "PSNR at 640", f(float(r["psnr_at640"]), 2))
r = sd[("mots_NoPathPlanning_1", "width", "3840")]; chk("B8", ["mots_NoPathPlanning_1", "width", "3840"], "LPIPS", f(float(r["lpips"]), 3))
r = sd[("btg_Row7.2_p3", "count", "15")]; chk("B8", ["btg_Row7.2_p3", "count", "15 / 15"], "Span", f(float(r["span_s"]), 2))
# B9
for run, col, nd in [("B/blt_gr_20220420", "ate_rmse_m", 3), ("B/blt_gr_20220915", "scale_m_per_unit", 3), ("B/bot_0_V2_rgb", "ate_rmse_m", 3), ("B/slam_Side_Short_Light", "scale_m_per_unit", 3), ("C/botrytis/0_V1/radiometry_only", "ate_rmse_m", 3)]:
    chk("B9", [run], col, f(float(geo[run][col]), nd))
# B10
for run, col, nd, hdr in [("B/mots_NoPathPlanning_1/interp", "psnr_cluster", 2, "PSNR on clusters"), ("B/mots_NoPathPlanning_1/block", "psnr_canopy", 2, "PSNR on canopy"), ("B/mots_NoPathPlanning_2/interp", "ssim_cluster", 3, "SSIM on clusters"), ("C/mots/mots_NoPathPlanning_2/ghost/seed0", "lpips_canopy", 3, "LPIPS on canopy"), ("B/mots_NoPathPlanning_3/block", "cluster_share", None, "cluster_share")]:
    rr = [x for x in regions if x["run"] == "results/full_program/" + run]
    m = sum(float(x[col]) for x in rr) / len(rr)
    chk("B10", [run], hdr, (P(m) + " %") if nd is None else f(m, nd))
print("checks per table:", per)
print("spot-checks passed:", ok, "failed:", bad)
