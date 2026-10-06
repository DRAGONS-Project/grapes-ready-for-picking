"""Sequence definitions for the characterisation measurements (run order = list order)."""

ZEN = {"mots": 10625595, "btg": 7330951, "bot": 7383601, "esc": 10362568, "emb": 3361736, "gst": 14019981, "slam": 11644289}
GR = "/Ktima Gerovassiliou/rosbag_compressed_"

BTG_CLIPS = {  # clip -> number of parts (Row4.1_1 ... Row4.1_4); verified against the Zenodo file list
    "Row4.1": 4, "Row4.2": 3, "Row4.3": 3, "Row4.4": 4, "Row6.1": 5, "Row6.2": 2, "Row6.3": 1,
    "Row7.1": 4, "Row7.2": 4, "Row7.3": 4, "Row7.4": 2, "Row8": 4,
}
BTG_MAIN = ["Row4.1", "Row6.1", "Row7.2", "Row8"]  # longest clip of each row by container frame count


def _video(ds, sid, pop, keys, notes=""):
    return dict(dataset=ds, sequence_id=sid, population=pop, kind="video", record=ZEN[ds], keys=keys, notes=notes)


def _btg(clip):
    return _video("btg", clip, "clip", [f"{clip}_{i}.mp4" for i in range(1, BTG_CLIPS[clip] + 1)])


SEQUENCES = (
    [_video("mots", f"NoPathPlanning_{i}", "row_pass", [f"NoPathPlanning_{i}.mp4"]) for i in (1, 2, 3)]
    + [dict(dataset="blt", sequence_id=f"gr_{d[:10].replace('-', '')}", population="gr_session", kind="blt", bag=GR + d + ".bag")
       for d in ("2022-03-23-12-27-15", "2022-04-20-09-37-20", "2022-06-08-14-03-03", "2022-07-13-15-38-32", "2022-09-15-14-23-20")]
    + [dict(dataset="blt", sequence_id="uk_20230726", population="uk_session", kind="blt", bag="/Riseholme/session2.bag")]
    + [_btg(c) for c in BTG_MAIN]
    + [dict(dataset="bot", sequence_id=f, population="flight", kind="bot", key=f + ".zip") for f in ("0_V1", "30_V1", "45_V1", "0_V2")]
    + [_video("mots", "PathPlanning_1", "orbit", ["PathPlanning_1.mp4"], "Table 3 source; 1920x1080"),
       _video("mots", "PathPlanning_8", "orbit", ["Pathplanning_8.mp4"], "longest 4K orbit (750 frames); PathPlanning_3 excluded because it is 1080p"),
       _video("mots", "PathPlanning_2", "orbit", ["PathPlanning_2.mp4"], "shortest 4K orbit (418 frames)")]
    + [dict(dataset="esc", sequence_id="photos", population="stills", kind="esc"),
       dict(dataset="emb", sequence_id="video_frames", population="video_frames", kind="emb_video"),
       dict(dataset="emb", sequence_id="stills", population="stills", kind="emb_stills"),
       dict(dataset="gcs", sequence_id="set5", population="stills", kind="gcs"),
       dict(dataset="gst", sequence_id="blue_white_setup1", population="stills", kind="gst")]
    + [_btg(c) for c in BTG_CLIPS if c not in BTG_MAIN]
    # added after the main program was fixed: Tomino UAV video with an RTK flight log (Zenodo 11644289)
    + [_video("slam", "Side_Short_Light", "uav_rtk", ["Side_Short_Light.mp4"], "RTK flight log Side_Short_Light.xlsx at 10 Hz ships with the video")]
    # quality rows at part level (a part = one source mp4; the clips are concatenations of separate
    # recordings). Largest part per clip, derived from the clip metas' part-join notes; Row6.3 is
    # single-part, so its clip sequence (index 29) is already its part-level row.
    + [_video("btg", f"{clip}_p{part}", "clip_part", [f"{clip}_{part}.mp4"],
              f"largest of {BTG_CLIPS[clip]} parts of {clip}")
       for clip, part in [("Row4.1", 2), ("Row4.2", 3), ("Row4.3", 1), ("Row4.4", 3), ("Row6.1", 1),
                          ("Row6.2", 2), ("Row7.1", 3), ("Row7.2", 3), ("Row7.3", 2), ("Row7.4", 2),
                          ("Row8", 1)]]
    # quality rows on the composed RGB (the paper's full method); same captures as the committed
    # green-band frame lists, so the held-out names coincide
    + [dict(dataset="bot", sequence_id=f + "_rgb", population="flight_rgb", kind="bot_rgb",
            key=f + ".zip", base="bot_" + f) for f in ("0_V1", "30_V1", "45_V1", "0_V2")]
    # front camera over the committed side-pass windows (the BLT masking ablations)
    + [dict(dataset="blt", sequence_id=f"gr_{d}_front", population="front_pass", kind="blt_front",
            bag=GR + b + ".bag", base=f"blt_gr_{d}")
       for d, b in (("20220323", "2022-03-23-12-27-15"), ("20220420", "2022-04-20-09-37-20"),
                    ("20220608", "2022-06-08-14-03-03"), ("20220713", "2022-07-13-15-38-32"),
                    ("20220915", "2022-09-15-14-23-20"))]
    + [dict(dataset="blt", sequence_id="uk_20230726_front", population="front_pass", kind="blt_front",
            bag="/Riseholme/session2.bag", base="blt_uk_20230726")]
    # 13 Jul and 15 Sep re-cut on header stamps: their bags record odometry seconds after the
    # images, so the bag-time cut above starts late and ends with the robot standing at the row end.
    # Same row as before (the rows are near-tied in length; on the corrected clock the rule alone
    # would pick another row), boundaries from the header clock.
    + [dict(dataset="blt", sequence_id=f"gr_{d}_hdr", population="gr_session", kind="blt",
            bag=GR + b + ".bag", clock="header", same_pass_as=f"blt_gr_{d}")
       for d, b in (("20220713", "2022-07-13-15-38-32"), ("20220915", "2022-09-15-14-23-20"))]
    + [dict(dataset="blt", sequence_id=f"gr_{d}_hdr_front", population="front_pass", kind="blt_front",
            bag=GR + b + ".bag", base=f"blt_gr_{d}_hdr")
       for d, b in (("20220713", "2022-07-13-15-38-32"), ("20220915", "2022-09-15-14-23-20"))]
)
for i, s in enumerate(SEQUENCES):
    s["index"] = i
    s["id"] = f"{s['dataset']}_{s['sequence_id']}"

if __name__ == "__main__":
    for s in SEQUENCES:
        print(s["index"], s["id"], s["population"], s["kind"])
