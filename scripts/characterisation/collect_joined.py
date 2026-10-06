"""Build results/joined.csv: the characterisation table joined to the quality table, one row
per sequence - every measured property next to every quality column. This is the source the
property-vs-outcome overlay figures are drawn from.

Join key: the sequence id (dataset_sequenceid). Quality rows without a characterisation row
(the composed-RGB and part-level sequences, whose properties equal their base/clip rows up
to the stated derivation) carry a `derived_from` pointer and inherit nothing silently.

usage: collect_joined.py
"""

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHAR = REPO / "results/characterisation"


def read(path, prefix=""):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = r.get("id") or f"{r['dataset']}_{r['sequence_id']}"
            rows[key] = {f"{prefix}{k}": v for k, v in r.items()}
    return rows


def derived_from(sid: str) -> str:
    if sid.endswith("_rgb"):
        return sid.removesuffix("_rgb")
    if "_p" in sid and sid.startswith("btg_"):
        return sid.rsplit("_p", 1)[0]
    return ""


def main():
    seq = read(CHAR / "sequences.csv", "char_")
    quality = read(REPO / "results/quality/quality.csv")
    fq = read(CHAR / "frame_quality.csv", "fq_")
    reg = read(CHAR / "registration.csv", "char_reg_")

    ids = sorted(set(seq) | set(quality))
    rows = []
    for sid in ids:
        r = {"id": sid, "derived_from": derived_from(sid)}
        base = r["derived_from"] or sid
        for src, own_key in ((seq, sid), (fq, sid), (reg, sid)):
            r.update(src.get(own_key, {}))
        if sid not in seq and base in seq:
            # properties of the base sequence, marked, for derived rows
            r.update({k: v for k, v in seq[base].items()})
            r.update({k: v for k, v in fq.get(base, {}).items()})
            r["properties_from"] = base
        r.update(quality.get(sid, {}))
        rows.append(r)
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    out = REPO / "results/joined.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    n_q = sum(1 for r in rows if r.get("interp_psnr_mean"))
    print(f"{len(rows)} rows ({n_q} with quality metrics) -> {out}")


if __name__ == "__main__":
    main()
