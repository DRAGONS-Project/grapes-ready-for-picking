"""Write results/RUNS.md: every CSV row traced to its run directory, SLURM job and date.

Commits are per-file: `git log --follow -- <path>` on branch `full-program` gives the commit
that introduced or last changed any artefact listed here.

usage: collect_runs.py
"""

import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def jread(p):
    return json.loads(p.read_text()) if p.exists() else {}


def mdate(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") \
        if p.exists() else ""


def main():
    lines = [
        "# Run provenance index",
        "",
        "Every table cell traces to a run directory below; each directory holds the",
        "registration row, `summary.json` (with the SLURM job id), per-view metrics and the",
        "kept renders. Commits: `git log --follow -- <path>` on branch `full-program`.",
        "Characterisation rows (sequences/frame_quality/registration.csv) come from the",
        "fixed-COLMAP arrays of 3 Oct (logs under `logs/characterise/`), per-sequence JSONs",
        "in `results/characterisation/per_sequence/`.",
        "",
        "## results/quality/quality.csv",
        "",
        "| id | split | run dir | job | date |",
        "| --- | --- | --- | --- | --- |",
    ]
    B = REPO / "results/full_program/B"
    for d in sorted(B.iterdir()):
        if not d.is_dir():
            continue
        for split in ("interp", "block"):
            s = jread(d / split / "summary.json")
            if s:
                lines.append(f"| {d.name} | {split} | {d.relative_to(REPO)}/{split}/ | "
                             f"{s.get('job', '')} | {mdate(d / split / 'summary.json')} |")
    lines += ["", "## results/ablations/*.csv", "",
              "| set | sequence | condition | seed | run dir | job | date |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    C = REPO / "results/full_program/C"
    for setdir in sorted(p for p in C.iterdir() if p.is_dir()):
        for seqdir in sorted(p for p in setdir.iterdir() if p.is_dir()):
            for cond in sorted(p for p in seqdir.iterdir() if p.is_dir()):
                seeds = sorted(cond.glob("seed*/summary.json"))
                if not seeds:
                    lines.append(f"| {setdir.name} | {seqdir.name} | {cond.name} | - | "
                                 f"{cond.relative_to(REPO)}/ | registration only | "
                                 f"{mdate(cond / 'registration.json')} |")
                for sf in seeds:
                    s = jread(sf)
                    lines.append(f"| {setdir.name} | {seqdir.name} | {cond.name} | "
                                 f"{s.get('seed')} | {sf.parent.relative_to(REPO)}/ | "
                                 f"{s.get('job', '')} | {mdate(sf)} |")
    lines += ["", "## results/sweeps/sweeps.csv", "",
              "| sequence | level | run dir | job | date |",
              "| --- | --- | --- | --- | --- |"]
    S = REPO / "results/sweeps"
    for seqdir in sorted(p for p in S.iterdir() if p.is_dir() and p.name != "_mask_cache"):
        for run in sorted(p for p in seqdir.iterdir() if p.is_dir()):
            s = jread(run / "summary.json")
            note = jread(run / "README.json")
            job = s.get("job", "") or ("= quality row" if note else "")
            when = mdate(run / "registration.json") or mdate(run / "README.json")
            lines.append(f"| {seqdir.name} | {run.name} | {run.relative_to(REPO)}/ | "
                         f"{job} | {when} |")
    lines += ["", "## results/repeats/*.csv (registration repeat study, job 6024964)", "",
              "| condition | run | COLMAP seed | run file | date |",
              "| --- | --- | --- | --- | --- |"]
    for rf in sorted((REPO / "results/repeats/runs").glob("*/run_*.json")):
        r = jread(rf)
        lines.append(f"| {r.get('condition')} | {r.get('run')} | {r.get('colmap_seed')} | "
                     f"{rf.relative_to(REPO)} | {mdate(rf)} |")
    lines += [
        "", "## Post-processing tables (no SLURM job; derived from kept artefacts)", "",
        "- `results/regions/mots_regions.csv` - from kept renders + the deposits' instance",
        "  maps (`scripts/characterisation/region_metric.py`); masks cached under",
        "  `results/regions/_mask_cache/`.",
        "- `results/anchors/geometric.csv` - from each run's poses CSV + its anchor",
        "  (`scripts/characterisation/anchors_eval.py`; RTK-video row from the shipped DJI",
        "  log). GPS caches: `results/anchors/_gps_cache/`.",
        "- `results/joined.csv` - `scripts/characterisation/collect_joined.py`.",
        "- `results/figures/` - `scripts/characterisation/figure_pack.py`, from kept seed-0",
        "  renders only.",
        "",
        "## Trainer-check and debug runs",
        "",
        "`results/debug/trainer_check/` (jobs 6015522-6015526, 6015641, 6015642, plus the",
        "superseded distorted debug run 2, job 6015392).",
    ]
    out = REPO / "results/RUNS.md"
    out.write_text("\n".join(lines) + "\n")
    n = sum(1 for ln in lines if ln.startswith("| ") and "---" not in ln)
    print(f"{n} table rows -> {out}")


if __name__ == "__main__":
    main()
