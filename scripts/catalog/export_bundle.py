"""Export the online catalog's data, text and figures from the manuscript source.

The catalog page (docs/) shows four tables of the paper and its decision-procedure figure. They are
converted here from the LaTeX source, never retyped: a LaTeX command the converter does not know
stops the export.

inputs (--src, the manuscript source as exported from Overleaf):
  text.tex, appendix.tex, mdpi_sensors.tex, bibliography.bib, data/quality/readiness_tree.pdf
outputs (--out, the site root):
  data/<table>.json                     datasets_found (Table 1), characterization_scene (C1),
                                        characterization_record (C2), readiness (7); JSON has
                                        columns, rows, caption, paper_number and the join keys the
                                        page needs
  text/abstract.txt, text/meta.json
  static/images/readiness_tree.{svg,png}
  index.html                            the regions between <!--@name--> and <!--/@name--> markers
                                        (title block, abstract, caption, footer) are refilled
usage: export_bundle.py --src DIR [--out docs] [--no-figures]
"""

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLES = [  # file name, LaTeX label
    ("datasets_found", "tab:dataset_overview"),
    ("characterization_scene", "tab:char_scene"),
    ("characterization_record", "tab:char_record"),
    ("readiness", "tab:readiness"),
]
# paper numbers stated in the catalog brief; the export stops if the source numbers differently
EXPECTED_NUMBERS = {"tab:dataset_overview": "1", "tab:char_scene": "C1", "tab:char_record": "C2",
                    "tab:readiness": "7"}
EXPECTED_ROWS = {"datasets_found": 18, "readiness": 10}
# join keys: Table 1 names to the column abbreviations of Tables C1 and C2
TABLE1_IDS = {"Bacchus (BLT)": "BLT", "Bodegas T. Gauda": "BTG", "Embrapa": "EMB", "EscaYard": "ESC",
              "GrapeCS-ML": "GCS", "GrapeSet": "GST", "MOTS UAV": "MOTS", "Multi. Botrytis": "BOT",
              "VineLiDAR": "VLD", "GrapeSLAM": "SLAM"}
CLASSES = ("ripe", "ripening", "unripe")
DATA_HOSTS = ("10.5281/zenodo", "zenodo.org", "kaggle.com", "10.34740/kaggle", "data.mendeley.com", "10.17632/",
              "researchoutput.csu.edu.au")
FIGURES = [  # id, source, label
    ("readiness_tree", "data/quality/readiness_tree.pdf", "fig:readiness_tree"),
]
REPO_URL = "https://github.com/DRAGONS-Project/grapes-ready-for-picking"
SITE_URL = "https://dragons-project.github.io/grapes-ready-for-picking/"
SUPERSCRIPTS = {"a": "\u1d43", "b": "\u1d47", "c": "\u1d9c", "\\dagger": "\u2020"}
ACCENTS = {"'": "\u0301", "`": "\u0300", "~": "\u0303", '"': "\u0308", "^": "\u0302", "c": "\u0327"}
UNWRAP = ("rev", "acc", "textbf", "textit", "emph", "mbox", "url")


# --- LaTeX -----------------------------------------------------------------------------------

def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in s.split("\n"))


def group_end(s, i):
    """Index just past the brace group that opens at s[i]."""
    assert s[i] == "{", s[i:i + 40]
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        j += 1
        if depth == 0:
            return j
    raise ValueError("unbalanced braces: " + s[i:i + 80])


def arg(s, cmd, start=0):
    """Content of the first argument of the first \\cmd{...} at or after start."""
    i = s.index("\\" + cmd + "{", start) + len(cmd) + 1
    return s[i + 1:group_end(s, i) - 1]


def replace_cmd(s, name, fn):
    pat = re.compile(r"\\" + name + r"\s*\{")
    while True:
        m = pat.search(s)
        if not m:
            return s
        b = m.end() - 1
        e = group_end(s, b)
        s = s[:m.start()] + fn(s[b + 1:e - 1]) + s[e:]


def math(m):
    inner = m.group(1).strip()
    if inner in ("\\times",):
        return "\u00d7"
    sup = re.fullmatch(r"\^\{([^}]*)\}", inner)
    if sup:
        parts = [p.strip() for p in sup.group(1).split(",")]
        if parts == ["\\circ"]:
            return "\u00b0"
        if all(p in SUPERSCRIPTS for p in parts):
            return "".join(SUPERSCRIPTS[p] for p in parts)
    raise ValueError("unknown math: $" + inner + "$")


def tex2text(s, refs, cites=None):
    """Plain text of a LaTeX fragment. Citation keys are appended to cites."""
    def cite(c):
        if cites is not None:
            cites.extend(k.strip() for k in c.split(","))
        return ""
    s = replace_cmd(s, "cite", cite)
    for name in UNWRAP:
        s = replace_cmd(s, name, lambda c: c)
    s = replace_cmd(s, "ref", lambda c: refs[c])
    s = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", s)
    s = re.sub(r"\\([" + re.escape("'`~\"^") + r"])\s*\{?([A-Za-z])\}?",
               lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENTS[m.group(1)]), s)
    s = re.sub(r"\\c\{([A-Za-z])\}", lambda m: unicodedata.normalize("NFC", m.group(1) + ACCENTS["c"]), s)
    s = re.sub(r"\{\\l\}|\\l\b", "\u0142", s)
    s = re.sub(r"\$([^$]*)\$", math, s)
    for a, b in (("\\,", " "), ("\\%", "%"), ("\\_", "_"), ("\\&", "&"), ("\\-", ""), ("\\quad", " "),
                 ("---", "\u2014"), ("--", "\u2013"), ("~", " "), ("``", "\u201c"), ("''", "\u201d")):
        s = s.replace(a, b)
    s = s.replace("{", "").replace("}", "")
    if "\\" in s:
        raise ValueError("unconverted LaTeX: " + s)
    return re.sub(r"\s+", " ", s).strip()


def numbering(text, appendix):
    """Label -> number as LaTeX prints it: sections, tables and figures, appendix lettered."""
    refs, sec, sub, tab, fig, env, last = {}, 0, 0, 0, 0, None, None
    tok = re.compile(r"\\(section|subsection)\s*\{|\\begin\{(table|sidewaystable|figure)\}|"
                     r"\\end\{(table|sidewaystable|figure)\}|\\label\{([^}]*)\}|\\setcounter\{(table|figure)\}\{0\}|"
                     r"\\(caption)\s*\{")
    for part, src in (("main", text), ("appendix", appendix)):
        if part == "appendix":
            sec = 0
        for m in tok.finditer(src):
            if m.group(1) == "section":
                sec, sub, last = sec + 1, 0, "section"
            elif m.group(1) == "subsection":
                sub, last = sub + 1, "subsection"
            elif m.group(2):
                env = "table" if m.group(2) != "figure" else "figure"
            elif m.group(6):  # a float may hold several captions, each numbered
                if env == "table":
                    tab += 1
                else:
                    fig += 1
            elif m.group(3):
                env = None
            elif m.group(5):
                tab, fig = (0, fig) if m.group(5) == "table" else (tab, 0)
            elif m.group(4):
                head = chr(64 + sec) if part == "appendix" else str(sec)
                if env == "table":
                    n = (head if part == "appendix" else "") + str(tab)
                elif env == "figure":
                    n = (head if part == "appendix" else "") + str(fig)
                else:
                    n = head if last == "section" else f"{head}.{sub}"
                refs[m.group(4)] = n
    return refs


def environment(src, label):
    i = src.index("\\label{" + label + "}")
    start = max(src.rfind("\\begin{" + e + "}", 0, i) for e in ("table", "sidewaystable", "figure"))
    end = src.index("\\end{", i)
    return src[start:end]


def parse_table(env):
    """Header cells, row groups (split at \\midrule) of cell lists, and the raw caption."""
    body = env[env.index("\\toprule") + len("\\toprule"):env.index("\\bottomrule")]
    parts = body.split("\\midrule")

    def rows(block):
        out = []
        for r in re.split(r"\\\\(?:\[[^\]]*\])?", block):
            r = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", r).strip()
            if r:
                out.append([c.strip() for c in re.split(r"(?<!\\)&", r)])
        return out
    header = rows(parts[0])
    assert len(header) == 1, header
    return header[0], [rows(p) for p in parts[1:]], arg(env, "caption")


# --- bibliography ----------------------------------------------------------------------------

def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        b = text.index("{", m.start())
        body = text[m.end():group_end(text, b) - 1]
        fields, i = {}, 0
        for f in re.finditer(r"(\w+)\s*=\s*", body):
            if f.start() < i:
                continue
            j = f.end()
            if body[j] == "{":
                e = group_end(body, j)
                val, i = body[j + 1:e - 1], e
            elif body[j] == '"':
                e = body.index('"', j + 1)
                val, i = body[j + 1:e], e + 1
            else:
                e = re.search(r"[,\n]|$", body[j:]).start() + j
                val, i = body[j:e], e
            fields[f.group(1).lower()] = re.sub(r"\s+", " ", val).strip()
        entries[m.group(2)] = dict(type=m.group(1).lower(), **fields)
    return entries


def link(entry, key):
    url = entry.get("url") or ""
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", entry.get("doi", ""))
    url = (url or ("https://doi.org/" + doi if doi else "")).replace("\\_", "_")
    if not url:
        return None  # the entry has neither; the row keeps its other links
    # by where the link points, not by entry type: some @misc entries carry the data paper's DOI
    kind = "data" if any(h in url.lower() for h in DATA_HOSTS) else "paper"
    return dict(key=key, kind=kind, url=url)


# --- outputs ---------------------------------------------------------------------------------

def write_table(out, name, number, caption, columns, rows, extra):
    data = dict(name=name, paper_number=number, caption=caption, columns=columns, rows=rows, **extra)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


def export_tables(src, out, refs, bib):
    full = src["text"] + src["appendix"]
    schema_head, schema_groups, _ = parse_table(environment(full, "tab:schema"))
    schema = [(tex2text(g[0][0], refs), len(g)) for g in schema_groups]
    # names of the ten columns, from the Table C1 caption (Table C2 refers to it)
    names = column_names(tex2text(parse_table(environment(full, "tab:char_scene"))[2], refs))
    counts = {}
    for name, label in TABLES:
        number = refs[label]
        if EXPECTED_NUMBERS[label] != number:
            raise ValueError(f"{label} is Table {number} in the source, the brief says {EXPECTED_NUMBERS[label]}")
        head, groups, cap = parse_table(environment(full, label))
        columns = [tex2text(c, refs) or "#" for c in head]
        caption = tex2text(cap, refs)
        rows, links = [], []
        for g in groups:
            for r in g:
                if len(r) != len(columns):
                    raise ValueError(f"{label}: {len(r)} cells, {len(columns)} columns: {r}")
                keys = []
                rows.append([tex2text(c, refs, keys) for c in r])
                links.append([x for x in (link(bib[k], k) for k in keys) if x])
        extra = {}
        if name == "datasets_found":
            extra["row_links"] = links
            # the footnotes of the caption, each starting at its superscript marker
            marks = "".join(SUPERSCRIPTS[k] for k in "abc")
            extra["notes"] = re.findall("[%s][^%s]*" % (marks, marks), caption[re.search("[%s][A-Z]" % marks, caption).start():])
            extra["notes"] = [n.strip() for n in extra["notes"]]
            extra["row_ids"] = [TABLE1_IDS.get(r[0]) for r in rows]
            yes = {r[0] for r in rows if r[columns.index("Characterized")] == "yes"}
            if yes != set(TABLE1_IDS):
                raise ValueError(f"characterized rows {sorted(yes)} differ from the join keys")
        elif name.startswith("characterization"):
            sel = schema[:2] if name.endswith("scene") else schema[2:]
            if [len(g) for g in groups] != [n for _, n in sel]:
                raise ValueError(f"{label}: groups {[len(g) for g in groups]} differ from Table 3 {sel}")
            extra["row_groups"] = [q for q, n in sel for _ in range(n)]
            extra["column_names"] = names
            if list(extra["column_names"]) != columns[1:]:
                raise ValueError(f"{label}: caption names {list(extra['column_names'])} vs {columns[1:]}")
        elif name == "readiness":
            ids = {v: k for k, v in names.items()}
            extra["row_ids"] = [ids[r[0]] for r in rows]
            extra["row_classes"] = [re.search(r"\((%s)\)" % "|".join(CLASSES), r[1]).group(1) for r in rows]
        if name in EXPECTED_ROWS and len(rows) != EXPECTED_ROWS[name]:
            raise ValueError(f"{name}: {len(rows)} rows, expected {EXPECTED_ROWS[name]}")
        write_table(out, name, ("Table " + number), caption, columns, rows, extra)
        counts[name] = len(rows)
    return counts


def column_names(caption):
    """{abbreviation: name} from 'The columns are Name (ABB), ... and Name (ABB).'"""
    tail = caption[caption.index("The columns are ") + len("The columns are "):].rstrip(".")
    out = {}
    for item in re.split(r",\s*(?:and\s+)?|\s+and\s+", tail):
        m = re.fullmatch(r"(.+?) \(([A-Z]+)\)", item.strip())
        out[m.group(2)] = m.group(1)
    return out


def export_text(src, out, refs):
    tex = src["main"]
    orcid = dict(re.findall(r"\\newcommand\{\\orcidauthor(\w+)\}\{([^}]*)\}", tex))
    authors = []
    for a in re.split(r",\s*\n|\s+and\s+", arg(tex, "Author").strip()):
        m = re.fullmatch(r"(.+?)\s*\$\^\{([^}]*)\}\$\\orcidauthor(\w+)\{\}", a.strip())
        marks = [p.strip() for p in m.group(2).split(",")]
        authors.append(dict(name=m.group(1), affiliations=[int(p) for p in marks if p.isdigit()],
                            equal_contribution="\\dagger" in marks, orcid=orcid[m.group(3)]))
    address = []
    for line in arg(tex, "address").split("\\\\"):
        m = re.match(r"\s*\$\^\{([^}]*)\}\$\s*\\quad\s*(.+)", line.strip())
        if m:
            address.append((m.group(1), tex2text(m.group(2), refs)))
    affiliations = [dict(id=int(k), name=v) for k, v in address if k.isdigit()]
    equal = [v for k, v in address if k == "\\dagger"][0]
    email = re.search(r"[\w.+-]+@[\w.-]+", arg(tex, "corres")).group(0)
    funding = tex2text(arg(tex, "funding"), refs)
    i = funding.index(" We gratefully acknowledge")
    old = {}
    if (out / "text/meta.json").exists():
        old = json.loads((out / "text/meta.json").read_text()).get("links", {})
    first = authors[0]
    meta = dict(
        title=tex2text(arg(tex, "Title"), refs), short_title="Grapes Ready for Picking",
        authors=authors, affiliations=affiliations, equal_contribution_note=equal,
        contact=dict(name=first["name"], role="corresponding author", email=email,
                     affiliation=affiliations[first["affiliations"][0] - 1]["name"]),
        keywords=[k.strip() for k in tex2text(arg(tex, "keyword"), refs).split(",")],
        links=dict(paper=old.get("paper"), archive_doi=old.get("archive_doi"), code=REPO_URL, catalog=SITE_URL),
        funding=funding[:i], acknowledgement=funding[i + 1:],
    )
    abstract = tex2text(arg(tex, "abstract"), refs)
    (out / "text").mkdir(parents=True, exist_ok=True)
    (out / "text/meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    (out / "text/abstract.txt").write_text(abstract + "\n")
    return meta, abstract


def export_figures(src_dir, out, refs, text, appendix, draw=True):
    img = out / "static/images"
    img.mkdir(parents=True, exist_ok=True)
    full = text + appendix
    figs = {}
    for fid, path, label in FIGURES:
        figs[fid] = dict(paper_number="Figure " + refs[label],
                         caption=tex2text(arg(environment(full, label), "caption"), refs))
        if draw:
            import fitz
            page = fitz.open(src_dir / path)[0]
            (img / f"{fid}.svg").write_text(page.get_svg_image(text_as_path=True))
            page.get_pixmap(matrix=fitz.Matrix(4, 4)).save(img / f"{fid}.png")
    return figs


def first_sentence(s):
    m = re.search(r"(?<!\b[A-Z])\.\s+(?=[A-Z(])", s)
    return s[:m.start() + 1] if m else s


def fill_index(out, meta, abstract, figs):
    path = out / "index.html"
    if not path.exists():
        return
    e = html.escape
    n = len(meta["authors"])
    authors = "\n".join(
        f'<span class="author-block">{e(a["name"])}<sup>{",".join(map(str, a["affiliations"]))}'
        f'{",&dagger;" if a["equal_contribution"] else ""}</sup>{"," if i < n - 1 else ""}</span>'
        for i, a in enumerate(meta["authors"]))
    affils = "\n".join(f'<span class="author-block"><sup>{a["id"]}</sup>{e(a["name"])}</span>'
                       for a in meta["affiliations"])
    affils += f'\n<span class="author-block"><sup>&dagger;</sup>{e(meta["equal_contribution_note"])}</span>'
    regions = dict(title=e(meta["title"]), authors=authors, affiliations=affils,
                   abstract=e(abstract),
                   funding=e(meta["funding"]), acknowledgement=e(meta["acknowledgement"]),
                   contact_name=e(meta["contact"]["name"]), contact_email=e(meta["contact"]["email"]),
                   contact_affiliation=e(meta["contact"]["affiliation"]))
    for fid, f in figs.items():
        regions[f"cap:{fid}"] = e(first_sentence(f["caption"]))
        regions[f"num:{fid}"] = e(f["paper_number"])
    doc = path.read_text()
    for k, v in regions.items():
        doc = re.sub(r"(<!--@%s-->).*?(<!--/@%s-->)" % (re.escape(k), re.escape(k)),
                     lambda m: m.group(1) + v + m.group(2), doc, flags=re.S)
    left = sorted(set(re.findall(r"<!--@([\w:]+)-->", doc)) - set(regions))
    if left:
        raise ValueError("index.html regions without content: " + ", ".join(left))
    path.write_text(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "docs")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    src = {k: strip_comments((a.src / f).read_text(encoding="utf-8"))
           for k, f in (("text", "text.tex"), ("appendix", "appendix.tex"), ("main", "mdpi_sensors.tex"))}
    refs = numbering(src["text"], src["appendix"])
    bib = parse_bib((a.src / "bibliography.bib").read_text(encoding="utf-8"))
    counts = export_tables(src, a.out, refs, bib)
    meta, abstract = export_text(src, a.out, refs)
    figs = export_figures(a.src, a.out, refs, src["text"], src["appendix"], draw=not a.no_figures)
    fill_index(a.out, meta, abstract, figs)
    print("rows:", counts)
    print("figures:", {k: v["paper_number"] for k, v in figs.items()})


if __name__ == "__main__":
    main()
