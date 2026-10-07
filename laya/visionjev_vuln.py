"""Vulnerability decision sets (code), license-clean:

* NIST Juliet C/C++ 1.3 (public domain; CC0 for foreign rights) -> ``juliet_cwe``, ``juliet_pair``
* PrimeVul paired (MIT) + NVD CVSS (US Government, public domain) -> ``primevul_cwe``, ``primevul_pair``,
  ``primevul_severity``; CWE names from MITRE's CWE list (free for commercial use under the CWE terms of use).

Questions:
* ``*_cwe`` (choice, 4-6): which weakness does this function contain? Distractors are real CWE names, about half
  "hard" (sharing words with the gold's name, e.g. two buffer bugs).
* ``*_pair`` (choice, 2): which of two functions contains the vulnerability? Juliet: a bad vs a good variant of the
  same test case; PrimeVul: the vulnerable function vs its own fix from the same commit.
* ``primevul_severity`` (score, 4): NVD CVSS base score (v3.1, else v3.0, else v2) -> low <4 / medium <7 / high <9 /
  critical.

Giveaways are removed before anything is written: comments are stripped and every identifier containing ``bad``,
``good`` or ``CWE`` (Juliet's naming) is renamed to a neutral one; the shortcut audit re-checks. Juliet test cases
come in near-identical flow variants (``_01`` .. ``_18``), so its split is by test-case group (never the same group
in train and val) and at most 2 variants per group are used.

    python -m laya.visionjev_vuln all
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import random
import re
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List

from .visionjev_data import DATA_ROOT, RAW_ROOT

JULIET_URL = "https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip"
CWE_URL = "https://cwe.mitre.org/data/csv/1000.csv.zip"
NVD_URL = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-%d.json.gz"
PRIMEVUL = "colin/PrimeVul"
MAX_CHARS = 2500
RAW = os.path.join(RAW_ROOT, "vuln")
LIC = {
    "juliet": {"annotations": "Public domain (NIST SARD Juliet 1.3; CC0 1.0 for foreign rights)", "images": "none (code)",
               "commercial": True, "verified": "samate.nist.gov/SARD/test-suites/112, 2026-09-25"},
    "primevul": {"annotations": "MIT (PrimeVul: GitHub DLVulDet/PrimeVul, HF colin/PrimeVul); CVSS from NVD (public "
                                "domain); CWE names MITRE (CWE terms of use)", "images": "none (code; snippets from "
                 "open-source projects under their own licenses)", "commercial": True,
                 "verified": "GitHub/HF license fields 2026-09-25"},
}


def _get(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (VisionJev data prep)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.rename(path + ".part", path)
    return path


def cwe_names() -> Dict[int, str]:
    import csv

    z = zipfile.ZipFile(_get(CWE_URL, os.path.join(RAW, "cwe_1000.csv.zip")))
    rows = csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8"))
    return {int(r["CWE-ID"]): r["Name"] for r in rows}


# ---- code sanitising ------------------------------------------------------------------------------------------

def strip_comments(code: str) -> str:
    out, i, n = [], 0, len(code)
    while i < n:
        if code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif code.startswith("//", i):
            j = code.find("\n", i)
            i = n if j < 0 else j
        elif code[i] in "\"'":
            q, j = code[i], i + 1
            while j < n and code[j] != q:
                j += 2 if code[j] == "\\" else 1
            out.append(code[i:j + 1])
            i = j + 1
        else:
            out.append(code[i])
            i += 1
    return re.sub(r"\n\s*\n+", "\n", "".join(out)).strip()


def neutralise(code: str) -> str:
    """Rename identifiers that give the answer away (Juliet: *bad*, *good*, CWE*). Each name maps to the SAME neutral
    name wherever it appears (hash-based), so the vulnerable and the fixed function of a pair rename identically; an
    index-based scheme leaked the label (the vulnerable function's own name took 'fn1')."""
    for name in {m for m in re.findall(r"\b[A-Za-z_]\w*\b", code) if re.search(r"bad|good|cwe", m, re.I)}:
        code = re.sub(r"\b%s\b" % re.escape(name), "v_" + hashlib.md5(name.encode()).hexdigest()[:6], code)
    return code


def as_process(func: str) -> str:
    """Replace the function's own name with 'process' BEFORE renaming anything else, and drop Juliet's structural
    tells: 'static' (good helpers are static, the bad function is not) and the empty ';' statements left where a
    '/* FLAW: ... */' comment was stripped."""
    func = re.sub(r"^([^(]*?)(\w+)\s*\(", r"\1process(", func, count=1)
    func = re.sub(r"^\s*static\s+", "", func)
    return re.sub(r"\n[ \t]*;[ \t]*(?=\n)", "", func.replace("\r\n", "\n"))


def extract_function(src: str, name_regex: str):
    m = re.search(r"(^|\n)([^\n;{}]*\b(%s)\s*\([^;{]*\)\s*\{)" % name_regex, src)
    if not m:
        return None
    start = m.start(2)
    depth, i = 0, src.index("{", m.start(2))
    while i < len(src):
        depth += src[i] == "{"
        depth -= src[i] == "}"
        if depth == 0:
            return src[start:i + 1]
        i += 1
    return None


# ---- sources --------------------------------------------------------------------------------------------------

def juliet_cases(rng):
    """(group, cwe_id, bad_code, good_code) per single-file test case, <= 2 flow variants per group."""
    z = zipfile.ZipFile(_get(JULIET_URL, os.path.join(RAW, "juliet_c_cpp_1.3.zip")))
    per_group = defaultdict(list)
    for name in z.namelist():
        m = re.search(r"/testcases/CWE(\d+)_[^/]*/(?:s\d+/)?(CWE\d+_\w+?)_(\d\d)\.(c|cpp)$", name)
        # flow variant 01 only: variants 02-18 wrap the same bug in control flow whose good versions carry tells
        # (e.g. printLine("Benign, fixed string") branches that never appear in the bad version)
        if m and m.group(3) == "01":
            per_group[m.group(2)].append((int(m.group(1)), name))
    for group, items in sorted(per_group.items()):
        rng.shuffle(items)
        for cwe, name in items[:2]:
            src = z.read(name).decode("latin-1").replace("\r\n", "\n")  # Juliet ships CRLF line endings
            bad = extract_function(src, r"\w*_bad")
            good = extract_function(src, r"good(?:G2B|B2G|1|2)\w*")
            if not bad or not good:
                continue
            bad = neutralise(as_process(strip_comments(bad)))
            good = neutralise(as_process(strip_comments(good)))
            if len(bad) <= MAX_CHARS and len(good) <= MAX_CHARS and bad != good:
                yield group, cwe, bad, good


def primevul_rows():
    from huggingface_hub import hf_hub_download

    for split, fname in (("train", "primevul_train_paired.jsonl"), ("val", "primevul_valid_paired.jsonl")):
        path = hf_hub_download(PRIMEVUL, fname, repo_type="dataset", cache_dir=os.path.join(RAW, "hf"))
        rows = [json.loads(l) for l in open(path)]
        groups = defaultdict(list)
        for r in rows:
            fn = re.search(r"\b(\w+)\s*\(", r["func"][:300])
            groups[(r["commit_id"], r["file_name"], fn.group(1) if fn else "")].append(r)
        for key, rs in groups.items():
            vul = [r for r in rs if int(r["target"]) == 1]
            fix = [r for r in rs if int(r["target"]) == 0]
            if len(vul) == 1 and len(fix) == 1:
                yield split, vul[0], fix[0]


def cvss_for(cves: List[str]) -> Dict[str, float]:
    years = sorted({int(c.split("-")[1]) for c in cves if re.match(r"CVE-\d{4}-\d+$", c)})
    need, out = set(cves), {}
    for y in years:
        path = _get(NVD_URL % y, os.path.join(RAW, "nvd", "nvdcve-2.0-%d.json.gz" % y))
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for v in data.get("vulnerabilities", []):
            c = v["cve"]
            if c["id"] not in need:
                continue
            m = c.get("metrics", {})
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if m.get(key):
                    out[c["id"]] = float(m[key][0]["cvssData"]["baseScore"])
                    break
    return out


# ---- question builders ----------------------------------------------------------------------------------------

def _words(s):
    return set(re.findall(r"[a-z]+", s.lower())) - {"of", "or", "the", "in", "to", "a", "an", "and", "improper", "use"}


def cwe_choice(gold: str, space: List[str], rng, kmin=4, kmax=6):
    k = min(len(space), rng.randint(kmin, kmax))
    others = [s for s in space if s != gold]
    hard = [s for s in sorted(others, key=lambda s: -len(_words(s) & _words(gold))) if _words(s) & _words(gold)][: 3 * k]
    hard = rng.sample(hard, min(len(hard), (k - 1) // 2 + 1))
    opts = [gold] + hard + rng.sample([s for s in others if s not in hard], k - 1 - len(hard))
    rng.shuffle(opts)
    return {"type": "choice", "instructions": rng.choice(("Which weakness does this function contain?",
                                                            "Classify the security weakness in this code.")),
            "criteria": opts}, opts.index(gold)


def length_balanced(pairs, rng):
    """Keep as many (vulnerable, fixed) pairs with the vulnerable one longer as shorter, so length alone says nothing
    (Juliet: the vulnerable function was longer 69% of the time)."""
    longer = [p for p in pairs if len(p[0]) > len(p[1])]
    shorter = [p for p in pairs if len(p[0]) < len(p[1])]
    n = min(len(longer), len(shorter))
    rng.shuffle(longer)
    rng.shuffle(shorter)
    out = longer[:n] + shorter[:n]
    rng.shuffle(out)
    return out


def pair_choice(vul: str, fix: str, rng):
    first = rng.random() < 0.5
    a, b = (vul, fix) if first else (fix, vul)
    return (json.dumps({"function_A": a, "function_B": b}),
            {"type": "choice", "instructions": rng.choice(("Which of these two functions contains a security vulnerability?",
                                                           "One of these functions is vulnerable. Which one?")),
             "criteria": ["function A", "function B"]}, 0 if first else 1)


SEVERITY = ["low: CVSS below 4.0", "medium: CVSS 4.0-6.9", "high: CVSS 7.0-8.9", "critical: CVSS 9.0 or higher"]


def _level(score):
    return 0 if score < 4 else 1 if score < 7 else 2 if score < 9 else 3


def write(name, rows: Dict[str, list], root, lic, meta, log=print):
    tmp, final = os.path.join(root, ".tmp-" + name), os.path.join(root, name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    labels = {}
    for split, rs in rows.items():
        with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
            for i, (state, q, lab) in enumerate(rs):
                f.write(json.dumps({"id": "%s-%s-%d" % (name, split, i), "state_text": state, "question": q, "label": lab},
                                   ensure_ascii=False) + "\n")
        labels[split] = dict(Counter(lab for _, _, lab in rs))
    m = dict({"dataset": name, "source": meta.pop("source"), "revision": meta.pop("revision", None), "license": lic,
              "counts": {s: len(r) for s, r in rows.items()}, "label_counts": labels,
              "validator": {"noul_min_minority": 0.05}}, **meta)
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    if os.path.exists(final):
        old = os.path.join(root, ".old-%s-%d" % (name, int(time.time())))
        os.rename(final, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    log("%s: %s labels %s" % (name, m["counts"], {s: dict(sorted(v.items())[:8]) for s, v in labels.items()}))


def build_juliet(root=DATA_ROOT, train_cap=8000, val_cap=1000, per_cwe=150, log=print):
    rng = random.Random(0)
    names = cwe_names()
    cases = list(juliet_cases(rng))
    by_cwe = defaultdict(list)
    for c in cases:
        by_cwe[c[1]].append(c)
    kept = []
    for cwe, cs in by_cwe.items():
        rng.shuffle(cs)
        kept += cs[:per_cwe]  # balance the CWE distribution
    space = sorted({"CWE-%d: %s" % (c, names.get(c, "")) for c in by_cwe if names.get(c)})
    rows_cwe, rows_pair = {"train": [], "val": []}, {"train": [], "val": []}
    pair_src = {"train": [], "val": []}
    for group, cwe, bad, good in kept:
        if not names.get(cwe):
            continue
        split = "val" if int(hashlib.md5(group.encode()).hexdigest(), 16) % 10 == 0 else "train"
        q, lab = cwe_choice("CWE-%d: %s" % (cwe, names[cwe]), space, rng)
        rows_cwe[split].append((json.dumps({"function": bad}), q, lab))
        pair_src[split].append((bad, good))
    for split in pair_src:
        rows_pair[split] = [pair_choice(v, f, rng) for v, f in length_balanced(pair_src[split], rng)]
    for rows in (rows_cwe, rows_pair):
        for s, cap in (("train", train_cap), ("val", val_cap)):
            rng.shuffle(rows[s])
            rows[s] = rows[s][:cap]
    meta = {"source": "NIST SARD Juliet C/C++ 1.3", "revision": "2017-10-01 v1.3", "cases": len(cases),
            "cwes": len(by_cwe), "split": "by test-case group (md5 10% val), <= 2 flow variants per group"}
    write("juliet_cwe", rows_cwe, root, LIC["juliet"], dict(meta), log)
    write("juliet_pair", rows_pair, root, LIC["juliet"], dict(meta), log)


def build_primevul(root=DATA_ROOT, min_per_cwe=20, log=print):
    rng = random.Random(1)
    names = cwe_names()
    pairs = [(s, v, f) for s, v, f in primevul_rows()
             if len(v["func"]) <= MAX_CHARS and len(f["func"]) <= MAX_CHARS]
    clean = lambda code: strip_comments(code)  # noqa: E731
    pair_src = {"train": [], "val": []}
    for split, v, f in pairs:
        a, b = clean(v["func"]), clean(f["func"])
        if a != b:
            pair_src[split].append((a, b))
    rows_pair = {split: [pair_choice(a, b, rng) for a, b in length_balanced(ps, rng)] for split, ps in pair_src.items()}

    def cwe_id(v):
        c = v.get("cwe") or []
        c = c if isinstance(c, list) else [c]
        ids = [int(x.split("-")[1]) for x in c if re.match(r"CWE-\d+$", str(x))]
        return ids[0] if len(ids) == 1 else None

    counts = Counter(cwe_id(v) for _, v, _ in pairs if cwe_id(v) and names.get(cwe_id(v)))
    frequent = {c for c, n in counts.items() if n >= min_per_cwe}
    space = sorted("CWE-%d: %s" % (c, names[c]) for c in frequent)
    rows_cwe = {"train": [], "val": []}
    for split, v, _ in pairs:
        c = cwe_id(v)
        if c in frequent:
            q, lab = cwe_choice("CWE-%d: %s" % (c, names[c]), space, rng)
            rows_cwe[split].append((json.dumps({"function": clean(v["func"])}), q, lab))

    cves = sorted({v["cve"] for _, v, _ in pairs if v.get("cve")})
    scores = cvss_for(cves)
    rows_sev = {"train": [], "val": []}
    for split, v, _ in pairs:
        if v.get("cve") in scores:
            q = {"type": "score", "instructions": rng.choice(("How severe is the vulnerability in this function?",
                                                                "Rate the severity of this function's security flaw.")),
                 "criteria": list(SEVERITY)}
            rows_sev[split].append((json.dumps({"function": clean(v["func"])}), q, _level(scores[v["cve"]])))
    meta = {"source": "PrimeVul paired (%s)" % PRIMEVUL, "pairs": len(pairs), "cwe_classes": len(frequent),
            "cves_scored": len(scores), "split": "PrimeVul official train / valid (test left untouched)"}
    write("primevul_pair", rows_pair, root, LIC["primevul"], dict(meta), log)
    write("primevul_cwe", rows_cwe, root, LIC["primevul"], dict(meta), log)
    write("primevul_severity", rows_sev, root, LIC["primevul"], dict(meta, severity_levels=SEVERITY), log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="juliet, primevul or all")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (["juliet", "primevul"] if a.names == ["all"] else a.names):
        try:
            (build_juliet if n == "juliet" else build_primevul)(a.root, log=lambda m: print(m, flush=True))
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("%s: FAILED %r" % (n, e), flush=True)
