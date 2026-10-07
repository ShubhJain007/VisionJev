"""Remove train/val image leakage inside a prepared image dataset, using the data validator's own matching
(exact bytes, or a dHash within 8 bits confirmed by 64x64 thumbnail correlation >= 0.98).

Images are grouped into connected components of "same photo" pairs; then either

* ``drop``    (default): val rows whose group contains any train image are removed;
* ``resplit``: every group goes wholly to train or val (md5 of the group's first file, ``val_pct``), keeping all
               rows — for sources whose official splits reuse images (e.g. VQA-RAD).

The manifest records what was done.

    python -m laya.visionjev_dedupe vizwiz_answerable cord_total pets
    python -m laya.visionjev_dedupe vqa_rad --mode resplit --val-pct 20
    python -m laya.visionjev_dedupe textvqa --against aokvqa vqav2   # also drop val photos in other sets' train
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

from .visionjev_data import DATA_ROOT

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import data_validator as dv  # noqa: E402


def _signature(path):
    from PIL import Image

    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with Image.open(path) as im:
        return sha, dv.dhash(im), dv.thumbnail(im)


def groups(paths):
    parent = {p: p for p in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    sig = {p: _signature(p) for p in paths}
    by_sha, by_dh = defaultdict(list), defaultdict(list)
    for p, (sha, dh, _) in sig.items():
        by_sha[sha].append(p)
        if dh is not None:
            by_dh[dh].append(p)
    for ps in by_sha.values():
        for q in ps[1:]:
            union(ps[0], q)
    keys = list(by_dh)
    for ha, hb in dv._near_pairs({k: 0 for k in keys}, {k: 0 for k in keys}, dv.DHASH_MAX_DISTANCE):
        for a in by_dh[ha]:
            for b in by_dh[hb]:
                if a != b and find(a) != find(b) and dv.correlation(sig[a][2], sig[b][2]) >= dv.MIN_THUMB_CORRELATION:
                    union(a, b)
    return {p: find(p) for p in paths}


def _train_images(name, root):
    base = os.path.join(root, name)
    return {os.path.normpath(os.path.join(base, r["image"])) for r in map(json.loads, open(os.path.join(base, "train.jsonl")))
            if r.get("image")}


def dedupe(name, root=DATA_ROOT, mode="drop", val_pct=20, against=(), log=print):
    """``against``: other datasets trained on alongside this one; in ``drop`` mode a val row is also removed when
    its photo is in one of their train splits (the joint validator checks leakage across the whole mix)."""
    base = os.path.join(root, name)
    rows = {s: [json.loads(l) for l in open(os.path.join(base, s + ".jsonl"))] for s in ("train", "val")}
    path = lambda r: os.path.normpath(os.path.join(base, r["image"]))  # noqa: E731
    other = set().union(*(_train_images(o, root) for o in against)) if against else set()
    g = groups(sorted({path(r) for s in rows for r in rows[s]} | other))
    before = {s: len(rs) for s, rs in rows.items()}
    if mode == "drop":
        train_groups = {g[path(r)] for r in rows["train"]} | {g[p] for p in other}
        rows["val"] = [r for r in rows["val"] if g[path(r)] not in train_groups]
    else:
        pooled = rows["train"] + rows["val"]
        is_val = lambda r: int(hashlib.md5(g[path(r)].encode()).hexdigest(), 16) % 100 < val_pct  # noqa: E731
        rows = {"train": [r for r in pooled if not is_val(r)], "val": [r for r in pooled if is_val(r)]}
    for s, rs in rows.items():
        with open(os.path.join(base, s + ".jsonl"), "w") as f:
            f.writelines(json.dumps(dict(r, id="%s-%s-%d" % (name, s, i)), ensure_ascii=False) + "\n"
                         for i, r in enumerate(rs))
    mpath = os.path.join(base, "manifest.json")
    m = json.load(open(mpath))
    m.setdefault("dedupe", []).append({"mode": mode, "val_pct": val_pct if mode == "resplit" else None,
                                       "against_train_of": list(against),
                                       "before": before, "after": {s: len(rs) for s, rs in rows.items()},
                                       "match": "sha256 or dHash<=8 bits + 64px thumbnail corr>=0.98"})
    m["counts"] = {s: len(rs) for s, rs in rows.items()}
    json.dump(m, open(mpath, "w"), indent=2)
    log("%s (%s): %s -> %s" % (name, mode, before, m["counts"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--mode", choices=["drop", "resplit"], default="drop")
    ap.add_argument("--val-pct", type=int, default=20)
    ap.add_argument("--against", nargs="*", default=[], help="datasets whose train images this val must avoid")
    a = ap.parse_args()
    for n in a.names:
        dedupe(n, a.root, a.mode, a.val_pct, a.against)
