"""Localisation and spatial decisions from Open Images human boxes (annotations CC BY 4.0; images CC BY 2.0, each
record keeps the image's attribution). Uses the photos ``visionjev_openimages --subset test`` already downloaded
(``oi_train_images``) and the same per-image train/val split (md5 % 100 < 5 -> val), so no photo crosses splits.

Only exact, box-derived labels; only single-instance objects (exactly one box of that class, not group-of, not a
depiction), on images with no rotation, so "the <object>" is unambiguous:

* ``oi_where``    (choice, all 9 grid cells) "Where in the image is the <object>?" The box centre must be at least
                  ``MARGIN`` from a cell boundary; gold cells are capped per cell, so the centre prior is not a shortcut.
* ``oi_relation`` (noul) "Is the <A> to the left of / above / larger than the <B>?" Only pairs with a clear gap
                  (or an area ratio >= 2.5); each true statement is asked either as stated or as its converse (false),
                  so the answer is balanced and no wording predicts it.
* ``oi_framing``  (choice, two photos) "Which photo shows the whole <object>?" Two equal-size crops of the same
                  photo: one contains the whole (untruncated, unoccluded) object, the other's edge cuts it through its centre,
                  so image size and aspect cannot give the answer away. Crops are adaptations (CC BY 2.0 allows them;
                  attribution kept and marked as modified).

    python -m laya.visionjev_oiloc                       # oi_where / oi_relation / oi_framing (train + val)
    python -m laya.visionjev_oiloc --subset validation   # openimages_* zero-shot eval sets (other photos)
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict

from .visionjev_data import DATA_ROOT
from .visionjev_openimages import CC_BY, LICENSE, RAW, UNCOUNTABLE

#: subset -> (image folder, metadata csv, boxes csv, dataset-name prefix, zero-shot?)
SUBSETS = {"test": ("oi_train_images", "test-images-with-rotation.csv", "test-annotations-bbox.csv", "oi", False),
           "validation": ("openimages_images", "validation-images-with-rotation.csv",
                          "validation-annotations-bbox.csv", "openimages", True)}
CELLS = [["top left", "top center", "top right"], ["middle left", "center", "middle right"],
         ["bottom left", "bottom center", "bottom right"]]
MARGIN = 0.04
WHERE_Q = ("Where in the image is {x}?", "In which part of the picture is {x}?", "Where is {x} located in this image?")
REL = {"left": ("Is {a} to the left of {b}?", "Is {a} on the left side of {b} in this image?"),
       "above": ("Is {a} above {b}?", "Is {a} higher up in the image than {b}?"),
       "larger": ("Does {a} take up more of the image than {b}?", "Is {a} larger in the picture than {b}?")}
FRAME_Q = ("Which photo shows the whole of {x}?", "In which photo is {x} fully visible?")


def _the(name: str) -> str:
    return "the " + name.lower()


def _split(image_id: str, zero_shot: bool) -> str:
    """Zero-shot subsets are all val; otherwise the per-photo split of ``visionjev_openimages`` (5% val)."""
    if zero_shot:
        return "val"
    return "val" if int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 100 < 5 else "train"


def _load(root, subset):
    """(meta, singles): per-image metadata (with ``_src`` folder and ``_split``) and single-instance objects."""
    src, meta_csv, box_csv, _, zero_shot = SUBSETS[subset]
    with open(os.path.join(RAW, "class-descriptions-boxable.csv"), newline="", encoding="utf-8") as f:
        names = {row[0]: row[1] for row in csv.reader(f)}
    have = {f[:-4] for f in os.listdir(os.path.join(root, src, "images"))}
    meta = {}
    with open(os.path.join(RAW, meta_csv), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ImageID"] in have and r["License"] == CC_BY and r["Rotation"] in ("", "0", "0.0"):
                meta[r["ImageID"]] = dict(r, _src=src, _split=_split(r["ImageID"], zero_shot))
    boxes = defaultdict(list)
    with open(os.path.join(RAW, box_csv), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ImageID"] in meta:
                boxes[r["ImageID"]].append(r)
    singles = {}
    for i, bs in boxes.items():
        per = Counter(b["LabelName"] for b in bs)
        objs = []
        for b in bs:
            # body parts ("Human hair", "Human hand") are not objects anyone asks about by name
            if per[b["LabelName"]] == 1 and b["IsGroupOf"] == "0" and b["IsDepiction"] == "0" and b["LabelName"] in names \
                    and not names[b["LabelName"]].startswith("Human "):
                x0, x1, y0, y1 = (float(b[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
                objs.append({"name": names[b["LabelName"]], "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                             "area": (x1 - x0) * (y1 - y0), "truncated": b["IsTruncated"] == "1",
                             "occluded": b["IsOccluded"] == "1"})
        # two classes with the same display name would make "the <name>" ambiguous
        dup = {n for n, c in Counter(o["name"] for o in objs).items() if c > 1}
        singles[i] = [o for o in objs if o["name"] not in dup]
    return meta, singles


def _base(meta, image_id, image=True):
    m = meta[image_id]
    rec = {"state_text": None, "image_license": "CC BY 2.0",
           "attribution": {"author": m["Author"], "page": m["OriginalLandingURL"], "title": m["Title"]}}
    if image:
        rec["image"] = "../%s/images/%s.jpg" % (m["_src"], image_id)
    return rec


def where_rows(meta, singles, rng):
    cand = []
    for i, objs in singles.items():
        for o in objs:
            if not 0.005 <= o["area"] <= 0.35:
                continue
            cx, cy = (o["x0"] + o["x1"]) / 2, (o["y0"] + o["y1"]) / 2
            if any(abs(c - t) < MARGIN for c in (cx, cy) for t in (1 / 3, 2 / 3)):
                continue
            cand.append((i, o, CELLS[min(2, int(cy * 3))][min(2, int(cx * 3))]))
    rng.shuffle(cand)
    cap = sorted(Counter(c for _, _, c in cand).values())[4] if cand else 0  # the median cell count
    used, per_img, rows = Counter(), Counter(), []
    for i, o, cell in cand:
        if used[cell] >= cap or per_img[i] >= 2:
            continue
        used[cell] += 1
        per_img[i] += 1
        # all nine cells, shuffled: with a sampled subset the gold was the longest name half the time (bottom and
        # middle cells are both frequent and long), which "pick the longest option" exploited
        opts = [c for row in CELLS for c in row]
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": rng.choice(WHERE_Q).format(x=_the(o["name"])), "criteria": opts}
        rows.append((meta[i]["_split"], dict(_base(meta, i), question=q, label=opts.index(cell))))
    return rows


def relation_rows(meta, singles, rng):
    rows = []
    for i, objs in singles.items():
        facts = []
        for a in objs:
            for b in objs:
                if a is b:
                    continue
                if a["x1"] + 0.02 < b["x0"]:
                    facts.append(("left", a, b))
                if a["y1"] + 0.02 < b["y0"]:
                    facts.append(("above", a, b))
                if a["area"] >= 2.5 * b["area"] and b["area"] >= 0.002:
                    facts.append(("larger", a, b))
        rng.shuffle(facts)
        for rel, a, b in facts[:3]:
            truth = rng.random() < 0.5
            x, y = (a, b) if truth else (b, a)
            q = {"type": "noul", "instructions": rng.choice(REL[rel]).format(a=_the(x["name"]), b=_the(y["name"])),
                 "criteria": None}
            rows.append((meta[i]["_split"], dict(_base(meta, i), question=q, label=int(truth))))
    return rows


def _crops(o, rng):
    """(whole, cut) crop boxes in normalised coords, same size; None if the object cannot be cut cleanly."""
    for axis in rng.sample(("x", "y"), 2):
        lo, hi = o[axis + "0"], o[axis + "1"]
        size = max(0.6, hi - lo + 0.1)
        if size >= 0.95:
            continue
        mid = (lo + hi) / 2
        cut_starts = [s for s in (mid, mid - size) if 0 <= s <= 1 - size]
        whole_lo, whole_hi = max(0.0, hi - size), min(lo, 1 - size)
        if not cut_starts or whole_lo > whole_hi:
            continue
        whole = rng.uniform(whole_lo, whole_hi)
        cut = rng.choice(cut_starts)
        return axis, size, whole, cut
    return None


def framing_rows(meta, singles, rng, root, out_dir):
    from PIL import Image

    rows = []
    os.makedirs(os.path.join(root, out_dir, "images"), exist_ok=True)
    for i, objs in singles.items():
        # a discrete thing with an outline: not a mass noun or category ("the whole clothing", "the whole building")
        good = [o for o in objs if not o["truncated"] and not o["occluded"] and 0.03 <= o["area"] <= 0.4
                and (o["name"] == "Person" or o["name"] not in UNCOUNTABLE)]
        if not good:
            continue
        o = rng.choice(good)
        spec = _crops(o, rng)
        if spec is None:
            continue
        axis, size, whole, cut = spec
        with Image.open(os.path.join(root, meta[i]["_src"], "images", i + ".jpg")) as im:
            im = im.convert("RGB")
            W, H = im.size
            paths = []
            for tag, start in (("whole", whole), ("cut", cut)):
                box = (start * W, 0, (start + size) * W, H) if axis == "x" else (0, start * H, W, (start + size) * H)
                p = os.path.join("images", "%s-%s.jpg" % (i, tag))
                im.crop(tuple(int(round(v)) for v in box)).save(os.path.join(root, out_dir, p), "JPEG", quality=92)
                paths.append(p)
        gold = rng.randrange(2)
        imgs = paths if gold == 0 else paths[::-1]
        rec = _base(meta, i, image=False)
        rec["attribution"]["modified"] = "cropped (VisionJev)"
        rec.update(images=imgs, label=gold,
                   question={"type": "choice", "instructions": rng.choice(FRAME_Q).format(x=_the(o["name"])),
                             "criteria": ["the first photo", "the second photo"]})
        rows.append((meta[i]["_split"], rec))
    return rows


def _write(root, name, rows, seed, use, subset, zero_shot):
    tmp, final = os.path.join(root, ".tmp-" + name), os.path.join(root, name)
    os.makedirs(tmp, exist_ok=True)  # oi_framing's crops are already in tmp
    counts, labels = Counter(), defaultdict(Counter)
    splits = ("val",) if zero_shot else ("train", "val")
    files = {s: open(os.path.join(tmp, s + ".jsonl"), "w") for s in splits}
    for split, rec in rows:
        rec = dict(rec, id="%s-%s-%d" % (name, split, counts[split]))
        files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[split] += 1
        labels[split][rec["label"]] += 1
    for f in files.values():
        f.close()
    m = {"dataset": name, "source": "Open Images V7 %s subset (v5 boxes), images from %s" % (subset, SUBSETS[subset][0]),
         "revision": "v7", "license": LICENSE, "seed": seed, "use": use, "counts": dict(counts),
         "zero_shot_only": zero_shot,
         "label_counts": {s: {str(k): v for k, v in c.items()} for s, c in labels.items()}}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    if os.path.exists(final):
        old = os.path.join(root, ".old-%s-%d" % (name, int(time.time())))
        os.rename(final, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    print("%s: %s labels %s" % (name, dict(counts), {s: dict(c) for s, c in labels.items()}), flush=True)


def build(root=DATA_ROOT, seed=0, subset="test"):
    _, _, _, prefix, zero_shot = SUBSETS[subset]
    meta, singles = _load(root, subset)
    print("oiloc %s: %d unrotated CC BY 2.0 images, %d single-instance objects" %
          (subset, len(singles), sum(map(len, singles.values()))), flush=True)
    for name in ("where", "relation", "framing"):
        shutil.rmtree(os.path.join(root, ".tmp-%s_%s" % (prefix, name)), ignore_errors=True)
    kw = dict(seed=seed, subset=subset, zero_shot=zero_shot)
    _write(root, prefix + "_where", where_rows(meta, singles, random.Random(seed)), use="localisation: grid cell", **kw)
    _write(root, prefix + "_relation", relation_rows(meta, singles, random.Random(seed + 1)),
           use="spatial relations: left / above / larger", **kw)
    rows = framing_rows(meta, singles, random.Random(seed + 2), root, ".tmp-%s_framing" % prefix)
    _write(root, prefix + "_framing", rows, use="framing: which crop shows the whole object", **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", choices=sorted(SUBSETS), default="test",
                    help="test = training sets oi_* (5%% val); validation = zero-shot eval sets openimages_*")
    a = ap.parse_args()
    build(a.root, a.seed, a.subset)
