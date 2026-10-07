"""Open Images V7 validation (annotations CC BY 4.0; images listed as CC BY 2.0) as zero-shot decision sets.

Only images whose metadata lists CC BY 2.0 are used; each record keeps the image's license and attribution
(author, landing page), as Open Images asks users to verify licenses per image. Images are stored at their
native aspect ratio with the long side <= ``MAX_SIDE`` (so tiling / native-resolution paths see real sizes).

* ``openimages_presence`` (noul): "Is there a <class> in this image?" from HUMAN-verified image-level labels,
  half positives (Confidence 1), half verified negatives (Confidence 0), several phrasings.
* ``openimages_count`` (score, 5 levels none / one / two / three / four or more): how many <class> — from the
  exhaustive human boxes of a verified-positive class, skipping classes with a group-of box (uncountable), plus
  "none" from verified negatives.

    python -m laya.visionjev_openimages --images 1500
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from .visionjev_data import DATA_ROOT, RAW_ROOT

RAW = os.path.join(RAW_ROOT, "openimages")
BUCKET = "https://open-images-dataset.s3.amazonaws.com/%s/%s.jpg"  # (subset, image id)
#: subset -> (image-folder dataset, presence dataset, count dataset, metadata csv, labels csv, boxes csv, zero-shot?)
SUBSETS = {
    "validation": ("openimages_images", "openimages_presence", "openimages_count", "validation-images-with-rotation.csv",
                   "validation-annotations-human-imagelabels-boxable.csv", "validation-annotations-bbox.csv", True),
    # training source: Open Images *test* (disjoint from the validation images the zero-shot sets use); 5% held out
    "test": ("oi_train_images", "oi_presence", "oi_count", "test-images-with-rotation.csv",
             "test-annotations-human-imagelabels-boxable.csv", "test-annotations-bbox.csv", False),
}
META_URLS = {"test-images-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
             "test-annotations-human-imagelabels-boxable.csv": "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv",
             "test-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"}
CC_BY = "https://creativecommons.org/licenses/by/2.0/"
MAX_SIDE = 1024
PAGES = "openimages_images"
LICENSE = {"annotations": "CC BY 4.0 (Google LLC)", "images": "listed as CC BY 2.0 per image (no warranty; verify)",
           "commercial": True, "verified": "storage.googleapis.com/openimages/web/factsfigures_v7.html, 2026-09-25"}
PRESENCE = ("Is there {x} in this image?", "Does this image contain {x}?", "Can you see {x} in the picture?")
COUNT = ("How many {x} are in this image?", "Count the {x} visible in the image.")
LEVELS = ["none", "one", "two", "three", "four or more"]
#: classes a person would count differently from the boxes (eyeballing: "Count the stairs" -> gold "one" = one
#: staircase box, but people count steps): collective / mass nouns, and names already plural or pair nouns
UNCOUNTABLE = {"Stairs", "Clothing", "Footwear", "Furniture", "Food", "Fashion accessory", "Tableware", "Kitchenware",
               "Toy", "Sports equipment", "Office supplies", "Plant", "Houseplant", "Flower", "Vegetable", "Fruit",
               "Seafood", "Baked goods", "Drink", "Cosmetics", "Auto part", "Vehicle", "Land vehicle", "Animal",
               "Mammal", "Bird", "Fish", "Insect", "Person", "Human body", "Human head", "Building", "Tree", "Window",
               "Tire", "Wheel", "Glasses", "Sunglasses", "Goggles", "Jeans", "Shorts", "Trousers", "Scissors",
               "Headphones", "Tongs", "Binoculars", "Swim cap"}


def _countable(name: str) -> bool:
    return name not in UNCOUNTABLE and not name.lower().endswith("s")


def _article(name: str) -> str:
    return ("an " if name[0].lower() in "aeiou" else "a ") + name.lower()


def _plural(name: str) -> str:
    n = name.lower()
    return n if n.endswith("s") else n + ("es" if n.endswith(("sh", "ch", "x")) else "s")


def _read(name):
    if not os.path.exists(os.path.join(RAW, name)) and name in META_URLS:
        urllib.request.urlretrieve(META_URLS[name], os.path.join(RAW, name))
    with open(os.path.join(RAW, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fetch(subset: str, image_id: str, rotation: str, path: str) -> bool:
    from PIL import Image

    try:
        data = urllib.request.urlopen(BUCKET % (subset, image_id), timeout=30).read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if rotation and rotation not in ("", "0", "0.0"):
            img = img.rotate(-float(rotation), expand=True)
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
        img.save(path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def build(root: str = DATA_ROOT, n_images: int = 1500, seed: int = 0, subset: str = "validation", log=print):
    PAGES, PRES, CNT, META_CSV, LABEL_CSV, BOX_CSV, zero_shot = SUBSETS[subset]
    rng = random.Random(seed)
    with open(os.path.join(RAW, "class-descriptions-boxable.csv"), newline="", encoding="utf-8") as f:
        names = {row[0]: row[1] for row in csv.reader(f)}
    meta = {r["ImageID"]: r for r in _read(META_CSV) if r["License"] == CC_BY}
    labels = defaultdict(dict)  # image -> class -> 0/1 (human-verified)
    for r in _read(LABEL_CSV):
        if r["ImageID"] in meta and r["LabelName"] in names:
            labels[r["ImageID"]][r["LabelName"]] = int(float(r["Confidence"]) > 0.5)
    boxes, groups = Counter(), set()
    for r in _read(BOX_CSV):
        if r["ImageID"] in meta:
            boxes[(r["ImageID"], r["LabelName"])] += 1
            if r["IsGroupOf"] == "1":
                groups.add((r["ImageID"], r["LabelName"]))
    pool = sorted(i for i, l in labels.items() if 1 in l.values() and 0 in l.values())
    rng.shuffle(pool)
    chosen = pool[:n_images]
    log("openimages: %d CC BY 2.0 images with verified pos+neg labels; using %d" % (len(pool), len(chosen)))

    tmp = {n: os.path.join(root, ".tmp-" + n) for n in (PAGES, PRES, CNT)}
    for d in tmp.values():
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    os.makedirs(os.path.join(tmp[PAGES], "images"))
    with ThreadPoolExecutor(16) as pool_ex:
        ok = dict(zip(chosen, pool_ex.map(lambda i: _fetch(subset, i, meta[i].get("Rotation", ""),
                                                            os.path.join(tmp[PAGES], "images", i + ".jpg")), chosen)))
    counts, lab = Counter(), {PRES: Counter(), CNT: Counter()}
    splits = ("val",) if zero_shot else ("train", "val")
    fp = {sp: open(os.path.join(tmp[PRES], sp + ".jsonl"), "w") for sp in splits}
    fc = {sp: open(os.path.join(tmp[CNT], sp + ".jsonl"), "w") for sp in splits}
    for i, image_id in enumerate(chosen):
        if not ok[image_id]:
            counts["download_failed"] += 1
            continue
        m = meta[image_id]
        split = "val" if zero_shot or int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 100 < 5 else "train"
        base = {"image": "../%s/images/%s.jpg" % (PAGES, image_id), "state_text": None,
                "image_license": "CC BY 2.0",
                "attribution": {"author": m["Author"], "page": m["OriginalLandingURL"], "title": m["Title"]}}
        pos = [c for c, v in labels[image_id].items() if v == 1]
        neg = [c for c, v in labels[image_id].items() if v == 0]
        want_pos = i % 2 == 0
        cls = rng.choice(pos if want_pos else neg)
        rec = dict(base, id=image_id + "-presence", label=int(cls in pos),
                   question={"type": "noul", "instructions": rng.choice(PRESENCE).format(x=_article(names[cls])),
                             "criteria": None})
        fp[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        lab[PRES][rec["label"]] += 1
        countable = [c for c in pos if boxes[(image_id, c)] > 0 and (image_id, c) not in groups and _countable(names[c])]
        neg = [c for c in neg if _countable(names[c])] or neg
        if countable and (i % 5 != 0 or not neg):
            cls = rng.choice(countable)
            level = min(boxes[(image_id, cls)], 4)
        else:
            cls, level = rng.choice(neg), 0
        rec = dict(base, id=image_id + "-count", label=level,
                   question={"type": "score", "instructions": rng.choice(COUNT).format(x=_plural(names[cls])),
                             "criteria": list(LEVELS)})
        fc[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        lab[CNT][level] += 1
    for f in list(fp.values()) + list(fc.values()):
        f.close()
    for n in tmp:
        m = {"dataset": n, "source": "Open Images V7 %s (v5 annotations)" % subset, "revision": "v7", "license": LICENSE,
             "seed": seed, "images": n_images, "max_side": MAX_SIDE, "skipped": dict(counts),
             "label_counts": {str(k): v for k, v in lab.get(n, {}).items()}, "zero_shot_only": zero_shot}
        with open(os.path.join(tmp[n], "manifest.json"), "w") as f:
            json.dump(m, f, indent=2)
        final = os.path.join(root, n)
        if os.path.exists(final):
            old = os.path.join(root, ".old-%s-%d" % (n, int(time.time())))
            os.rename(final, old)
            shutil.rmtree(old, ignore_errors=True)
        os.rename(tmp[n], final)
        with open(os.path.join(final, "_READY"), "w") as f:
            json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    log("openimages: done %s, skipped %s" % ({k: dict(v) for k, v in lab.items()}, dict(counts)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--images", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", choices=sorted(SUBSETS), default="validation",
                    help="validation = the zero-shot sets; test = a training source (5%% held out as val)")
    a = ap.parse_args()
    build(a.root, a.images, a.seed, a.subset)
