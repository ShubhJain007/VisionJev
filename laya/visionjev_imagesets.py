"""Commercially usable image decision sets (licenses checked 2026-09-25, see CONTEXT.md section 9).

Every source yields ``(split, image_key, image_bytes, state_text, question, label, target|None)``; the builder saves
each image once (RGB JPEG, native aspect, long side <= ``MAX_SIDE``), caps the splits, balances ``noul`` sets that
need it and writes ``<root>/<name>/{train,val}.jsonl`` + ``images/`` + a manifest with the license. Distractors are
always real (other answers / labels / numbers from the same receipt), never generated.

* ``textvqa``            choice: reading text in photos (gold = majority answer; distractors from other questions)
* ``vizwiz_answerable``  noul: can the question be answered from this (blind user's) photo? soft target = share of
                         the 10 annotators who could answer -> a human escalation signal
* ``vqa_rad``            noul: closed yes/no radiology questions (CC0)
* ``pets``               choice: pet breed, 4-8 sampled options of 37
* ``eurosat``            choice: land use from satellite imagery, 4-8 sampled options of 10
* ``cord_total``         choice: which number is the receipt's total? distractors = the receipt's own other amounts

    python -m laya.visionjev_imagesets all
"""
import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from typing import Dict, Iterator, List, Optional

from .visionjev_data import DATA_ROOT, _hub_revision

TRAIN_CAP, VAL_CAP, MAX_SIDE = 8000, 1000, 1024


def _stream(repo, split, config=None):
    import datasets

    ds = datasets.load_dataset(repo, config, split=split, streaming=True)
    return ds.cast_column("image", datasets.Image(decode=False))


def _bytes(img) -> Optional[bytes]:
    if img is None:
        return None
    if img.get("bytes"):
        return img["bytes"]
    if img.get("path") and os.path.exists(img["path"]):
        with open(img["path"], "rb") as f:
            return f.read()
    return None


def _norm(a: str) -> str:
    return re.sub(r"\s+", " ", a.strip().lower().rstrip("."))


def _choice(ins, gold, others, rng):
    opts = [gold] + others
    rng.shuffle(opts)
    return {"type": "choice", "instructions": ins, "criteria": opts}, opts.index(gold)


def _noul(ins):
    return {"type": "noul", "instructions": ins, "criteria": None}


def _sample_labels(ins, gold, space, rng, kmin=4, kmax=8):
    k = min(len(space), rng.randint(kmin, kmax))
    return _choice(ins, gold, rng.sample([s for s in space if s != gold], k - 1), rng)


def textvqa(rng) -> Iterator:
    pool: List[str] = []
    for hub, split in (("train", "train"), ("validation", "val")):
        for i, ex in enumerate(_stream("lmms-lab-encoder/textvqa", hub)):
            answers = [_norm(a) for a in ex["answers"] if a and a.strip()]
            if not answers:
                continue
            gold, votes = Counter(answers).most_common(1)[0]
            if votes < 3 or gold in ("unanswerable", "yes", "no"):
                pool.append(gold)
                continue
            num = gold.replace(",", "").replace(".", "").isdigit()
            cands = [p for p in pool if p not in set(answers) and p.replace(",", "").replace(".", "").isdigit() == num]
            pool.append(gold)
            if len(set(cands)) < 3:
                continue
            q, lab = _choice(ex["question"], gold, rng.sample(sorted(set(cands[-500:])), 3), rng)
            yield split, "tvqa-%s" % ex["image_id"], _bytes(ex["image"]), None, q, lab, None


def vizwiz_answerable(rng) -> Iterator:
    for repo, hub, split in (("Multimodal-Fatima/VizWiz_train", "train", "train"), ("lmms-lab-encoder/VizWiz-VQA", "val", "val")):
        for i, ex in enumerate(_stream(repo, hub)):
            answers = [_norm(a if isinstance(a, str) else a.get("answer", "")) for a in (ex["answers"] or [])]
            if len(answers) < 5:
                continue
            p = 1 - sum(a in ("unanswerable", "unsuitable", "unsuitable image") for a in answers) / len(answers)
            q = _noul("Can this question be answered from the photo: \"%s\"" % ex["question"].strip())
            key = "vizwiz-%s-%s" % (split, ex.get("question_id") or ex.get("id") or i)
            yield split, key, _bytes(ex["image"]), None, q, int(p >= 0.5), [round(1 - p, 3), round(p, 3)]


def vqa_rad(rng) -> Iterator:
    for hub, split in (("train", "train"), ("test", "val")):
        for i, ex in enumerate(_stream("flaviagiammarino/vqa-rad", hub)):
            a = _norm(ex["answer"])
            if a not in ("yes", "no"):
                continue
            yield split, "rad-%s-%d" % (split, i), _bytes(ex["image"]), None, _noul(ex["question"]), int(a == "yes"), None


def pets(rng) -> Iterator:
    import datasets

    names = [n.replace("_", " ") for n in datasets.load_dataset_builder("timm/oxford-iiit-pet").info.features["label"].names]
    for hub, split in (("train", "train"), ("test", "val")):
        for i, ex in enumerate(_stream("timm/oxford-iiit-pet", hub)):
            q, lab = _sample_labels("Which breed is the pet in this photo?", names[ex["label"]], names, rng)
            yield split, "pet-%s-%d" % (split, i), _bytes(ex["image"]), None, q, lab, None


def eurosat(rng) -> Iterator:
    import datasets

    names = [n.lower() for n in datasets.load_dataset_builder("blanchon/EuroSAT_RGB").info.features["label"].names]
    for hub, split in (("train", "train"), ("validation", "val")):
        for i, ex in enumerate(_stream("blanchon/EuroSAT_RGB", hub)):
            q, lab = _sample_labels("What is the main land use shown in this satellite image?", names[ex["label"]], names, rng)
            yield split, "eurosat-%s-%d" % (split, i), _bytes(ex["image"]), None, q, lab, None


def cord_total(rng) -> Iterator:
    for hub, split in (("train", "train"), ("validation", "val"), ("test", "val")):
        for i, ex in enumerate(_stream("naver-clova-ix/cord-v2", hub)):
            gt = json.loads(ex["ground_truth"]).get("gt_parse", {})
            total = str((gt.get("total") or {}).get("total_price") or "").strip()
            if not total:
                continue
            amounts = set(re.findall(r"\d[\d.,]*\d", json.dumps(gt)))
            others = sorted(a for a in amounts if a != total and len(a) >= 3)
            if len(others) < 3:
                continue
            q, lab = _choice("Which amount is this receipt's total?", total, rng.sample(others, 3), rng)
            # key by the SOURCE split: validation and test both feed "val", and "cord-val-<i>" collided, pairing
            # test questions with validation receipts (caught by eyeballing: gold 21.000 on a 44,000 receipt)
            yield split, "cord-%s-%d" % (hub, i), _bytes(ex["image"]), None, q, lab, None


SOURCES: Dict[str, dict] = {
    "textvqa": {"fn": textvqa, "repo": "lmms-lab-encoder/textvqa (facebook/textvqa)", "license": "CC BY 4.0",
                "images": "OpenImages (listed CC BY 2.0)", "use": "reading text in photos"},
    "vizwiz_answerable": {"fn": vizwiz_answerable, "repo": "VizWiz (Multimodal-Fatima/VizWiz_train, lmms-lab-encoder/VizWiz-VQA)",
                          "license": "CC BY 4.0 (vizwiz.org)", "images": "VizWiz (CC BY 4.0)",
                          "use": "escalation: answerable? (human vote shares)", "balance": True},
    "vqa_rad": {"fn": vqa_rad, "repo": "flaviagiammarino/vqa-rad", "license": "CC0-1.0", "images": "CC0",
                "use": "medical yes/no", "balance": True},
    "pets": {"fn": pets, "repo": "timm/oxford-iiit-pet", "license": "CC BY-SA 4.0 (share-alike)",
             "images": "CC BY-SA 4.0", "use": "fine-grained recognition"},
    "eurosat": {"fn": eurosat, "repo": "blanchon/EuroSAT_RGB", "license": "MIT (GitHub phelber/EuroSAT)",
                "images": "Sentinel-2 (Copernicus open data)", "use": "satellite land use"},
    "cord_total": {"fn": cord_total, "repo": "naver-clova-ix/cord-v2", "license": "CC BY 4.0", "images": "CC BY 4.0",
                   "use": "receipt reading / document triage"},
}


def _save(data: bytes, path: str) -> bool:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
        img.save(path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def build(name: str, root: str = DATA_ROOT, seed: int = 0, log=print):
    spec = SOURCES[name]
    rng = random.Random(seed)
    tmp, final = os.path.join(root, ".tmp-" + name), os.path.join(root, name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    rows = {"train": [], "val": []}
    caps = {"train": TRAIN_CAP, "val": VAL_CAP}
    over = 3 if spec.get("balance") else 1
    saved, failed, owner = set(), 0, {}
    for split, key, data, state, q, label, target in spec["fn"](rng):
        if len(rows[split]) >= over * caps[split]:
            if all(len(rows[s]) >= over * caps[s] for s in rows):
                break
            continue
        if data is None:
            failed += 1
            continue
        path = "images/%s.jpg" % re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        digest = hashlib.sha256(data).hexdigest()
        if path in owner and owner[path] != digest:
            raise ValueError("image key collision in %s: %s maps to two different images" % (name, path))
        owner[path] = digest
        if path not in saved:
            if not _save(data, os.path.join(tmp, path)):
                failed += 1
                continue
            saved.add(path)
        rows[split].append({"image": path, "state_text": state, "question": q, "label": label, "target": target})
    labels = {}
    for split, rs in rows.items():
        if spec.get("balance"):
            pos, neg = [r for r in rs if r["label"] == 1], [r for r in rs if r["label"] == 0]
            n = min(len(pos), len(neg), caps[split] // 2)
            rs = pos[:n] + neg[:n]
            rng.shuffle(rs)
        rs = rs[:caps[split]]
        rows[split] = rs
        with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
            for i, r in enumerate(rs):
                rec = {"id": "%s-%s-%d" % (name, split, i), "image": r["image"], "state_text": r["state_text"],
                       "question": r["question"], "label": r["label"]}
                if r["target"] is not None:
                    rec["target"] = r["target"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        labels[split] = dict(Counter(r["label"] for r in rs))
    manifest = {"dataset": name, "source": spec["repo"], "revision": _hub_revision(spec["repo"].split(" ")[0]),
                "license": {"annotations": spec["license"], "images": spec["images"], "commercial": True,
                            "verified": "2026-09-25 HF card / upstream; see CONTEXT.md section 9"},
                "use": spec["use"], "seed": seed, "max_side": MAX_SIDE, "images_saved": len(saved),
                "failed": failed, "counts": {k: len(v) for k, v in rows.items()}, "label_counts": labels,
                "validator": {"noul_min_minority": 0.05}}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if os.path.exists(final):
        old = os.path.join(root, ".old-%s-%d" % (name, int(time.time())))
        os.rename(final, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    log("%s: %s images %d failed %d" % (name, manifest["counts"], len(saved), failed))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (list(SOURCES) if a.names == ["all"] else a.names):
        try:
            build(n, a.root, log=lambda m: print(m, flush=True))
        except Exception as e:
            print("%s: FAILED %r" % (n, e), flush=True)
