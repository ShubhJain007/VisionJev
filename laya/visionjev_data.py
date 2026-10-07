"""VisionJev data prep: commercially usable datasets in the prepared-dataset format ``laya.vlm_train`` reads.

Ported from ``laya/vision_data.py`` on the ``siglip-projector-experiment`` branch, with two additions:

* a license registry (``LICENSES``): every dataset records its annotation and image licenses and whether it may go
  into a commercial training mix, and ``build_dataset`` refuses one that may not unless asked;
* a COCO per-image license filter: COCO images are Flickr photos under per-image licenses (the ``licenses`` table of
  the COCO annotation files: 1-3 NonCommercial, 4 BY, 5 BY-SA, 6 BY-ND, 7 no known restrictions, 8 US Gov). Only
  images whose license id is in ``COCO_COMMERCIAL`` are kept, and each record carries its ``image_license``.

Layout, one directory per dataset under a root (default ``$VISIONJEV_DATA``, else ``../data/vqa``)::

    <root>/<name>/images/<image_id>.jpg   RGB JPEG, long side <= 512 px (may be shared by several records)
    <root>/<name>/{train,val}.jsonl       one record per line
    <root>/<name>/manifest.json           source, pinned revision, licenses, filter, counts
    <root>/<name>/_READY                  written last

Record: ``{"id", "image", "state_text", "question": {"type", "instructions", "criteria"}, "label", "image_license"}``.

Needs the COCO license table and the official A-OKVQA annotations (the Hub copy has no COCO image id)::

    python -m laya.visionjev_data fetch-meta           # -> <raw>/coco_image_license.json, <raw>/aokvqa_v1p0_*.json
    python -m laya.visionjev_data build aokvqa vqav2_yesno
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

MAX_SIDE = 512
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("VISIONJEV_DATA", os.path.normpath(os.path.join(HERE, "..", "..", "data", "vqa")))
RAW_ROOT = os.environ.get("VISIONJEV_RAW", os.path.normpath(os.path.join(HERE, "..", "..", "data", "raw")))

#: COCO license ids kept for the commercial mix. 6 (BY-ND) is left out by default: resizing and training on an image
#: may count as a derivative. ``--coco-licenses 4,5,6,7,8`` puts it back.
COCO_COMMERCIAL = frozenset({4, 5, 7, 8})
COCO_LICENSE_NAMES = {1: "CC BY-NC-SA 2.0", 2: "CC BY-NC 2.0", 3: "CC BY-NC-ND 2.0", 4: "CC BY 2.0",
                      5: "CC BY-SA 2.0", 6: "CC BY-ND 2.0", 7: "No known copyright restrictions",
                      8: "United States Government Work"}
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
AOKVQA_URL = "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"

#: Every dataset VisionJev may prepare. ``commercial``: may go into the shippable training mix (after any per-image
#: filter). ``verified``: where the license was read (see CONTEXT.md section 9). Not legal advice.
LICENSES: Dict[str, Dict] = {
    "aokvqa": {"annotations": "Apache-2.0", "images": "COCO (per-image Flickr licenses)", "image_filter": "coco",
               "commercial": True, "verified": "github.com/allenai/aokvqa LICENSE"},
    "vqav2_yesno": {"annotations": "CC BY 4.0", "images": "COCO (per-image Flickr licenses)", "image_filter": "coco",
                    "commercial": True, "verified": "visualqa.org/terms.html"},
    "scienceqa": {"annotations": "CC BY-NC-SA 4.0", "images": "ScienceQA", "commercial": False,
                  "verified": "upstream ScienceQA terms; the Hub mirror's CC BY-SA conflicts, treated as NC"},
}


def coco_license_table(raw: str = RAW_ROOT) -> Dict[int, int]:
    with open(os.path.join(raw, "coco_image_license.json")) as f:
        return {int(k): int(v) for k, v in json.load(f).items()}


# ---- sources: yield (record_without_image_path, image_id, image_bytes, split, coco_image_id or None) ----------------


def _stream(name: str, split: str, revision: Optional[str]):
    import datasets

    ds = datasets.load_dataset(name, split=split, streaming=True, revision=revision)
    return ds.cast_column("image", datasets.Image(decode=False))


def _choice(instructions: str, choices: List[str]) -> Optional[Dict]:
    choices = [str(c) for c in choices]
    if len(choices) < 2 or len(set(choices)) != len(choices):
        return None  # choice criteria become dict keys in the loader; duplicates would shift label indices
    if any(not c.strip() for c in choices):
        return None  # a blank option (A-OKVQA val has one) renders as an empty line the gold may point at
    return {"type": "choice", "instructions": instructions, "criteria": choices}


def aokvqa_source(revision: Optional[str] = None, raw: str = RAW_ROOT) -> Iterator[Tuple]:
    for split, out in (("train", "train"), ("validation", "val")):
        with open(os.path.join(raw, "aokvqa_v1p0_%s.json" % ("val" if out == "val" else "train"))) as f:
            image_of = {q["question_id"]: q["image_id"] for q in json.load(f)}
        for ex in _stream("HuggingFaceM4/A-OKVQA", split, revision):
            q = _choice(ex["question"], ex["choices"])
            if q is None or ex["correct_choice_idx"] is None or ex["question_id"] not in image_of:
                continue
            coco_id = int(image_of[ex["question_id"]])
            rec = {"id": ex["question_id"], "state_text": None, "question": q, "label": int(ex["correct_choice_idx"])}
            yield rec, "%012d" % coco_id, ex["image"]["bytes"], out, coco_id


def coco_reserved_splits(raw: str = RAW_ROOT) -> Dict[int, str]:
    """COCO image id -> the split another prepared COCO dataset already put it in (A-OKVQA's official train / val).
    Re-split datasets must follow it, or the same photo lands in one set's train and another's val."""
    out = {}
    for split in ("train", "val"):
        with open(os.path.join(raw, "aokvqa_v1p0_%s.json" % split)) as f:
            out.update({int(q["image_id"]): split for q in json.load(f)})
    return out


def _vqav2_split(image_id: int, reserved: Dict[int, str], val_pct: int = 10) -> str:
    if image_id in reserved:
        return reserved[image_id]
    return "val" if int(hashlib.md5(str(image_id).encode()).hexdigest(), 16) % 100 < val_pct else "train"


def vqav2_yesno_source(revision: Optional[str] = None, raw: str = RAW_ROOT) -> Iterator[Tuple]:
    """Official VQAv2 *validation* only (it has answers), re-split by image id: an image A-OKVQA already uses keeps
    A-OKVQA's split, the rest go by md5 bucket (~10% val). No image is in both splits, in this set or across them."""
    reserved = coco_reserved_splits(raw)
    for ex in _stream("lmms-lab-encoder/VQAv2", "validation", revision):
        if ex["answer_type"] != "yes/no" or ex["multiple_choice_answer"] not in ("yes", "no"):
            continue
        q = {"type": "noul", "instructions": ex["question"], "criteria": None}
        rec = {"id": str(ex["question_id"]), "state_text": None, "question": q,
               "label": int(ex["multiple_choice_answer"] == "yes")}
        yield rec, "%012d" % int(ex["image_id"]), ex["image"]["bytes"], _vqav2_split(int(ex["image_id"]), reserved), int(ex["image_id"])


SOURCES: Dict[str, Tuple[str, Callable]] = {
    "aokvqa": ("HuggingFaceM4/A-OKVQA", aokvqa_source),
    "vqav2_yesno": ("lmms-lab-encoder/VQAv2", vqav2_yesno_source),
}


# ---- writing ------------------------------------------------------------------------------------------------------


def reencode_jpeg(data: bytes, max_side: int = MAX_SIDE) -> bytes:
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_side, max_side), Image.BICUBIC)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=90)
    return out.getvalue()


def _hub_revision(repo: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def build_dataset(name: str, root: str = DATA_ROOT, caps: Optional[Dict[str, int]] = None, raw: str = RAW_ROOT,
                  coco_licenses: Set[int] = COCO_COMMERCIAL, allow_noncommercial: bool = False, workers: int = 8,
                  log=print) -> Dict:
    """Stream ``name`` from the Hub into ``<root>/.tmp-<name>``, then rename it to ``<root>/<name>`` and write _READY.

    ``caps`` bounds the records per split (0 or missing = no cap). Records whose COCO image license is not in
    ``coco_licenses`` are skipped before their image is decoded."""
    lic = LICENSES[name]
    if not lic["commercial"] and not allow_noncommercial:
        raise ValueError("%s is not cleared for commercial use (%s); pass allow_noncommercial" % (name, lic["annotations"]))
    caps = caps or {}
    hub_id, source = SOURCES[name]
    revision = _hub_revision(hub_id)
    coco = coco_license_table(raw) if lic.get("image_filter") == "coco" else None

    tmp = os.path.join(root, ".tmp-" + name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    splits = ("train", "val")
    files = {s: open(os.path.join(tmp, "%s.jsonl" % s), "w") for s in splits}
    counts = {s: 0 for s in splits}
    labels: Dict[str, Dict] = {s: {} for s in splits}
    lic_counts: Dict[str, int] = {}
    skipped = {"license": 0, "unknown_coco_id": 0, "undecodable": 0}
    written: Set[str] = set()
    t0 = time.time()

    def full(s):
        return bool(caps.get(s)) and counts[s] >= caps[s]

    def save(image_id, data):
        try:
            with open(os.path.join(tmp, "images", image_id + ".jpg"), "wb") as f:
                f.write(reencode_jpeg(data))
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(workers) as pool:
        pending: List = []
        for rec, image_id, data, split, coco_id in source(revision=revision, raw=raw):
            if split not in counts or full(split):
                if all(full(s) for s in splits):
                    break
                continue
            if coco is not None:
                lid = coco.get(coco_id)
                if lid is None:
                    skipped["unknown_coco_id"] += 1
                    continue
                if lid not in coco_licenses:
                    skipped["license"] += 1
                    continue
                rec["image_license"] = COCO_LICENSE_NAMES[lid]
            if image_id not in written:
                written.add(image_id)
                pending.append((rec, split, pool.submit(save, image_id, data)))
            else:
                pending.append((rec, split, None))
            rec["image"] = "images/%s.jpg" % image_id
            counts[split] += 1
            if len(pending) >= 256:
                skipped["undecodable"] += _flush(pending, files, labels, counts, lic_counts)
            n = sum(counts.values())
            if n % 5000 == 0:
                log("%s: %s kept, %s skipped (%.0fs)" % (name, counts, skipped, time.time() - t0))
        skipped["undecodable"] += _flush(pending, files, labels, counts, lic_counts)
    for f in files.values():
        f.close()

    manifest = {
        "dataset": name,
        "source": hub_id,
        "revision": revision,
        "license": {**lic, "coco_licenses_kept": sorted(coco_licenses) if coco is not None else None,
                    "image_licenses": lic_counts},
        "counts": counts,
        "label_counts": labels,
        "images": len(written),
        "caps": caps,
        "max_side": MAX_SIDE,
        "skipped": skipped,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    final = os.path.join(root, name)
    if os.path.exists(final):
        old = os.path.join(root, ".old-%s-%d" % (name, int(time.time())))
        os.rename(final, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    log("%s: done %s" % (name, json.dumps({k: manifest[k] for k in ("counts", "images", "skipped", "seconds")})))
    return manifest


def _flush(pending, files, labels, counts, lic_counts) -> int:
    """Write records whose image saved; drop the rest (their image failed to decode)."""
    bad_images, bad = set(), 0
    for rec, split, fut in pending:
        if fut is not None and not fut.result():
            bad_images.add(rec["image"])
        if rec["image"] in bad_images:
            counts[split] -= 1
            bad += 1
            continue
        files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        labels[split][rec["label"]] = labels[split].get(rec["label"], 0) + 1
        if "image_license" in rec:
            lic_counts[rec["image_license"]] = lic_counts.get(rec["image_license"], 0) + 1
    pending.clear()
    return bad


# ---- metadata fetch -----------------------------------------------------------------------------------------------


def fetch_meta(raw: str = RAW_ROOT, log=print) -> None:
    """Download the COCO license table and the official A-OKVQA annotations into ``raw``."""
    import tarfile
    import urllib.request
    import zipfile

    os.makedirs(raw, exist_ok=True)
    table = os.path.join(raw, "coco_image_license.json")
    if not os.path.exists(table):
        zpath = os.path.join(raw, "annotations_trainval2017.zip")
        if not os.path.exists(zpath):
            log("downloading " + COCO_ANNOTATIONS_URL)
            urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zpath)
        lic = {}
        with zipfile.ZipFile(zpath) as z:
            for s in ("train", "val"):
                d = json.loads(z.read("annotations/captions_%s2017.json" % s))
                names = {x["id"]: x["name"] for x in d["licenses"]}
                assert "NonCommercial" in names[1] and names[4] == "Attribution License", names  # the table we assume
                lic.update({im["id"]: im["license"] for im in d["images"]})
        with open(table, "w") as f:
            json.dump(lic, f)
        log("wrote %s (%d images)" % (table, len(lic)))
    if not os.path.exists(os.path.join(raw, "aokvqa_v1p0_train.json")):
        tpath = os.path.join(raw, "aokvqa_v1p0.tar.gz")
        log("downloading " + AOKVQA_URL)
        urllib.request.urlretrieve(AOKVQA_URL, tpath)
        with tarfile.open(tpath) as t:
            t.extractall(raw, filter="data")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-meta")
    b = sub.add_parser("build")
    b.add_argument("names", nargs="+", choices=sorted(SOURCES))
    b.add_argument("--root", default=DATA_ROOT)
    b.add_argument("--train-cap", type=int, default=0)
    b.add_argument("--val-cap", type=int, default=0)
    b.add_argument("--coco-licenses", default=",".join(map(str, sorted(COCO_COMMERCIAL))))
    b.add_argument("--allow-noncommercial", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "fetch-meta":
        fetch_meta()
        return
    os.makedirs(a.root, exist_ok=True)
    kept = {int(x) for x in a.coco_licenses.split(",") if x}
    for name in a.names:
        build_dataset(name, a.root, {"train": a.train_cap, "val": a.val_cap}, coco_licenses=kept,
                      allow_noncommercial=a.allow_noncommercial, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
