"""DocLayNet (CDLA-Permissive-1.0) as VisionJev decision data: one pass over the pages, four prepared datasets.

Source: ``docling-project/DocLayNet-v1.2`` (parquet: 1025 x 1025 page renders, layout boxes in pixel ``[x, y, w, h]``,
the PDF text cells inside each box, and the page's ``doc_category``). Official ``train`` -> train, ``validation`` ->
val; ``test`` is left untouched for later held-out evaluation.

Per page (images stored once, at native 1025 px, under ``<root>/doclaynet_pages/images/``):

* ``doclaynet_type``        choice over the 6 document categories (option order shuffled per record).
* ``doclaynet_layout``      noul "does this page contain a <table / formula / footnote / ...>?", balanced: half the
                            questions ask about an element on the page, half about one that is not.
* ``doclaynet_read_small``  choice "which of these snippets appears on this page?": the gold is a text cell of this
* ``doclaynet_read_large``  page, the 3 distractors real cells of other pages of the same category (similar length,
                            checked absent from this page). Split by the gold cell's height in pixels at 1025 px
                            (``SMALL_TEXT_PX``): small text is what a single 512 px tile cannot resolve, so this pair
                            is the probe for image splitting.

Distractors are never generated: they are drawn from the real snippet pool (CONTEXT.md section 9).

    python -m laya.visionjev_doclaynet --train-per-category 1500 --val-per-category 200
"""
import argparse
import io
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional

from .visionjev_data import DATA_ROOT, _hub_revision

HUB_ID = "docling-project/DocLayNet-v1.2"
LICENSE = {"annotations": "CDLA-Permissive-1.0", "images": "DocLayNet page renders (CDLA-Permissive-1.0)",
           "commercial": True, "verified": "HF card docling-project/DocLayNet(-v1.2) Licensing Information"}
PAGES = "doclaynet_pages"
DATASETS = ("doclaynet_type", "doclaynet_layout", "doclaynet_read_small", "doclaynet_read_large")
MAX_SIDE = 1025          # native render size; splitting experiments need the full resolution
SMALL_TEXT_PX = 10.0     # gold cell height (px at 1025) below which the text counts as small (median is ~9.8)

CATEGORIES = {  # doc_category -> option text
    "financial_reports": "financial report",
    "scientific_articles": "scientific article",
    "laws_and_regulations": "law or regulation",
    "government_tenders": "government tender",
    "manuals": "manual",
    "patents": "patent",
}
LAYOUT = {  # category_id -> noun phrase (10 = Text is on nearly every page, so it is not asked about)
    1: "a figure or table caption", 2: "a footnote", 3: "a mathematical formula", 4: "a list item",
    5: "a page footer", 6: "a page header", 7: "a picture or chart", 8: "a section header", 9: "a table",
    11: "a document title",
}
TYPE_INSTRUCTIONS = ("What kind of document is this page from?", "Which type of document does this page belong to?",
                     "Classify the document this page comes from.")
LAYOUT_INSTRUCTIONS = ("Does this page contain {x}?", "Is there {x} on this page?", "Does the page layout include {x}?")
READ_INSTRUCTIONS = ("Which of these text snippets appears on this page?",
                     "Which of the following phrases is printed on this page?",
                     "One of these snippets is on the page. Which one?")
POOL_SIZE = 4000          # snippets remembered per category for distractors
MIN_POOL = 200            # skip the read question until a category's pool has this many snippets


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _snippet_ok(text: str) -> bool:
    t = text.strip()
    words = t.split()
    if not 3 <= len(words) <= 10 or not 15 <= len(t) <= 60:
        return False
    return sum(ch.isalpha() for ch in t) / len(t) >= 0.7


def _page_snippets(ex) -> List[Dict]:
    out = []
    for cells in ex["pdf_cells"] or []:
        for c in cells or []:
            text = (c.get("text") or "").strip()
            if text and _snippet_ok(text):
                out.append({"text": re.sub(r"\s+", " ", text), "h": float(c["bbox"][3])})
    return out


def _page_text(ex) -> str:
    return _norm(" ".join((c.get("text") or "") for cells in (ex["pdf_cells"] or []) for c in (cells or [])))


def _choice(instructions: str, gold: str, others: List[str], rng: random.Random):
    opts = [gold] + others
    rng.shuffle(opts)
    return {"type": "choice", "instructions": instructions, "criteria": opts}, opts.index(gold)


def _stream(split: str, seed: int):
    import datasets

    ds = datasets.load_dataset(HUB_ID, split=split, streaming=True,
                               columns=["image", "bboxes", "category_id", "pdf_cells", "metadata"])
    ds = ds.cast_column("image", datasets.Image(decode=False))
    return ds.shuffle(seed=seed, buffer_size=500)  # the files are ordered by collection; shuffle shards + buffer


def _save_page(data: bytes, path: str) -> bool:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
        img.save(path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def build(root: str = DATA_ROOT, per_category: Optional[Dict[str, int]] = None, seed: int = 0, log=print) -> Dict:
    per_category = per_category or {"train": 1500, "val": 200}
    revision = _hub_revision(HUB_ID)
    rng = random.Random(seed)
    tmp = {name: os.path.join(root, ".tmp-" + name) for name in (PAGES,) + DATASETS}
    for d in tmp.values():
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    os.makedirs(os.path.join(tmp[PAGES], "images"))
    files = {(n, s): open(os.path.join(tmp[n], s + ".jsonl"), "w") for n in DATASETS for s in per_category}
    counts = {n: Counter() for n in DATASETS}
    labels = {n: Counter() for n in DATASETS}
    pages = Counter()
    skipped = Counter()
    pool: Dict[str, deque] = defaultdict(lambda: deque(maxlen=POOL_SIZE))
    seen = {sp: {e: [0, 0] for e in LAYOUT} for sp in per_category}  # split -> element -> [absent, present] asked
    t0 = time.time()

    for split, hub_split in (("train", "train"), ("val", "validation")):
        cap = per_category[split]
        per_cat = Counter()
        for ex in _stream(hub_split, seed):
            meta = ex["metadata"]
            cat = meta["doc_category"]
            if cat not in CATEGORIES:
                skipped["unknown_category"] += 1
                continue
            snippets = _page_snippets(ex)
            if per_cat[cat] >= cap:
                pool[cat].extend(s["text"] for s in snippets[:5])  # still feed the distractor pool
                if all(per_cat[c] >= cap for c in CATEGORIES):
                    break
                continue
            key = meta["page_hash"][:20]
            rel = "images/%s.jpg" % key
            if not _save_page(ex["image"]["bytes"], os.path.join(tmp[PAGES], rel)):
                skipped["undecodable"] += 1
                continue
            per_cat[cat] += 1
            pages[split] += 1
            image = "../%s/%s" % (PAGES, rel)
            base = {"image": image, "state_text": None}

            # 1. document type
            q, lab = _choice(rng.choice(TYPE_INSTRUCTIONS), CATEGORIES[cat],
                             [v for k, v in CATEGORIES.items() if k != cat], rng)
            _write(files, counts, labels, "doclaynet_type", split, dict(base, id=key + "-type", question=q, label=lab))

            # 2. layout element present / absent, balanced per element: ask about an element whose answer on this
            #    page is the one that element is short of, so no question text predicts its own answer
            present = {c for c in ex["category_id"] if c in LAYOUT}
            need = [e for e in LAYOUT if (e in present) == (seen[split][e][1] <= seen[split][e][0])]
            elem = rng.choice(need or sorted(LAYOUT))
            seen[split][elem][int(elem in present)] += 1
            q = {"type": "noul", "instructions": rng.choice(LAYOUT_INSTRUCTIONS).format(x=LAYOUT[elem]), "criteria": None}
            _write(files, counts, labels, "doclaynet_layout", split,
                   dict(base, id=key + "-layout", question=q, label=int(elem in present)))

            # 3. read a snippet (small or large text, alternating when the page has both)
            if snippets and len(pool[cat]) >= MIN_POOL:
                small = [s for s in snippets if s["h"] < SMALL_TEXT_PX]
                large = [s for s in snippets if s["h"] >= SMALL_TEXT_PX]
                prefer_small = pages[split] % 2 == 0
                group = (small if prefer_small else large) or small or large
                gold = rng.choice(group)
                page_text = _page_text(ex)
                n = len(gold["text"])
                cands = [t for t in pool[cat] if 0.6 * n <= len(t) <= 1.4 * n and _norm(t) not in page_text
                         and _norm(t) != _norm(gold["text"])]
                cands = list(dict.fromkeys(cands))
                if len(cands) >= 3:
                    q, lab = _choice(rng.choice(READ_INSTRUCTIONS), gold["text"], rng.sample(cands, 3), rng)
                    name = "doclaynet_read_small" if gold["h"] < SMALL_TEXT_PX else "doclaynet_read_large"
                    _write(files, counts, labels, name, split,
                           dict(base, id=key + "-read", question=q, label=lab))
                else:
                    skipped["read_no_distractors"] += 1
            else:
                skipped["read_pool_warmup" if snippets else "read_no_snippet"] += 1
            pool[cat].extend(s["text"] for s in snippets[:5])

            if sum(pages.values()) % 500 == 0:
                log("doclaynet: %s pages %s, per category %s (%.0fs)" % (split, dict(pages), dict(per_cat), time.time() - t0))
        log("doclaynet: %s done, per category %s" % (split, dict(per_cat)))

    for f in files.values():
        f.close()
    manifests = {}
    for name in (PAGES,) + DATASETS:
        final = os.path.join(root, name)
        m = {"dataset": name, "source": HUB_ID, "revision": revision, "license": LICENSE, "seed": seed,
             "per_category": per_category, "max_side": MAX_SIDE, "small_text_px": SMALL_TEXT_PX,
             "counts": dict(counts[name]) if name in counts else dict(pages),
             "label_counts": {"%s/%s" % k: v for k, v in labels[name].items()} if name in labels else {},
             "skipped": dict(skipped), "seconds": round(time.time() - t0, 1)}
        if name == "doclaynet_layout":
            m["layout_balance"] = {sp: {LAYOUT[e]: {"absent": v[0], "present": v[1]} for e, v in d.items()}
                                   for sp, d in seen.items()}
        with open(os.path.join(tmp[name], "manifest.json"), "w") as f:
            json.dump(m, f, indent=2)
        if os.path.exists(final):
            old = os.path.join(root, ".old-%s-%d" % (name, int(time.time())))
            os.rename(final, old)
            shutil.rmtree(old, ignore_errors=True)
        os.rename(tmp[name], final)
        with open(os.path.join(final, "_READY"), "w") as f:
            json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
        manifests[name] = m
    log("doclaynet: done pages %s, questions %s, skipped %s"
        % (dict(pages), {n: dict(counts[n]) for n in DATASETS}, dict(skipped)))
    return manifests


def _write(files, counts, labels, name, split, rec) -> None:
    files[(name, split)].write(json.dumps(rec, ensure_ascii=False) + "\n")
    counts[name][split] += 1
    labels[name][(split, rec["label"])] += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--train-per-category", type=int, default=1500)
    ap.add_argument("--val-per-category", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.root, exist_ok=True)
    build(a.root, {"train": a.train_per_category, "val": a.val_per_category}, a.seed, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
