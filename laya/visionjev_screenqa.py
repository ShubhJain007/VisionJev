"""RICO-ScreenQA (CC BY 4.0; Rico screenshots): questions about real Android app screens, human answers.

* ``rico_screenqa``: an answerable question becomes a choice between its short answer (the answer's UI-element texts)
  and the short answers of *other questions about the same screen* first, then of other screens (so every option is
  text that appears on some real screen); an unanswerable one ("<no answer>") becomes "can this be answered from the
  screen?" (noul, 0), paired with answerable questions of the same screen asked the same way (1).
  ScreenQA train shards -> train, test shards -> val; screenshots stored once per screen, long side 960; val rows
  whose screenshot duplicates a train screenshot are dropped (``visionjev_dedupe``).

    python -m laya.visionjev_screenqa --train-shards 3 --test-shards 1
"""
import argparse
import ast
import io
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict

from .visionjev_data import DATA_ROOT

REPO = "rootsautomation/RICO-ScreenQA"
MAX_SIDE = 960


def _short(gt):
    texts = [e.get("text", "").strip() for e in gt.get("ui_elements", []) if e.get("text", "").strip()]
    return " ".join(texts)[:80]


def build(root=DATA_ROOT, train_shards=3, test_shards=1, seed=0):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    name = "rico_screenqa"
    tmp = os.path.join(root, ".tmp-" + name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    rng = random.Random(seed)
    screens = defaultdict(list)  # (split, screen) -> [(question, short answer or None)]
    for split, prefix, n_sh, total in (("train", "train", train_shards, 17), ("val", "test", test_shards, 2)):
        for i in range(n_sh):
            t = pq.read_table(hf_hub_download(REPO, "data/%s-%05d-of-%05d.parquet" % (prefix, i, total), repo_type="dataset"))
            cols = {c: t.column(c).to_pylist() for c in ("screen_id", "question", "ground_truth", "image")}
            for sid, q, gt, img in zip(cols["screen_id"], cols["question"], cols["ground_truth"], cols["image"]):
                gts = ast.literal_eval(gt) if isinstance(gt, str) else gt
                ans = None
                if gts and "<no answer>" not in str(gts[0].get("full_answer", "")):
                    ans = _short(gts[0]) or None
                key = (split, str(sid))
                p = os.path.join(tmp, "images", "screen-%s.jpg" % sid)
                if not os.path.exists(p):
                    im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                    im.thumbnail((MAX_SIDE, MAX_SIDE))
                    im.save(p, "JPEG", quality=85)
                screens[key].append((q, ans, "<no answer>" in str(gts)))
    all_answers = defaultdict(list)
    for (split, _), qs in screens.items():
        all_answers[split] += [a for _, a, _ in qs if a]
    files = {s: open(os.path.join(tmp, s + ".jsonl"), "w") for s in ("train", "val")}
    counts, labels = Counter(), defaultdict(Counter)

    def write(split, sid, question, label):
        rec = {"id": "%s-%s-%d" % (name, split, counts[split]), "image": "images/screen-%s.jpg" % sid,
               "state_text": None, "question": question, "label": label, "image_license": "CC BY 4.0 (Rico / ScreenQA)"}
        files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[split] += 1
        labels[split]["%s:%s" % (question["type"], label)] += 1

    for (split, sid), qs in sorted(screens.items()):
        same = [a for _, a, _ in qs if a]
        unans = [q for q, a, na in qs if na]
        for q, a, na in qs:
            if a:
                opts = [a] + [x for x in dict.fromkeys(same) if x != a][:2]
                while len(opts) < 4:
                    x = rng.choice(all_answers[split])
                    if x not in opts:
                        opts.append(x)
                rng.shuffle(opts)
                write(split, sid, {"type": "choice", "instructions": q, "criteria": opts}, opts.index(a))
        for q in unans:  # unanswerable, balanced by an answerable question of the same screen when there is one
            write(split, sid, {"type": "noul", "instructions": "Can this question be answered from the screen? %s" % q,
                               "criteria": None}, 0)
            ok = [q2 for q2, a2, _ in qs if a2]
            if ok:
                write(split, sid, {"type": "noul", "instructions": "Can this question be answered from the screen? %s"
                                   % rng.choice(ok), "criteria": None}, 1)
    for f in files.values():
        f.close()
    m = {"dataset": name, "source": "%s (%d train / %d test shards)" % (REPO, train_shards, test_shards), "revision": "main",
         "license": {"annotations": "CC BY 4.0 (ScreenQA)", "images": "Rico screenshots (CC BY 4.0)", "commercial": True,
                     "verified": "Hub card 2026-09-27; VERIFY Rico app-screenshot terms before release"},
         "counts": dict(counts), "label_counts": {s: dict(c) for s, c in labels.items()},
         # a few frequent questions ("Which tab is selected?") are answerable on almost every screen: a real prior
         "validator": {"noul_min_minority": 0.05}}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    final = os.path.join(root, name)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    print("%s: %s %s" % (name, dict(counts), {s: dict(c) for s, c in labels.items()}), flush=True)
    # Rico holds many identical / near-identical screens under different ids, some in both ScreenQA splits: drop val
    # rows whose screenshot matches a train screenshot (the validator's matcher)
    from .visionjev_dedupe import dedupe

    dedupe(name, root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--train-shards", type=int, default=3)
    ap.add_argument("--test-shards", type=int, default=1)
    a = ap.parse_args()
    build(a.root, a.train_shards, a.test_shards)
