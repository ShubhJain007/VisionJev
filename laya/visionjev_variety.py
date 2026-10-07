"""Variety for generalisation (CONTEXT.md section 10): exact, label-preserving rewrites of prepared questions.

Two disjoint pools of rewrites, so the held-out evaluation measures generalisation to formulations never trained on:

* ``--mode train``  -> ``<name>_var/train.jsonl``: a share of the train rows rewritten with the TRAIN pool.
* ``--mode heldout`` -> ``<name>_hof/val.jsonl``: every val row rewritten with the HELD-OUT pool (eval only).

Rewrites (all exact; the gold is recomputed and asserted, never guessed):
* instruction wrappers (the question text is kept verbatim inside a new framing);
* fewer options for ``choice`` questions with >= 4 options (one or two distractors removed; the gold stays);
* ``score`` re-bucketing: adjacent levels merged into a coarser scale with an explicit mapping, the gold moved with it,
  new level names built from the merged ones (e.g. "safe: read-only / low: local, reversible changes").

Targets (soft labels) are merged/sliced consistently. Images are referenced from the source dataset (no copies).

    python -m laya.visionjev_variety --mode train synth_invoice synth_shell_risk laya_typed ...
    python -m laya.visionjev_variety --mode heldout synth_invoice synth_shell_risk laya_typed ...
"""
import argparse
import json
import os
import random
import shutil
import time
from typing import List, Optional, Tuple

from .visionjev_data import DATA_ROOT

WRAP = {
    "train": ["{q}", "Question: {q}", "Decide the following. {q}", "Based on the input, {ql}", "Please answer: {q}",
              "Using only the information given, {ql}"],
    "heldout": ["Your task: {q}", "Considering everything shown, {ql}", "Evaluate this carefully. {q}",
                "Given the case above, answer the question: {q}", "Short decision needed. {q}"],
}
# coarser scales (lists of groups of adjacent level indices), per number of original levels
MERGES = {
    "train": {4: [[[0], [1], [2, 3]], [[0, 1], [2, 3]]], 5: [[[0], [1, 2], [3, 4]], [[0, 1], [2], [3, 4]]]},
    "heldout": {4: [[[0, 1], [2], [3]], [[0], [1, 2, 3]]], 5: [[[0, 1, 2], [3, 4]], [[0], [1], [2], [3, 4]]]},
}


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s[:2] != s[:2].upper() else s


def rewrite(rec: dict, pool: str, rng: random.Random) -> Optional[dict]:
    q, label = dict(rec["question"]), rec["label"]
    target = rec.get("target")
    ins = q["instructions"]
    q["instructions"] = rng.choice(WRAP[pool]).format(q=ins, ql=_lower_first(ins))
    t, crit = q["type"], q.get("criteria")
    if t == "choice" and isinstance(crit, list) and len(crit) >= 4 and rng.random() < 0.6:
        drop = rng.sample([i for i in range(len(crit)) if i != label], rng.randint(1, min(2, len(crit) - 3)))
        keep = [i for i in range(len(crit)) if i not in drop]
        q["criteria"] = [crit[i] for i in keep]
        label = keep.index(label)
        if target is not None:
            kept = [target[i] for i in keep]
            total = sum(kept)
            target = [x / total for x in kept] if total > 0 else None
        assert q["criteria"][label] == crit[rec["label"]]
    elif t == "score" and isinstance(crit, list) and len(crit) in MERGES[pool] and rng.random() < 0.6:
        groups = rng.choice(MERGES[pool][len(crit)])
        assert sorted(i for g in groups for i in g) == list(range(len(crit)))
        new_label = next(j for j, g in enumerate(groups) if rec["label"] in g)
        merged = [sum(target[i] for i in g) for g in groups] if target is not None else None
        # merging can move a soft target's peak off the gold's group ([.3,.3,.4] as [[0,1],[2]] -> [.6,.4]); the coarse
        # question then has no single right answer, so keep the original scale for that row
        if merged is None or max(merged) <= merged[new_label] + 1e-9:
            q["criteria"] = [" / ".join(crit[i] for i in g) for g in groups]
            label, target = new_label, merged
            assert rec["label"] in groups[label]
    out = {k: v for k, v in rec.items() if k not in ("question", "label", "target", "image", "images")}
    out.update(question=q, label=label)
    if target is not None:
        out["target"] = list(target)  # unrounded: rounding broke exact ties between the gold and another option
    return out


def build(name: str, mode: str, root: str = DATA_ROOT, share: float = 0.3, seed: int = 0, log=print):
    src = os.path.join(root, name)
    suffix = "_var" if mode == "train" else "_hof"
    split = "train" if mode == "train" else "val"
    pool = "train" if mode == "train" else "heldout"
    rng = random.Random("%s-%s-%d" % (name, mode, seed))
    rows = [json.loads(l) for l in open(os.path.join(src, split + ".jsonl"))]
    if mode == "train":
        rows = rng.sample(rows, int(len(rows) * share))
    out_name = name + suffix
    tmp, final = os.path.join(root, ".tmp-" + out_name), os.path.join(root, out_name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    n = 0
    with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
        for i, r in enumerate(rows):
            new = rewrite(r, pool, rng)
            if new is None:
                continue
            new["id"] = "%s-%s-%d" % (out_name, split, i)
            for key in ("image",):
                if r.get(key):
                    new[key] = os.path.relpath(os.path.join(src, r[key]), final)  # reference the source image
            if r.get("images"):
                new["images"] = [os.path.relpath(os.path.join(src, p), final) for p in r["images"]]
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
            n += 1
    m = json.load(open(os.path.join(src, "manifest.json")))
    m.update({"dataset": out_name, "variety_of": name, "variety_mode": mode, "variety_pool": pool, "share": share if mode == "train" else 1.0,
              "counts": {split: n}, "eval_only": mode == "heldout"})
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    if os.path.exists(final):
        old = os.path.join(root, ".old-%s-%d" % (out_name, int(time.time())))
        os.rename(final, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    log("%s: %d %s rows (%s pool)" % (out_name, n, split, pool))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--mode", choices=["train", "heldout"], required=True)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--share", type=float, default=0.3)
    a = ap.parse_args()
    for n in a.names:
        build(n, a.mode, a.root, a.share, log=lambda m: print(m, flush=True))
