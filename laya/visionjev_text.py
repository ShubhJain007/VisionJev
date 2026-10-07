"""Text-only decision data for VisionJev: the Laya typed-decisions benchmark (Apache-2.0) in the prepared format.

``LocalLLaMA/typed-decisions`` (config ``all``): workflows agent-trace observability, customer service, invoice
processing, security incidents; each row a JSON ``state`` with several typed questions (choice / noul / score) and a
gold block with a label and a probability per option. One prepared record per question, no image:

    {"id", "state_text": <the state JSON>, "question": {...}, "label": int, "target": [...], "workflow": ...}

``target`` is the gold probability vector when its argmax agrees with the gold label (else dropped: the label wins,
CONTEXT.md section 9). ``noul`` keeps its outcome descriptions (``{"false": ..., "true": ...}``, rendered by the repo).
Official ``train`` -> train, ``test`` -> val: the test split is also the comparability eval against Laya/Jev, so it is
never trained on.

VERIFY: how the gold distributions were produced (``label_agreement`` suggests several annotators or judges). Until
that is known the val split is a comparability set, not a human-labelled calibration set.

    python -m laya.visionjev_text
"""
import argparse
import json
import os
import random
import shutil
import time
from collections import Counter

from .visionjev_data import DATA_ROOT, _hub_revision

HUB_ID = "LocalLLaMA/typed-decisions"
NAME = "laya_typed"
LICENSE = {"annotations": "Apache-2.0", "images": "none (text-only)", "commercial": True,
           "verified": "HF card LocalLLaMA/typed-decisions (and mirror Luni/laya-jev-benchmark): apache-2.0"}


def records(row):
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    for qid, q in questions.items():
        g = gold.get(qid)
        if g is None:
            continue
        t, crit = q["type"], q.get("criteria")
        if t == "choice":
            opts = list(crit.keys()) if isinstance(crit, dict) else list(crit)
            label = opts.index(g["label"]) if g["label"] in opts else None
        elif t == "score":
            opts = [str(i) for i in range(len(crit))]
            label = int(g["label"]) if str(g["label"]).isdigit() else None
        else:
            opts = ["false", "true"]
            label = {"false": 0, "true": 1}.get(str(g["label"]).lower())
            crit = crit if isinstance(crit, dict) else None
        if label is None or not 0 <= label < len(opts):
            continue
        rec = {"id": "%s-%s" % (row["id"], qid), "state_text": row["state"],
               "question": {"type": t, "instructions": q["instructions"], "criteria": crit}, "label": label,
               "workflow": row["workflow"]}
        probs = g.get("probabilities") or {}
        target = [float(probs.get(o, 0.0)) for o in opts]
        if sum(target) > 0:
            total = sum(target)
            target = [p / total for p in target]
            if max(range(len(target)), key=lambda i: target[i]) == label or target[label] >= max(target) - 1e-9:
                rec["target"] = [round(p, 6) for p in target]
        if t == "choice":  # the source lists options in a fixed order, so the gold position mirrors the class prior
            perm = list(range(len(opts)))
            random.Random(rec["id"]).shuffle(perm)
            c = rec["question"]["criteria"]
            rec["question"]["criteria"] = ({opts[i]: c[opts[i]] for i in perm} if isinstance(c, dict)
                                           else [c[i] for i in perm])
            rec["label"] = perm.index(label)
            if "target" in rec:
                rec["target"] = [rec["target"][i] for i in perm]
        yield rec


def build(root: str = DATA_ROOT, log=print):
    import datasets

    revision = _hub_revision(HUB_ID)
    tmp, final = os.path.join(root, ".tmp-" + NAME), os.path.join(root, NAME)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    counts, types, soft = Counter(), Counter(), Counter()
    for hub_split, split in (("train", "train"), ("test", "val")):
        with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
            for row in datasets.load_dataset(HUB_ID, "all", split=hub_split, revision=revision):
                for rec in records(row):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    types["%s/%s" % (split, rec["question"]["type"])] += 1
                    soft[split] += "target" in rec
    manifest = {"dataset": NAME, "source": HUB_ID, "config": "all", "revision": revision, "license": LICENSE,
                # real base rates (e.g. duplicate invoices are ~10% of invoices): not a labelling artefact
                "validator": {"noul_min_minority": 0.05},
                "counts": dict(counts), "types": dict(types), "with_soft_target": dict(soft)}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if os.path.exists(final):
        old = os.path.join(root, ".old-%s-%d" % (NAME, int(time.time())))
        os.rename(final, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    log("%s: %s" % (NAME, json.dumps({k: manifest[k] for k in ("counts", "types", "with_soft_target")})))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    build(ap.parse_args().root)
