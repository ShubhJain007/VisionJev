"""Mind2Web (data CC BY 4.0) as zero-shot text decision sets for web agents (router / agent-monitor use cases).

Each action step of a task becomes, with the task and the actions taken so far as the state:

* ``mind2web_element`` (choice, 4 options): "Which page element should the agent interact with next?" — the
  annotated target element and 3 real non-target elements of the same page (distractors from the page, never
  generated; informative elements preferred so the choice is not trivial), rendered as ``<tag> key text``.
* ``mind2web_operation`` (choice): "Which operation should the agent perform next?" over CLICK / TYPE / SELECT.

Element text comes from the step's cleaned HTML (the text right after the element's ``backend_node_id``).

    python -m laya.visionjev_mind2web --tasks 250
"""
import argparse
import json
import os
import random
import re
import shutil
import time
from collections import Counter

from .visionjev_data import DATA_ROOT, _hub_revision

HUB_ID = "osunlp/Mind2Web"
LICENSE = {"annotations": "CC BY 4.0", "images": "none (text-only; HTML of third-party sites)", "commercial": True,
           "verified": "HF card osunlp/Mind2Web license: cc-by-4.0; code MIT (GitHub), 2026-09-25"}
KEYS = ("aria_label", "aria-label", "placeholder", "title", "alt", "name", "value", "type", "role", "id")
OPS = ["CLICK", "TYPE", "SELECT"]


def _text(html: str, node: str) -> str:
    m = re.search(r'backend_node_id="%s"[^>]*>([^<]{1,120})' % re.escape(node), html or "")
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _render(cand, html: str) -> str:
    try:
        attrs = json.loads(cand["attributes"])
    except Exception:
        attrs = {}
    parts = [_text(html, cand["backend_node_id"])]
    parts += ["%s=%s" % (k, str(attrs[k])[:40]) for k in KEYS if attrs.get(k)]
    body = " ".join(p for p in parts if p)[:90]
    return ("<%s> %s" % (cand["tag"], body)).strip()


def build(root: str = DATA_ROOT, n_tasks: int = 250, seed: int = 0, log=print):
    import datasets

    rng = random.Random(seed)
    revision = _hub_revision(HUB_ID)
    names = ("mind2web_element", "mind2web_operation")
    tmp = {n: os.path.join(root, ".tmp-" + n) for n in names}
    for d in tmp.values():
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    files = {n: open(os.path.join(tmp[n], "val.jsonl"), "w") for n in names}
    counts, skipped, labels = Counter(), Counter(), {n: Counter() for n in names}
    for t, ex in enumerate(datasets.load_dataset(HUB_ID, split="train", streaming=True, revision=revision)):
        if t >= n_tasks:
            break
        for i, act in enumerate(ex["actions"]):
            state = json.dumps({"website": ex["website"], "task": ex["confirmed_task"],
                                "previous_actions": ex["action_reprs"][:i]}, ensure_ascii=False)
            base = {"state_text": state, "workflow": ex["domain"]}
            op = act["operation"]["op"]
            if op in OPS:
                opts = list(OPS)
                rng.shuffle(opts)
                rec = dict(base, id="%s-%d-op" % (ex["annotation_id"], i), label=opts.index(op),
                           question={"type": "choice", "instructions": "Which operation should the agent perform next?",
                                     "criteria": opts})
                files["mind2web_operation"].write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["mind2web_operation"] += 1
                labels["mind2web_operation"][op] += 1
            pos = [c for c in act["pos_candidates"] if c.get("is_original_target")] or act["pos_candidates"]
            if not pos:
                skipped["no_target"] += 1
                continue
            html = act.get("cleaned_html") or ""
            gold = _render(pos[0], html)
            negs = {}
            for c in act["neg_candidates"]:
                r = _render(c, html)
                if r != gold and len(r) > len(c["tag"]) + 4:  # informative: more than a bare tag
                    negs.setdefault(r, c)
            if len(negs) < 3 or len(gold) <= len(pos[0]["tag"]) + 3:
                skipped["too_few_informative"] += 1
                continue
            opts = [gold] + rng.sample(sorted(negs), 3)
            rng.shuffle(opts)
            rec = dict(base, id="%s-%d-el" % (ex["annotation_id"], i), label=opts.index(gold),
                       question={"type": "choice", "instructions": "Which page element should the agent interact with next?",
                                 "criteria": opts})
            files["mind2web_element"].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["mind2web_element"] += 1
            labels["mind2web_element"][rec["label"]] += 1
    for f in files.values():
        f.close()
    for n in names:
        m = {"dataset": n, "source": HUB_ID, "split": "train (none of it is trained on: zero-shot eval)",
             "revision": revision, "license": LICENSE, "tasks": n_tasks, "seed": seed, "counts": {"val": counts[n]},
             "label_counts": {str(k): v for k, v in labels[n].items()}, "skipped": dict(skipped), "zero_shot_only": True,
             # operation labels follow the real mix (CLICK dominates web actions)
             "validator": {"noul_min_minority": 0.05}}
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
    log("mind2web: %s questions, skipped %s, labels %s" % (dict(counts), dict(skipped),
                                                           {k: dict(v) for k, v in labels.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--tasks", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.root, a.tasks, a.seed)
