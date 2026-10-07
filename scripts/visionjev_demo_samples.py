"""Sample decisions for the agent demos (results/demos/samples.json): a seeded random draw of held-out (val) rows per
demo task, scored by one checkpoint with per-decision latency. Nothing is cherry-picked: wrong answers stay in.

    .venv/bin/python scripts/visionjev_demo_samples.py ../checkpoints/best --n 12
"""
import argparse
import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
from laya.vlm import VLMAgent, render_options  # noqa: E402
from laya.vlm_train import jsonl_example  # noqa: E402

TASKS = {  # demo task -> datasets drawn in equal parts
    "documents": ["synth_invoice_doc", "synth_refund_doc", "cord_total"],
    "android": ["ac_next_action", "ac_element", "ac_goal_reached"],
    "guard": ["synth_shell_risk", "prompt_injection"],
    "tickets": ["banking77", "support_verticals"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--n", type=int, default=12, help="rows per task")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--out", default=os.path.join(WORKSPACE, "results", "demos", "samples.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    agent = VLMAgent(a.checkpoint, device=a.device, dtype="bf16" if a.device == "cuda" else None)
    out = {"checkpoint": os.path.basename(os.path.realpath(a.checkpoint)), "seed": a.seed, "device": a.device, "tasks": {}}
    warm = True
    for task, sets in TASKS.items():
        rows = []
        for ds in sets:
            recs = [json.loads(l) for l in open(os.path.join(a.root, ds, "val.jsonl"))]
            for rec in rng.sample(recs, a.n // len(sets)):
                ex = jsonl_example(rec, os.path.join(a.root, ds), ds)
                if ex is None:
                    continue
                for _ in range(2 if warm else 1):  # the very first call pays one-off set-up
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    t = time.perf_counter()
                    res = agent.predict(ex["state"], {"q": rec["question"]})
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    ms = (time.perf_counter() - t) * 1000
                warm = False
                opts = [str(o) for o in render_options(ex["q"])]
                ans = res["answers"]["q"] if "answers" in res else res["q"]
                if ans["type"] == "noul":
                    opts, pr = ["no", "yes"], None
                    probs = [1.0 - float(ans["noul"]), float(ans["noul"])]
                else:
                    pr = ans["probabilities"]
                    probs = [float(pr[k]) for k in pr]
                    assert len(probs) == len(opts), (ds, list(pr), opts)
                pred = max(range(len(probs)), key=probs.__getitem__)
                rows.append({"id": rec["id"], "dataset": ds, "image": os.path.join(a.root, ds, rec["image"]) if rec.get("image") else None,
                             "context": rec.get("state_text"), "question": rec["question"]["instructions"], "type": rec["question"]["type"],
                             "options": opts, "prob_keys": list(pr) if pr else None, "probs": probs, "unknown": ans.get("unknown"), "pred": pred,
                             "label": int(rec["label"]), "correct": pred == int(rec["label"]), "confidence": max(probs),
                             "latency_ms": round(ms, 1), "input_tokens": res.get("usage", {}).get("input_tokens"),
                             "image_license": rec.get("image_license")})
        rng.shuffle(rows)
        out["tasks"][task] = rows
        print(task, "n", len(rows), "acc %.2f" % (sum(r["correct"] for r in rows) / len(rows)),
              "median ms %.0f" % sorted(r["latency_ms"] for r in rows)[len(rows) // 2], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
