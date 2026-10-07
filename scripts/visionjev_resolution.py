"""Accuracy and latency against the image pixel budget, and a resolution cascade (low resolution first, full resolution
only when the answer is not confident). Batch 1, serial, serving fast path, held-out (val) rows.

    .venv/bin/python scripts/visionjev_resolution.py ../checkpoints/best --n 100
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
from laya.qwen_backbone import configure_processor  # noqa: E402
from laya.visionjev_log import log_experiment  # noqa: E402
from laya.vlm import VLMAgent  # noqa: E402
from laya.vlm_train import jsonl_example  # noqa: E402

GROUPS = {"documents": ["synth_invoice_doc", "cord_total", "doclaynet_read_small", "chart_qa"],
          "screens": ["ac_element", "ac_next_action", "ac_goal_reached", "ui_refexp", "rico_screenqa"],
          "photos": ["aokvqa", "oi_count", "textvqa", "vqav2_yesno"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--n", type=int, default=100, help="rows per dataset")
    ap.add_argument("--edges", default="768,640,512,384,256", help="pixel budgets as square edges, largest first")
    ap.add_argument("--tau", type=float, default=0.8, help="cascade: rerun at the largest budget below this confidence")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(WORKSPACE, "results", "latency", "resolution.json"))
    a = ap.parse_args()
    edges = [int(e) for e in a.edges.split(",")]
    agent = VLMAgent(a.checkpoint, device="cuda", dtype="bf16", fast=True, image_cache=0)  # timing every call: no cache
    trained = agent.cfg.get("qwen_max_pixels")
    rng = random.Random(a.seed)
    rows = []
    for group, sets in GROUPS.items():
        for ds in sets:
            recs = [json.loads(l) for l in open(os.path.join(a.root, ds, "val.jsonl"))]
            for rec in rng.sample(recs, min(a.n, len(recs))):
                ex = jsonl_example(rec, os.path.join(a.root, ds), ds)
                if ex is not None:
                    rows.append((group, ds, ex, rec["question"]))
    res = {e: [] for e in edges}
    for e in edges:
        configure_processor(agent.processor, e * e)
        for rep in range(2):  # the first pass records the vision graphs of this budget's grids
            out = []
            for group, ds, ex, q in rows:
                torch.cuda.synchronize()
                t = time.perf_counter()
                r = agent.predict(ex["state"], {"q": q})
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) * 1000
                ans = r["answers"]["q"]
                p = [1 - ans["noul"], ans["noul"]] if ans["type"] == "noul" else list(ans["probabilities"].values())
                pred = max(range(len(p)), key=p.__getitem__)
                out.append({"ok": pred == ex["label"], "conf": max(p), "ms": ms, "tokens": r["usage"]["input_tokens"]})
        res[e] = out
        print("edge %d: acc %.1f | p50 %.0f ms | tokens %.0f" % (e, 100 * sum(o["ok"] for o in out) / len(out),
              statistics.median(o["ms"] for o in out), statistics.mean(o["tokens"] for o in out)), flush=True)
    full = edges[0]
    table = {}
    for scope in ["all"] + list(GROUPS) + sorted({r[1] for r in rows}):
        ix = [i for i, r in enumerate(rows) if scope == "all" or r[0] == scope or r[1] == scope]
        t = {"n": len(ix)}
        for e in edges:
            o = [res[e][i] for i in ix]
            t[str(e)] = {"acc": round(100 * sum(x["ok"] for x in o) / len(o), 1), "p50_ms": round(statistics.median(x["ms"] for x in o), 1),
                         "mean_ms": round(statistics.mean(x["ms"] for x in o), 1), "tokens": round(statistics.mean(x["tokens"] for x in o))}
            if e != full:  # cascade: this budget first, the full one when not confident
                esc = [res[e][i]["conf"] < a.tau for i in ix]
                ok = [res[full][i]["ok"] if s else res[e][i]["ok"] for i, s in zip(ix, esc)]
                ms = [res[e][i]["ms"] + (res[full][i]["ms"] if s else 0) for i, s in zip(ix, esc)]
                t[str(e)]["cascade"] = {"acc": round(100 * sum(ok) / len(ok), 1), "mean_ms": round(statistics.mean(ms), 1),
                                        "rerun_share": round(sum(esc) / len(esc), 3)}
        table[scope] = t
    gpu = torch.cuda.get_device_name(0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"gpu": gpu, "trained_max_pixels": trained, "tau": a.tau, "table": table}, open(a.out, "w"), indent=1)
    log_experiment("latency", "resolution:%s" % os.path.basename(os.path.realpath(a.checkpoint)),
                   config={"gpu": gpu, "edges": edges, "n_per_dataset": a.n, "tau": a.tau, "seed": a.seed, "fast": True,
                           "trained_max_pixels": trained, "datasets": GROUPS},
                   metrics={k: table[k] for k in ["all"] + list(GROUPS)}, notes="full table: " + a.out)
    for k in ["all"] + list(GROUPS):
        print(k, json.dumps(table[k]))


if __name__ == "__main__":
    main()
