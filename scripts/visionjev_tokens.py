"""Fewer image tokens at full input resolution: token merging inside the vision tower and token pooling before the
language model (laya/qwen_tokens.py), against the plain model, on the rows of the resolution study.

    .venv/bin/python scripts/visionjev_tokens.py ../checkpoints/best --n 100
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
from laya.qwen_tokens import TokenMerging  # noqa: E402
from laya.visionjev_log import log_experiment  # noqa: E402
from laya.vlm import VLMAgent  # noqa: E402
from laya.vlm_train import jsonl_example  # noqa: E402
from visionjev_resolution import GROUPS  # noqa: E402

#: name -> (merge fraction per step or 0, merge layers, pool k)
VARIANTS = {"plain": (0, (), 1), "merge 20% x3": (0.2, (4, 10, 16), 1), "merge 30% x3": (0.3, (4, 10, 16), 1),
            "merge 40% x3": (0.4, (4, 10, 16), 1), "merge 50% at layer 2": (0.5, (2,), 1), "pool 2x2": (0, (), 2),
            "merge 30% x3 + pool 2x2": (0.3, (4, 10, 16), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(WORKSPACE, "results", "latency", "tokens.json"))
    a = ap.parse_args()
    agent = VLMAgent(a.checkpoint, device="cuda", dtype="bf16", fast=True, image_cache=0)
    enc = agent.model.encoder
    rng = random.Random(a.seed)
    rows = []
    for group, sets in GROUPS.items():
        for ds in sets:
            recs = [json.loads(l) for l in open(os.path.join(a.root, ds, "val.jsonl"))]
            for rec in rng.sample(recs, min(a.n, len(recs))):
                ex = jsonl_example(rec, os.path.join(a.root, ds), ds)
                if ex is not None:
                    rows.append((group, ex, rec["question"]))
    vis_ms = []
    inner = agent.model.encode_images

    def timed_encode(*x, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        o = inner(*x, **k)
        torch.cuda.synchronize()
        vis_ms.append((time.perf_counter() - t) * 1000)
        return o

    agent.model.encode_images = timed_encode
    table, base = {}, None
    for name, (frac, layers, pool) in VARIANTS.items():
        enc.__dict__.pop("token_merging", None)
        if frac:
            enc.token_merging = TokenMerging(enc.inner.visual, layers, frac)
        agent.token_pool = pool
        for rep in range(2):
            out = []
            vis_ms.clear()
            for group, ex, q in rows:
                torch.cuda.synchronize()
                t = time.perf_counter()
                r = agent.predict(ex["state"], {"q": q})
                torch.cuda.synchronize()
                ans = r["answers"]["q"]
                p = [1 - ans["noul"], ans["noul"]] if ans["type"] == "noul" else list(ans["probabilities"].values())
                pred = max(range(len(p)), key=p.__getitem__)
                out.append({"g": group, "ok": pred == ex["label"], "pred": pred, "ms": (time.perf_counter() - t) * 1000,
                            "tokens": r["usage"]["input_tokens"]})
        if base is None:
            base = out
        t = {"vision_p50_ms": round(statistics.median(vis_ms), 1)}
        for scope in ["all"] + list(GROUPS):
            o = [(x, b) for x, b in zip(out, base) if scope in ("all", x["g"])]
            t[scope] = {"acc": round(100 * sum(x["ok"] for x, _ in o) / len(o), 1),
                        "p50_ms": round(statistics.median(x["ms"] for x, _ in o), 1),
                        "tokens": round(statistics.mean(x["tokens"] for x, _ in o)),
                        "answers_changed": round(100 * sum(x["pred"] != b["pred"] for x, b in o) / len(o), 1)}
        table[name] = t
        print(name, json.dumps(t), flush=True)
    gpu = torch.cuda.get_device_name(0)
    json.dump({"gpu": gpu, "n_per_dataset": a.n, "seed": a.seed, "table": table}, open(a.out, "w"), indent=1)
    log_experiment("latency", "token-reduction:%s" % os.path.basename(os.path.realpath(a.checkpoint)),
                   config={"gpu": gpu, "variants": {k: list(map(str, v)) for k, v in VARIANTS.items()}, "n_per_dataset": a.n,
                           "seed": a.seed, "fast": True, "datasets": GROUPS, "retrained": False},
                   metrics=table, notes="laya/qwen_tokens.py; no retraining")


if __name__ == "__main__":
    main()
