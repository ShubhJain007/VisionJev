"""Fast path vs the eager model on the same checkpoint: decisions, probabilities and latency (batch 1, serial).

    .venv/bin/python scripts/visionjev_fast_check.py ../checkpoints/best [--jevbench ../../external/jevbench]
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
from laya.vlm import VLMAgent  # noqa: E402
from laya.visionjev_log import log_experiment  # noqa: E402


def requests(jevbench, tiers=("easy", "original", "hard")):
    path = os.path.join(WORKSPACE, "results", "demos", "samples.json")
    data = json.load(open(path if os.path.exists(path) else os.path.join(HERE, "..", "jobs", "demo_samples.json")))
    for rows in data["tasks"].values():  # image paths were recorded on the workstation: re-root them under this workspace
        for r in rows:
            if r["image"]:
                r["image"] = os.path.join(WORKSPACE, "data", "vqa", r["image"].split("/data/vqa/", 1)[1])
    for group, tasks in (("text", ("guard", "tickets")), ("image", ("documents", "android"))):
        for t in tasks:
            for r in data["tasks"][t]:
                state = {k: v for k, v in (("image", r["image"]), ("context", r["context"])) if v}
                q = {"type": r["type"], "instructions": r["question"],
                     "criteria": None if r["type"] == "noul" else (r["prob_keys"] if r["type"] == "choice" else r["options"])}
                yield group, state or "", q
    if jevbench:
        for tier in tiers:
            for line in open(os.path.join(jevbench, "datasets", "public", tier + ".jsonl")):
                it = json.loads(line)
                yield "jevbench_" + tier, it["state"], it["question"]


def run(agent, reqs, reps):
    probs, times = [], [[] for _ in reqs]
    for rep in range(reps + 1):
        for i, (_, state, q) in enumerate(reqs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            a = agent.predict(state, {"q": q})["answers"]["q"]
            torch.cuda.synchronize()
            if rep:
                times[i].append((time.perf_counter() - t) * 1000)
            elif True:
                probs.append([1 - a["noul"], a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values()))
    return probs, [min(t) for t in times]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--jevbench", default="")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tiers", default="easy,original,hard", help="JevBench tiers (hard needs more than 16 GB)")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    reqs = list(requests(a.jevbench, a.tiers.split(",")))
    agent = VLMAgent(a.checkpoint, device="cuda", dtype="bf16", image_cache=0)  # every call pays the vision tower
    p0, t0 = run(agent, reqs, a.reps)
    t = time.perf_counter()
    info = agent.enable_fast(merge=not a.no_merge)
    info["setup_s"] = round(time.perf_counter() - t, 1)
    print("fast path:", json.dumps(info), flush=True)
    p1, t1 = run(agent, reqs, a.reps)
    out = {}
    for g in dict.fromkeys(r[0] for r in reqs):
        ix = [i for i, r in enumerate(reqs) if r[0] == g]
        flips = [i for i in ix if int(np.argmax(p0[i])) != int(np.argmax(p1[i]))]
        diff = [float(np.abs(np.array(p0[i]) - np.array(p1[i])).max()) for i in ix]
        med = lambda xs: round(statistics.median(xs), 1)  # noqa: E731
        p95 = lambda xs: round(sorted(xs)[int(.95 * (len(xs) - 1))], 1)  # noqa: E731
        out[g] = {"n": len(ix), "eager_p50_ms": med([t0[i] for i in ix]), "fast_p50_ms": med([t1[i] for i in ix]),
                  "eager_p95_ms": p95([t0[i] for i in ix]), "fast_p95_ms": p95([t1[i] for i in ix]),
                  "flips": len(flips), "flip_eager_top_prob": [round(max(p0[i]), 2) for i in flips],
                  "max_prob_diff": round(max(diff), 4), "p90_prob_diff": round(sorted(diff)[int(.9 * (len(diff) - 1))], 4)}
        print(g, json.dumps(out[g]), flush=True)
    gpu = torch.cuda.get_device_name(0)
    print("gpu", gpu, "| max memory %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    log_experiment("latency", "fast-path:%s" % os.path.basename(os.path.realpath(a.checkpoint)),
                   config={"gpu": gpu, "dtype": "bf16", "merge_lora": not a.no_merge, "fast": info, "reps": a.reps,
                           "batch": 1, "timing": "min of reps per request, median over requests"},
                   metrics=out, notes="eager vs laya.qwen_fast on the same loaded checkpoint")


if __name__ == "__main__":
    main()
