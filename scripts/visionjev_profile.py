"""Where one decision's time goes (batch 1, serial): image preprocessing, vision tower, input building, language model,
head. Rows come from results/demos/samples.json (held-out text and image decisions).

    .venv/bin/python scripts/visionjev_profile.py ../checkpoints/best [--fast]
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
import laya.vlm as V  # noqa: E402

T = defaultdict(list)


def timed(name, fn):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        T[name].append((time.perf_counter() - t) * 1000)
        return out
    return wrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    path = os.path.join(WORKSPACE, "results", "demos", "samples.json")
    data = json.load(open(path if os.path.exists(path) else os.path.join(HERE, "..", "jobs", "demo_samples.json")))
    for rows in data["tasks"].values():  # image paths were recorded on the workstation: re-root them under this workspace
        for r in rows:
            if r["image"]:
                r["image"] = os.path.join(WORKSPACE, "data", "vqa", r["image"].split("/data/vqa/", 1)[1])
    agent = V.VLMAgent(a.checkpoint, device="cuda", dtype="bf16", image_cache=0, **({"fast": True} if a.fast else {}))
    V.vlm_prefix = timed("1 image preprocessing (CPU)", V.vlm_prefix)
    V.build_vlm_inputs = timed("3 tokenize + build inputs (CPU)", V.build_vlm_inputs)
    agent.model.encode_images = timed("2 vision tower", agent.model.encode_images)
    enc = agent.model.encoder
    enc.forward = timed("4 language model", enc.forward)
    agent.model.forward = timed("4+5 model forward (LM + head)", agent.model.forward)
    out = {}
    for group, tasks in (("text", ("guard", "tickets")), ("image", ("documents", "android"))):
        rows = [r for t in tasks for r in data["tasks"][t]]
        T.clear()
        total = []
        for rep in range(a.reps + 1):
            for r in rows:
                state = {}
                if r["image"]:
                    state["image"] = r["image"]
                if r["context"]:
                    state["context"] = r["context"]
                q = {"type": r["type"], "instructions": r["question"],
                     "criteria": None if r["type"] == "noul" else (r["prob_keys"] if r["type"] == "choice" else r["options"])}
                torch.cuda.synchronize()
                t = time.perf_counter()
                agent.predict(state or "", {"q": q})
                torch.cuda.synchronize()
                if rep:
                    total.append((time.perf_counter() - t) * 1000)
            if not rep:
                T.clear()
        med = lambda xs: round(statistics.median(xs), 1)  # noqa: E731
        out[group] = {"total_p50": med(total), "total_p95": round(sorted(total)[int(.95 * (len(total) - 1))], 1),
                      **{k: med(v) for k, v in sorted(T.items())}}
        print(group, json.dumps(out[group]), flush=True)
    print("gpu", torch.cuda.get_device_name(0), "| max memory %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))


if __name__ == "__main__":
    main()
