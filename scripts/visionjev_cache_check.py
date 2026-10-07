"""The image cache: same answers as without it, and what a second question about the same image costs.

For each sample image three calls are timed with the cache off and with it on: the item's own question, a different
question, and the first question again. With the cache on, calls 2 and 3 reuse the encoded image.

    .venv/bin/python scripts/visionjev_cache_check.py ../checkpoints/best
"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
from laya.visionjev_log import log_experiment  # noqa: E402
from laya.vlm import VLMAgent  # noqa: E402

OTHER = {"type": "noul", "instructions": "Is any text visible in this image?", "criteria": None}


def main():
    ckpt = sys.argv[1]
    data = json.load(open(os.path.join(HERE, "..", "jobs", "demo_samples.json")))
    rows = [r for t in ("documents", "android") for r in data["tasks"][t]]
    agent = VLMAgent(ckpt, device="cuda", dtype="bf16", fast=True, image_cache=0)

    def run():
        probs, times = [], []
        for r in rows:
            state = {"image": os.path.join(WORKSPACE, "data", "vqa", r["image"].split("/data/vqa/", 1)[1])}
            if r["context"]:
                state["context"] = r["context"]
            q = {"type": r["type"], "instructions": r["question"],
                 "criteria": None if r["type"] == "noul" else (r["prob_keys"] if r["type"] == "choice" else r["options"])}
            for qq in (q, OTHER, q):
                torch.cuda.synchronize()
                t = time.perf_counter()
                a = agent.predict(state, {"q": qq})["answers"]["q"]
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t) * 1000)
                probs.append([a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values()))
        return probs, times

    run()  # warm-up
    p0, t0 = run()
    agent.image_cache_size = 32
    p1, t1 = run()
    med = lambda xs: round(statistics.median(xs), 1)  # noqa: E731
    out = {"images": len(rows), "identical_answers": p0 == p1,
           "max_prob_diff": max(abs(x - y) for a, b in zip(p0, p1) for x, y in zip(a, b)),
           "no_cache_ms": {"first_question": med(t0[0::3]), "different_question": med(t0[1::3]), "repeat": med(t0[2::3])},
           "cache_ms": {"first_question": med(t1[0::3]), "different_question": med(t1[1::3]), "repeat": med(t1[2::3])},
           "cache_stats": dict(agent.image_cache_stats), "entries": len(agent.image_cache),
           "gpu_mb_per_entry": round(sum(f.numel() * f.element_size() for f in [next(iter(agent.image_cache.values()))[1][0]]
                                         + list(next(iter(agent.image_cache.values()))[1][1])) / 1e6, 1)}
    print(json.dumps(out, indent=1))
    log_experiment("latency", "image-cache:%s" % os.path.basename(os.path.realpath(ckpt)),
                   config={"gpu": torch.cuda.get_device_name(0), "fast": True, "cache_size": 32, "calls_per_image": 3},
                   metrics=out, notes="scripts/visionjev_cache_check.py")


if __name__ == "__main__":
    main()
