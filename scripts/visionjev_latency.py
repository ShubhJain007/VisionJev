"""Latency of saved VisionJev checkpoints: ``VLMAgent.predict`` on real val pages, bf16, one image per call.

Every page is asked the questions its prepared records carry (grouped by image), so a call is what the product would
send: one encoded image, several typed questions. Reports median / p90 ms and image views per call per checkpoint.

    .venv/bin/python scripts/visionjev_latency.py ../checkpoints/doc-split0 ../checkpoints/doc-split1024 \\
        --datasets doclaynet_type,doclaynet_layout,doclaynet_read_small,doclaynet_read_large --n 100
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
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

from laya.visionjev_log import log_experiment  # noqa: E402
from laya.vlm import VLMAgent  # noqa: E402


def pages(root, names, n):
    by_image = defaultdict(dict)
    for name in names:
        base = os.path.join(root, name)
        with open(os.path.join(base, "val.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                img = os.path.normpath(os.path.join(base, r["image"]))
                by_image[img]["%s-%s" % (name, r["id"])] = r["question"]
    return list(by_image.items())[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--datasets", default="doclaynet_type,doclaynet_layout,doclaynet_read_small,doclaynet_read_large")
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args(argv)
    from PIL import Image

    work = pages(a.root, [d for d in a.datasets.split(",") if d], a.n + a.warmup)
    out = {}
    for ckpt in a.checkpoints:
        agent = VLMAgent(ckpt, device="cuda", dtype=torch.bfloat16)
        ts, views, nq = [], [], []
        for i, (img, qs) in enumerate(work):
            image = Image.open(img).convert("RGB")
            torch.cuda.synchronize()
            t = time.perf_counter()
            res = agent.predict({"image": image}, qs)
            torch.cuda.synchronize()
            if i >= a.warmup:
                ts.append((time.perf_counter() - t) * 1000)
                views.append(res["usage"]["input_tokens"])
                nq.append(len(qs))
        ts.sort()
        out[ckpt] = {"split_edge": agent.prep.split_edge, "median_ms": statistics.median(ts),
                     "p90_ms": ts[int(0.9 * (len(ts) - 1))], "questions_per_call": statistics.mean(nq),
                     "input_tokens_per_call": statistics.mean(views), "n": len(ts)}
        print("%-60s split %4d  median %6.1f ms  p90 %6.1f ms  (%.1f questions, %.0f input tokens per call)"
              % (ckpt, agent.prep.split_edge, out[ckpt]["median_ms"], out[ckpt]["p90_ms"],
                 out[ckpt]["questions_per_call"], out[ckpt]["input_tokens_per_call"]), flush=True)
        del agent
        torch.cuda.empty_cache()
    log_experiment("latency", "latency-" + "+".join(os.path.basename(os.path.normpath(c)) for c in a.checkpoints),
                   config=vars(a), metrics=out, datasets=[d for d in a.datasets.split(",") if d],
                   notes="bf16, one image per call, all of its prepared questions in one predict()")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
