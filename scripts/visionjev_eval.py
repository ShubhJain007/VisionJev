"""Evaluate saved checkpoints on prepared val sets (e.g. zero-shot on sets never trained on) and log everything.

Per checkpoint and dataset, with the checkpoint's own temperatures: accuracy, ECE, NLL, KL from the (soft) gold,
multiclass Brier, selective accuracy at 60% / 20% coverage, and (image sets) the shuffled-image control. Each
checkpoint's results are appended to ``results/experiments.jsonl`` (``laya.visionjev_log``) and written to
``results/evals/<tag>.json``.

    .venv/bin/python scripts/visionjev_eval.py ../checkpoints/qwen2b-mix ../checkpoints/mix-split1024 \\
        --datasets openimages_presence,openimages_count,mind2web_element,mind2web_operation --tag zeroshot-oi-m2w
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from laya.visionjev_log import log_experiment  # noqa: E402
from laya.vlm import VLMAgent  # noqa: E402
from laya.vlm_train import collect_logits, load_jsonl_examples, metrics_from  # noqa: E402
from visionjev_finetune import selective, shuffled_images  # noqa: E402


def soft_metrics(records, T):
    out = {}
    for name in sorted({r["dataset"] for r in records}) + ["all"]:
        kl = br = n = 0
        for r in records:
            if name != "all" and r["dataset"] != name:
                continue
            p = torch.softmax(r["logits"] / T[r["qtype"]], -1)
            t = r["target"] / r["target"].sum()
            kl += float((t * (torch.log(t.clamp_min(1e-12)) - torch.log(p.clamp_min(1e-12)))).sum())
            br += float(((p - t) ** 2).sum())
            n += 1
        out[name] = {"kl": kl / max(1, n), "brier": br / max(1, n)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--kind", default="eval", help="log kind, e.g. zero_shot")
    ap.add_argument("--max-per-dataset", type=int, default=0)
    ap.add_argument("--no-shuffle-control", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args(argv)
    names = [d for d in a.datasets.split(",") if d]
    examples = sum((load_jsonl_examples(a.root, n, a.split, limit=a.max_per_dataset or None) for n in names), [])
    has_images = [ex for ex in examples if isinstance(ex["state"], dict) and ex["state"].get("image")]
    os.makedirs(os.path.join(WORKSPACE, "results", "evals"), exist_ok=True)
    report = {}
    for ckpt in a.checkpoints:
        run = os.path.basename(os.path.normpath(ckpt))
        agent = VLMAgent(ckpt, device="cuda")
        T = list(agent.cfg.get("temperature") or [1.0, 1.0, 1.0])[:3]
        bs = 4 if agent.model.is_qwen else max(2, 32 // agent.prep.max_tiles)
        kw = dict(batch_size=bs, num_workers=a.num_workers)
        recs = collect_logits(agent.model, agent.processor, examples, **kw)
        res = {"temperature": T, "raw": metrics_from(recs), "calibrated": metrics_from(recs, T),
               "soft": soft_metrics(recs, T), "selective_calibrated": selective(recs, T)}
        if has_images and not a.no_shuffle_control:
            res["shuffled_images"] = metrics_from(collect_logits(agent.model, agent.processor,
                                                                 shuffled_images(has_images), **kw), T)
        report[run] = res
        print("\n== %s (%s)" % (run, a.tag))
        for ds in names + ["all"]:
            c, s = res["calibrated"].get(ds), res["soft"].get(ds, {})
            if not c:
                continue
            sh = res.get("shuffled_images", {}).get(ds)
            sel = res["selective_calibrated"].get(ds, {})
            print("  %-22s n=%5d acc %5.1f%%  ece %.3f  kl %.3f  brier %.3f  @60%% %5.1f%%  shuffled %s" % (
                ds, c["n"], 100 * c["acc"], c["ece"], s.get("kl", float("nan")), s.get("brier", float("nan")),
                100 * sel.get("60%", {}).get("acc", float("nan")), "%.1f%%" % (100 * sh["acc"]) if sh else "–"))
        out = os.path.join(WORKSPACE, "results", "evals", "%s-%s.json" % (a.tag, run))
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        log_experiment(a.kind, "%s:%s" % (a.tag, run), config=vars(a), metrics=res, datasets=names,
                       checkpoint=ckpt, artifacts=[out])
        del agent
        torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    main()
