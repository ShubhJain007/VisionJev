"""Robustness and consistency evals (``laya.robustness``) on VisionJev val sets and local checkpoints, logged.

For ``--n`` seeded val rows per dataset, the meaning-preserving variants of each family are scored with the
checkpoint's temperatures and compared with the unperturbed row (flip rate, accuracy / ECE deltas with cluster
bootstrap intervals):

* ``option_order``  the same options in other orders (answer should not move)
* ``text``          rule-based rewordings (prefix, suffix, spacing, case, punctuation, noul framing)
* ``image``         JPEG 70/40/20, centre / random crop, downscale x0.5, brightness +-10%
* ``image_shuffle`` / ``text_only``  controls: does the model use the image?
* ``negation``      (consistency) a negated frame of a noul question: P(yes | Q) + P(yes | not Q) should be ~1
* ``form_choice``   (consistency) a noul question re-asked as a two-option choice: same answer?

Opt-in: ``option_set``, ``abstain``, ``inject_text``, ``inject_image`` (see ``laya.robustness``). Per checkpoint the
row-level predictions and the summary are written to ``results/robustness/<tag>-<run>/`` (a new directory; an
existing one is refused) and the compact summary is appended to ``results/experiments.jsonl``.

    .venv/bin/python scripts/visionjev_robustness.py ../checkpoints/mix-split1024 ../checkpoints/qwen2b-mix \\
        --datasets aokvqa,vqav2_yesno,doclaynet_type,laya_typed --n 200 --tag robust-v1
"""
import argparse
import gzip
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

from laya import robustness as R  # noqa: E402
from laya.visionjev_log import log_experiment  # noqa: E402
from laya.vlm import VLMAgent  # noqa: E402
from laya.vlm_train import load_jsonl_examples  # noqa: E402

DEFAULT_FAMILIES = list(R.FAMILIES) + ["negation", "form_choice"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    ap.add_argument("--n", type=int, default=200, help="source rows per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--tag", default="robust")
    ap.add_argument("--num-workers", type=int, default=6)
    a = ap.parse_args(argv)
    names = [d for d in a.datasets.split(",") if d]
    fams = [f for f in a.families.split(",") if f]
    rows = []
    for name in names:
        rows += R.source_rows(load_jsonl_examples(a.root, name, a.split), n=a.n, seed=a.seed, dataset=name)
    variants = R.build_variants(rows, fams, a.seed)
    counts = {}
    for v in variants:
        counts[v["family"]] = counts.get(v["family"], 0) + 1
    print("%d source rows, %d rows to score: %s" % (len(rows), len(variants), counts), flush=True)
    for ckpt in a.checkpoints:
        run = os.path.basename(os.path.normpath(ckpt))
        out_dir = os.path.join(WORKSPACE, "results", "robustness", "%s-%s" % (a.tag, run))
        if os.path.exists(out_dir):
            raise SystemExit("%s exists; pass a new --tag" % out_dir)
        t0 = time.time()
        agent = VLMAgent(ckpt, device="cuda")
        T = list(agent.temperature)
        bs = 4 if agent.model.is_qwen else max(2, 32 // agent.prep.max_tiles)
        preds = R.score_rows(agent.model, agent.processor, variants, T, batch_size=bs, num_workers=a.num_workers)
        summary = R.summarize(preds, a.n_boot, a.seed)
        print("\n== %s (%s, %.1f min)" % (run, a.tag, (time.time() - t0) / 60))
        print(R.format_table(summary), flush=True)
        os.makedirs(out_dir)
        with gzip.open(os.path.join(out_dir, "predictions.jsonl.gz"), "wt") as f:
            for p in preds:
                f.write(json.dumps(p, sort_keys=True) + "\n")
        meta = {"checkpoint": ckpt, "datasets": names, "n_per_dataset": a.n, "families": fams, "seed": a.seed,
                "temperature": T, "row_counts": counts, "gpu": torch.cuda.get_device_name(0),
                "minutes": round((time.time() - t0) / 60, 1)}
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({"meta": meta, **summary}, f, indent=1)
        log_experiment("robustness", "%s:%s" % (a.tag, run), config=dict(vars(a), families=fams),
                       metrics={"compact": R.compact(summary), "meta": meta}, datasets=names, checkpoint=ckpt,
                       artifacts=[os.path.join(out_dir, "summary.json"), os.path.join(out_dir, "predictions.jsonl.gz")])
        del agent
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
