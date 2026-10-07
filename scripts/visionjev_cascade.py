"""Tier-0 / tier-1 cascade analysis: how much can escalating from a small to a large decision model gain?

Runs both checkpoints on the same val questions (calibrated with each checkpoint's own temperatures), caches the
per-question results, and reports per dataset:

* the 2 x 2 table (both right / only small / only large / both wrong) -> the **upper bound**: escalating can only
  ever help on "only large" questions;
* the **oracle cascade** (escalate exactly those): the best any escalation policy can reach;
* the **threshold cascade**: escalate when the small model's calibrated confidence < tau, for a grid of tau:
  accuracy, escalation rate, mean and p90 latency (each question pays t_small, escalated ones also t_large);
* a **learned deferral** (supervised, CONTEXT.md "learning to defer"): a logistic model on the small model's own
  outputs (confidence, margin, normalised entropy, option count, question type) predicting P(large right) -
  P(small right), 2-fold cross-fitted so no question is scored by a model that saw it; escalate when the
  predicted gain exceeds a cost. Compared with the threshold at equal escalation rates.

    .venv/bin/python scripts/visionjev_cascade.py --small ../checkpoints/mix-split1024 --large ../checkpoints/qwen2b-mix \\
        --t-small 40 --t-large 120 --out ../results/cascade
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

from laya.visionjev_log import log_experiment  # noqa: E402
from laya.vlm import VLMAgent  # noqa: E402
from laya.vlm_train import collect_logits, load_jsonl_examples  # noqa: E402

DEFAULT_SETS = "aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,doclaynet_read_small,doclaynet_read_large,laya_typed"


def run_model(path, examples, cache):
    if os.path.exists(cache):
        with open(cache) as f:
            return [dict(r, p=np.array(r["p"])) for r in json.load(f)]
    agent = VLMAgent(path, device="cuda")
    T = list(agent.cfg.get("temperature") or [1.0, 1.0, 1.0])[:3]
    bs = max(2, 32 // max(1, agent.prep.max_tiles)) if not agent.model.is_qwen else 4
    recs = collect_logits(agent.model, agent.processor, examples, batch_size=bs, num_workers=4)
    out = []
    for r in recs:
        p = torch.softmax(r["logits"] / T[r["qtype"]], -1)
        out.append({"p": p.numpy(), "label": int(r["label"]), "qtype": int(r["qtype"]), "dataset": r["dataset"]})
    with open(cache, "w") as f:  # plain JSON: rerunning the analysis never needs the GPU again
        json.dump([dict(r, p=[float(x) for x in r["p"]]) for r in out], f)
    del agent
    torch.cuda.empty_cache()
    return out


def feats(r):
    p = np.sort(r["p"])[::-1]
    k = len(p)
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum() / math.log(k)
    return [p[0], p[0] - p[1], ent, 1.0 / k] + [float(r["qtype"] == t) for t in range(3)]


def fit_logistic(X, y, w=None, steps=500, l2=1e-3):
    X = torch.tensor(np.c_[X, np.ones(len(X))], dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    theta = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=0.5, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ theta, y) + l2 * (theta[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach()


def predict_logistic(theta, X):
    X = torch.tensor(np.c_[X, np.ones(len(X))], dtype=torch.float32)
    return torch.sigmoid(X @ theta).numpy()


def curve(conf_order_key, s_right, l_right, t_s, t_l, rates=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)):
    """Escalate the fraction `rate` of questions with the highest `conf_order_key` (higher = escalate first)."""
    order = np.argsort(-conf_order_key)
    n = len(order)
    rows = []
    for rate in rates:
        m = int(round(rate * n))
        esc = np.zeros(n, bool)
        esc[order[:m]] = True
        acc = np.where(esc, l_right, s_right).mean()
        lat = t_s + esc * t_l
        rows.append({"escalated": rate, "acc": float(acc), "mean_ms": float(lat.mean()),
                     "p90_ms": float(np.percentile(lat, 90))})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--datasets", default=DEFAULT_SETS)
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--t-small", type=float, default=40.0, help="ms per question, tier 0")
    ap.add_argument("--t-large", type=float, default=120.0, help="ms per question, tier 1")
    ap.add_argument("--out", default=os.path.join(WORKSPACE, "results", "cascade"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    names = [d for d in a.datasets.split(",") if d]
    examples = sum((load_jsonl_examples(a.root, n, "val") for n in names), [])
    small = run_model(a.small, examples, os.path.join(a.out, "small-%s.json" % os.path.basename(os.path.normpath(a.small))))
    large = run_model(a.large, examples, os.path.join(a.out, "large-%s.json" % os.path.basename(os.path.normpath(a.large))))
    assert len(small) == len(large) == len(examples)

    s_right = np.array([int(np.argmax(r["p"])) == r["label"] for r in small])
    l_right = np.array([int(np.argmax(r["p"])) == r["label"] for r in large])
    s_conf = np.array([float(np.max(r["p"])) for r in small])
    ds = np.array([r["dataset"] for r in small])

    # learned deferral: 2-fold cross-fitted P(large right) and P(small right) from the small model's outputs
    X = np.array([feats(r) for r in small])
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 2, len(X))
    gain = np.zeros(len(X))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        th_l = fit_logistic(X[tr], l_right[tr].astype(float))
        th_s = fit_logistic(X[tr], s_right[tr].astype(float))
        gain[te] = predict_logistic(th_l, X[te]) - predict_logistic(th_s, X[te])

    report = {"t_small_ms": a.t_small, "t_large_ms": a.t_large, "datasets": {}}
    lines = ["# Cascade: %s -> %s" % (os.path.basename(os.path.normpath(a.small)), os.path.basename(os.path.normpath(a.large))),
             "", "Latency model: every question pays %.0f ms; escalated ones also %.0f ms." % (a.t_small, a.t_large), "",
             "| dataset | n | small acc | large acc | only-large (upper bound) | only-small | oracle cascade |",
             "|---|---|---|---|---|---|---|"]
    for name in names + ["all"]:
        m = np.ones(len(ds), bool) if name == "all" else ds == name
        sr, lr_ = s_right[m], l_right[m]
        d = {"n": int(m.sum()), "small_acc": float(sr.mean()), "large_acc": float(lr_.mean()),
             "only_large": float((~sr & lr_).mean()), "only_small": float((sr & ~lr_).mean()),
             "oracle_acc": float((sr | lr_).mean()),
             "threshold_curve": curve(-s_conf[m], sr, lr_, a.t_small, a.t_large),
             "learned_curve": curve(gain[m], sr, lr_, a.t_small, a.t_large)}
        report["datasets"][name] = d
        lines.append("| %s | %d | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
            name, d["n"], 100 * d["small_acc"], 100 * d["large_acc"], 100 * d["only_large"], 100 * d["only_small"],
            100 * d["oracle_acc"]))
    lines += ["", "## Accuracy vs escalation rate (all questions)", "",
              "| escalated | threshold acc | learned-deferral acc | mean ms | p90 ms |", "|---|---|---|---|---|"]
    for t, l in zip(report["datasets"]["all"]["threshold_curve"], report["datasets"]["all"]["learned_curve"]):
        lines.append("| %.0f%% | %.1f%% | %.1f%% | %.0f | %.0f |" % (100 * t["escalated"], 100 * t["acc"], 100 * l["acc"],
                                                                 t["mean_ms"], t["p90_ms"]))
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(a.out, "cascade.md"), "w") as f:
        f.write(text)
    with open(os.path.join(a.out, "cascade.json"), "w") as f:
        json.dump(report, f, indent=2)
    log_experiment("cascade", "%s->%s" % (os.path.basename(os.path.normpath(a.small)),
                                          os.path.basename(os.path.normpath(a.large))),
                   config=vars(a), metrics=report, datasets=names, checkpoint=a.large,
                   artifacts=[os.path.join(a.out, "cascade.md"), os.path.join(a.out, "cascade.json")])


if __name__ == "__main__":
    main()
