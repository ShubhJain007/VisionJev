"""Local fine-tune for VisionJev (one GPU, no Modal): the ``modal_app.finetune`` recipe plus the VisionJev gates.

1. Runs the data validator (``../data_validator.py``) over every dataset first and stops if it fails.
2. Fine-tunes (default SmolVLM2-500M, ``freeze="full"``: language model + head train, vision tower frozen) with the
   repo's objective (``laya.vlm_train.vlm_loss``): datasets sampled in proportion to size, at most ``--max-passes``
   epochs each, cross-entropy annealed away so training ends on proper scoring rules. The last ``--n-calib`` train
   records per dataset are held out; they pick the best eval step (lowest NLL) and fit the temperatures.
3. Reports, per dataset, raw and calibrated accuracy / ECE, the selective-accuracy-at-coverage curve, and the
   shuffled-image control (each val image swapped for another val image of the same dataset: accuracy should fall
   toward the prior, otherwise the model is answering from the text alone).

    .venv/bin/python scripts/visionjev_finetune.py --datasets aokvqa,vqav2_yesno --minutes 30 --run-name baseline

Writes ``<out>/<run>/`` (the agent, loadable with ``VLMAgent(path)``) and ``<out>/<run>/metrics.json``.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

from laya.common import QTYPES  # noqa: E402
from laya.visionjev_log import log_experiment  # noqa: E402
from laya.vlm import SMOLVLM2_500M_BACKBONE, VLMAgent, set_trainable  # noqa: E402
from laya.vlm_train import (ACT_COSTS, collect_logits, fit_temperatures_from, format_metrics,  # noqa: E402
                            load_jsonl_examples, metrics_from, train)

COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)


def validate(root: str, names, allow_noncommercial: bool) -> None:
    sys.path.insert(0, WORKSPACE)
    import data_validator

    rep, _ = data_validator.run([os.path.join(root, n) for n in names], allow_noncommercial=allow_noncommercial)
    if rep.n_errors:
        raise SystemExit("data validator failed (%d violations); not training" % rep.n_errors)


def load(root: str, names, n_calib: int, max_train: int, max_val: int):
    train_ex, calib_ex, val_ex = [], [], []
    for name in names:
        tr = load_jsonl_examples(root, name, "train", limit=max_train + n_calib if max_train else None)
        calib_ex += tr[-n_calib:]
        train_ex += tr[:-n_calib]
        # train-only sets (the *_var variety sets: their held-out counterpart is the *_hof eval set)
        has_val = os.path.exists(os.path.join(root, name, "val.jsonl"))
        va = load_jsonl_examples(root, name, "val", limit=max_val or None) if has_val else []
        val_ex += va
        print("dataset %s: %d train, %d calib, %d val" % (name, len(tr) - n_calib, min(n_calib, len(tr)), len(va)))
    return train_ex, calib_ex, val_ex


def data_fingerprint(root: str, names) -> str:
    """Changes when a dataset of the mix is added, removed or rebuilt (name and size of every split file)."""
    import hashlib

    h = hashlib.sha256()
    for n in sorted(names):
        for sp in ("train", "val"):
            p = os.path.join(root, n, sp + ".jsonl")
            h.update(("%s/%s:%d\n" % (n, sp, os.path.getsize(p) if os.path.exists(p) else -1)).encode())
    return h.hexdigest()[:16]


def validated(marker: str, fingerprint: str) -> bool:
    """True when ``marker`` records a passed validation of exactly this data."""
    try:
        return json.load(open(marker)).get("fingerprint") == fingerprint
    except (OSError, ValueError, AttributeError):
        return False


def attach_notes(examples, root: str, tag: str) -> int:
    """Set ``ex["notes"]`` from ``<root>/<dataset>/notes/<tag>.jsonl`` by row id; returns how many rows got notes."""
    by_set, n = {}, 0
    for ex in examples:
        name = ex["dataset"]
        if name not in by_set:
            path = os.path.join(root, name, "notes", tag + ".jsonl")
            by_set[name] = ({r["id"]: r["notes"] for r in map(json.loads, open(path)) if r.get("notes")}
                            if os.path.exists(path) else {})
        notes = by_set[name].get(ex.get("id"))
        if notes:
            ex["notes"] = notes
            n += 1
    return n


def shuffled_images(examples, seed: int = 0):
    """Each example with another example's image (same dataset, never its own)."""
    rng = random.Random(seed)
    out, by_ds = [], {}
    for ex in examples:
        by_ds.setdefault(ex["dataset"], []).append(ex)
    for exs in by_ds.values():
        imgs = [ex["state"].get("image") if isinstance(ex["state"], dict) else None for ex in exs]
        perm = list(range(len(exs)))
        rng.shuffle(perm)
        for i, ex in enumerate(exs):
            j = perm[i] if perm[i] != i or len(exs) == 1 else perm[(i + 1) % len(exs)]
            if isinstance(ex["state"], dict) and imgs[j] is not None:
                ex = dict(ex, state=dict(ex["state"], image=imgs[j]))
            out.append(ex)
    return out


def selective(records, temperatures):
    """Accuracy when acting only on the most confident fraction of rows, per coverage level (the product metric)."""
    out = {}
    groups = {"all": records}
    for r in records:
        groups.setdefault(r["dataset"], []).append(r)
    for name, rs in groups.items():
        conf, hit = [], []
        for r in rs:
            p = torch.softmax(r["logits"] / temperatures[r["qtype"]], -1)
            conf.append(float(p.max()))
            hit.append(float(int(p.argmax()) == r["label"]))
        order = np.argsort(-np.array(conf))
        hit = np.array(hit)[order]
        conf = np.array(conf)[order]
        out[name] = {}
        for c in COVERAGES:
            n = max(1, int(round(c * len(hit))))
            out[name]["%d%%" % round(100 * c)] = {"acc": float(hit[:n].mean()), "threshold": float(conf[n - 1])}
    return out


def act_policies(records, temperatures, costs=ACT_COSTS):
    """Mean utility per question of three decide-or-escalate policies under the act head's cost matrix
    (``ACT_COSTS``: +1 act and right, -3 act and wrong, -0.5 escalate):

    * ``act_head``: act when the trained act head says P(act) > 0.5 (Laya's learned policy);
    * ``calibrated_threshold``: act when the calibrated confidence exceeds the cost-optimal threshold
      (1.0 * p - 3.0 * (1 - p) > -0.5  ->  p > 0.625), i.e. no extra head, just honest probabilities;
    * ``always_act``: the baseline of never escalating.
    """
    c_ok, c_bad, c_esc = costs
    tau = (c_esc - c_bad) / (c_ok - c_bad)
    out = {}
    groups = {"all": records}
    for r in records:
        groups.setdefault(r["dataset"], []).append(r)
    for name, rs in groups.items():
        res = {k: {"utility": 0.0, "act_rate": 0.0, "acc_when_acting": [0, 0]} for k in
               ("act_head", "calibrated_threshold", "always_act")}
        for r in rs:
            p = torch.softmax(r["logits"] / temperatures[r["qtype"]], -1)
            right = int(p.argmax()) == r["label"]
            decisions = {"act_head": r.get("act_probability", 1.0) > 0.5,
                         "calibrated_threshold": float(p.max()) > tau, "always_act": True}
            for k, act in decisions.items():
                res[k]["utility"] += (c_ok if right else c_bad) if act else c_esc
                res[k]["act_rate"] += act
                if act:
                    res[k]["acc_when_acting"][0] += right
                    res[k]["acc_when_acting"][1] += 1
        n = max(1, len(rs))
        out[name] = {k: {"utility": v["utility"] / n, "act_rate": v["act_rate"] / n,
                         "acc_when_acting": v["acc_when_acting"][0] / max(1, v["acc_when_acting"][1])}
                     for k, v in res.items()}
    return out


def format_act(a) -> str:
    return "\n".join("  %-22s " % name + "  ".join("%s: utility %+.3f, acts %.0f%%, acc when acting %.1f%%"
                                                    % (k, v["utility"], 100 * v["act_rate"], 100 * v["acc_when_acting"])
                                                    for k, v in pol.items()) for name, pol in a.items())


def format_selective(s) -> str:
    return "\n".join("  %-12s " % name + "  ".join("cov %s: acc %.3f (τ=%.2f)" % (k, v["acc"], v["threshold"])
                                                     for k, v in curve.items()) for name, curve in s.items())


def mix_from_config(path, root):
    """``configs/mix_*.json`` -> (dataset names, per-dataset weight multipliers, alpha, max_passes, report). Each
    domain gets its target share; inside a domain the sampler's ``size ** alpha`` spreads it (``ItemStream``: probability
    ``weight_k * n_k ** alpha``), so ``weight_k = share_domain / sum_{j in domain} n_j ** alpha``. A domain may set
    ``"dataset_weight": {name: m}`` (default 1) to scale a dataset inside its domain without changing the domain's
    share: ``weight_k = share * m_k / sum_j m_j n_j ** alpha``."""
    with open(path) as f:
        cfg = json.load(f)
    alpha = float(cfg.get("alpha", 0.5))
    names, weights, report = [], {}, {}
    for dom, spec in cfg["domains"].items():
        present = [d for d in spec["datasets"] if os.path.exists(os.path.join(root, d, "train.jsonl"))]
        if not present:
            continue
        sizes = {d: sum(1 for _ in open(os.path.join(root, d, "train.jsonl"))) for d in present}
        mult = {d: float(spec.get("dataset_weight", {}).get(d, 1.0)) for d in present}
        z = sum(mult[d] * n ** alpha for d, n in sizes.items())
        for d, n in sizes.items():
            names.append(d)
            weights[d] = spec["share"] * mult[d] / z
        report[dom] = {"share": spec["share"], "datasets": sizes,
                       **({"dataset_weight": spec["dataset_weight"]} if spec.get("dataset_weight") else {})}
    total = sum(spec["share"] for dom, spec in cfg["domains"].items() if dom in report)
    weights = {k: v / total for k, v in weights.items()}  # renormalise over the domains that exist
    return names, weights, alpha, cfg.get("max_passes"), {"version": cfg.get("version"), "domains": report}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="aokvqa,vqav2_yesno")
    ap.add_argument("--mix-config", default=None,
                    help="configs/mix_*.json: domain shares + alpha; overrides --datasets, --mix-alpha (and --max-passes if set)")
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--out", default=os.path.join(WORKSPACE, "checkpoints"))
    ap.add_argument("--run-name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    ap.add_argument("--backbone", default=SMOLVLM2_500M_BACKBONE)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=10**9)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--freeze", choices=["head", "last_n", "full", "lora"], default="full")
    ap.add_argument("--lora-rank", type=int, default=64, help="with --freeze lora")
    ap.add_argument("--lora-alpha", type=int, default=0, help="default 2 x rank")
    ap.add_argument("--lora-experts", action="store_true",
                    help="MoE backbones: also put LoRA on the fused expert weights (mlp.experts.gate_up_proj/down_proj)")
    ap.add_argument("--lora-expert-rank", type=int, default=0,
                    help="LoRA rank per expert for --lora-experts (default: --lora-rank). peft gives a fused expert "
                         "weight rank r x n_experts, so 30B-A3B (128 experts) at r=64 would train ~2.5B params")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="backbone storage dtype at build time (bf16 for 30B-A3B: fp32 would need ~124 GB); the "
                         "trainable parameters are kept in fp32 either way")
    ap.add_argument("--n-last", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--option-attention", choices=["causal", "block"], default="block")
    ap.add_argument("--w-ce-schedule", choices=["const", "anneal"], default="anneal",
                    help="anneal: cross-entropy weight decays to 0, so training ends on proper scoring rules alone")
    ap.add_argument("--max-passes", type=float, default=2.0,
                    help="epochs per dataset; training stops when every dataset hits it (0 = no cap)")
    ap.add_argument("--mix-alpha", type=float, default=1.0,
                    help="sampling weight = size ** alpha: 1 = proportional to size (a plain shuffle), 0 = equal "
                         "(small sets get many epochs: the first baseline overfit A-OKVQA that way)")
    ap.add_argument("--image-split-edge", type=int, default=0,
                    help="0 = one 512 px view per image; N = the processor's tiling: longest edge resized to N, cut into "
                         "512 px tiles plus a global view (1024 -> 5 views, 1536 -> 10, 2048 -> 17)")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute activations in the backward pass (same math, ~30%% slower, much less memory); "
                         "on by default when --image-split-edge is set")
    ap.add_argument("--qwen-max-pixels", type=int, default=0,
                    help="Qwen3-VL backbones: pixel budget per image (32 px per token, e.g. 589824 = 768^2 -> 576 tokens)")
    ap.add_argument("--frozen-bf16", action="store_true",
                    help="store the parameters --freeze leaves frozen in bf16 (trainable ones stay fp32): fits 2B on 16 GB")
    ap.add_argument("--train-act", action="store_true",
                    help="train Laya's act/escalate head on its cost matrix (off in the repo's published runs)")
    ap.add_argument("--no-keep-best", action="store_true",
                    help="keep the final weights instead of the eval step with the lowest val NLL")
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--eval-per-dataset", type=int, default=200,
                    help="val rows per dataset in the periodic evals (with ~100 datasets, 200 is ~20k rows per eval)")
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--validate-only", action="store_true",
                    help="validate the mix, write its marker (<root>/.validated-<mix>) and exit: do this once on the "
                         "workstation so a pod does not spend GPU time opening every image")
    ap.add_argument("--deadline-lead", type=float, default=12.0,
                    help="minutes kept between this process starting and its training loop (model load, baseline eval) "
                         "when data/logs/train_deadline shortens --minutes")
    ap.add_argument("--notes-tag", default="",
                    help="think-if-unsure training: attach <root>/<dataset>/notes/<tag>.jsonl ({id, notes}: the student's "
                         "own working from scripts/visionjev_notes.py) to the training rows")
    ap.add_argument("--notes-prob", type=float, default=0.5, help="with --notes-tag: share of noted rows that show the notes")
    ap.add_argument("--length-groups", type=int, default=1,
                    help="run each batch as up to N forward passes of similar-length rows instead of padding every row to "
                         "the longest (same rows, loss and updates; B2 computed 75%% padding). 4 is a good value")
    ap.add_argument("--group-token-budget", type=int, default=0, help="cap on one forward pass's padded tokens (0 = none)")
    ap.add_argument("--grad-checkpointing-above", type=int, default=None,
                    help="gradient checkpointing only for steps whose padded tokens exceed this (overrides "
                         "--grad-checkpointing; a 4B on 80 GB fits ~12000 per device without it)")
    ap.add_argument("--devices", default="",
                    help="comma-separated devices to spread a step's forward passes over, e.g. cuda:0,cuda:1 (one "
                         "optimizer, one loss; needs --length-groups > 1 to have passes to spread)")
    ap.add_argument("--logit-cap", type=float, default=0,
                    help="soft-cap option logits at cap*tanh(z/cap) (e.g. 15); stored in the checkpoint config. 0 = off")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop when the calibration NLL has not improved for this many evals (0: never); the best "
                         "weights are kept either way")
    ap.add_argument("--augment", action="store_true",
                    help="train-time setup augmentation (laya.visionjev_augment: state formats, negation, choice<->noul, "
                         "none-of-the-above, option subsets, planted injections, image ops; labels derived exactly)")
    ap.add_argument("--allow-noncommercial", action="store_true")
    ap.add_argument("--max-len", type=int, default=0,
                    help="total token budget per sequence (image tokens + state + question + options); 0 = the "
                         "backbone default (Qwen: 768 + image budget). Text-only rows get all of it")
    ap.add_argument("--rationale-weight", type=float, default=0.0,
                    help="weight of the language-model loss on each example's worked rationale, appended after the "
                         "option block (training only; the read-out is computed as at inference). 0 = off")
    ap.add_argument("--rationale-max-tokens", type=int, default=96)
    ap.add_argument("--abstain", type=float, default=0.0,
                    help="train the abstain + reason head (Hypothesis/V6.2.md section 3) with this BCE weight; turns on "
                         "the gold_removed / cbd_removed augmentations and masks the option loss on abstain rows. 0 = off")
    ap.add_argument("--init-checkpoint", default="",
                    help="continue from a saved agent (head + LoRA + temperatures + config) instead of a fresh head; "
                         "for a short delta stage after a broad run")
    a = ap.parse_args(argv)
    # A pod has a hard time cap (scripts/pod_autopilot.sh deletes it): <workspace>/data/logs/train_deadline holds the epoch
    # second by which the training loop must have ended, so the checkpoint and the evals still fit. It only ever shortens.
    # <workspace>/data/logs/train_overrides.json: argument overrides for a long run launched by a script whose command
    # line can no longer be changed (e.g. {"length_groups": 4}); ignored by smoke runs (--steps <= 1000).
    over = os.path.join(WORKSPACE, "data", "logs", "train_overrides.json")
    if os.path.exists(over) and a.steps > 1000:
        for k, v in json.load(open(over)).items():
            if not hasattr(a, k):
                raise SystemExit("train_overrides.json: unknown argument %r" % k)
            print("override from %s: %s = %r (was %r)" % (over, k, v, getattr(a, k)), flush=True)
            setattr(a, k, v)
    deadline = os.path.join(WORKSPACE, "data", "logs", "train_deadline")
    if os.path.exists(deadline) and a.steps > 1000:
        left = (float(open(deadline).read().strip()) - time.time()) / 60 - a.deadline_lead
        if left < a.minutes:
            print("--minutes cut from %g to %.0f by %s" % (a.minutes, max(5.0, left), deadline), flush=True)
            a.minutes = max(5.0, left)

    names = [n for n in a.datasets.split(",") if n]
    mix_weights, mix_report = None, None
    if a.mix_config:
        names, mix_weights, a.mix_alpha, passes, mix_report = mix_from_config(a.mix_config, a.root)
        a.max_passes = passes or a.max_passes
        print("mix %s: %d datasets, alpha %.2f, domains %s" % (mix_report["version"], len(names), a.mix_alpha,
              {d: v["share"] for d, v in mix_report["domains"].items()}), flush=True)
    # the validator opens every image (~40 min for the round-3 mix): once a mix has passed on this machine's copy of the
    # data, a marker lets the next run of the same mix (the real run after the smoke run) start at once
    # (validate once, e.g. locally with --validate-only, and ship the marker with the data: scripts/mix_data_dirs.py)
    marker = os.path.join(a.root, ".validated-" + os.path.basename(a.mix_config or ",".join(sorted(names))[:80]))
    fp = data_fingerprint(a.root, names)
    if validated(marker, fp):
        print("data validation skipped: %s records a pass for this data (%s)" % (marker, fp), flush=True)
    else:
        validate(a.root, names, a.allow_noncommercial)
        json.dump({"fingerprint": fp, "datasets": len(names), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  open(marker, "w"))
    if a.validate_only:
        print("validated %d datasets (%s); marker %s" % (len(names), fp, marker), flush=True)
        return
    train_ex, calib_ex, val_ex = load(a.root, names, a.n_calib, a.max_train, a.max_val)

    t0 = time.time()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    extra = {}
    if a.freeze == "lora":
        extra["lora"] = {"r": a.lora_rank, "alpha": a.lora_alpha or 2 * a.lora_rank,
                         "parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"] if a.lora_experts else []}
        if a.lora_experts and a.lora_expert_rank:
            extra["lora"]["expert_r"] = a.lora_expert_rank
    if a.logit_cap:
        extra["logit_cap"] = a.logit_cap
    if a.abstain:
        extra["abstain_head"] = True
    if a.init_checkpoint:
        agent = VLMAgent(a.init_checkpoint, device="cuda", dtype=a.dtype, **({"max_len": a.max_len} if a.max_len else {}),
                         **({"abstain_head": True} if a.abstain else {}))
        print("continuing from %s (max_len %d)" % (a.init_checkpoint, agent.cfg["max_len"]), flush=True)
    else:
        agent = VLMAgent(backbone=a.backbone, device="cuda", option_attention=a.option_attention, dtype=a.dtype, **extra,
                         preprocess="processor",  # photos of mixed sizes; "gpu" is for fixed-size game frames
                         **({"image_split_edge": a.image_split_edge} if a.image_split_edge else {}),
                         **({"qwen_max_pixels": a.qwen_max_pixels} if a.qwen_max_pixels else {}),
                         **({"max_len": a.max_len} if a.max_len else {}))
    model, proc = agent.model, agent.processor
    print("image prep: split_edge %d -> up to %d views x %d tokens; max_len %d"
          % (agent.prep.split_edge, agent.prep.max_tiles, agent.prep.image_seq_len, agent.cfg["max_len"]), flush=True)
    if a.grad_checkpointing or a.image_split_edge:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.dtype == "bf16":  # trainable parameters (head, LoRA, unfrozen layers) keep fp32 master weights
        set_trainable(model, a.freeze, n_last=a.n_last)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    if a.frozen_bf16:
        # train() calls set_trainable again with the same arguments, so the split is the same; frozen weights never
        # receive updates, so bf16 storage loses nothing, and the trainable ones keep fp32 master weights
        n_train = set_trainable(model, a.freeze, n_last=a.n_last)
        for p in model.parameters():
            if not p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
        torch.cuda.empty_cache()
        print("frozen parameters stored in bf16; %.0fM trainable in fp32" % (n_train / 1e6), flush=True)
    # eval batches hold ~32 image views whatever the tiling (32 pages x 17 tiles ran out of memory on 16 GB)
    ev_kw = dict(batch_size=max(2, 32 // agent.prep.max_tiles), num_workers=a.num_workers)
    small_val = []
    for name in names:
        small_val += [ex for ex in val_ex if ex["dataset"] == name][:a.eval_per_dataset]

    log = {"run": a.run_name, "args": vars(a), "evals": [], "mix": mix_report}
    base = metrics_from(collect_logits(model, proc, small_val, **ev_kw))
    print("[eval step 0, untrained head] " + format_metrics(base), flush=True)
    log["evals"].append({"step": 0, **base})

    # "best" is chosen on the calibration set (already reserved for tuning), never on val, so val stays unbiased
    best = {"nll": metrics_from(collect_logits(model, proc, calib_ex, **ev_kw))["all"]["nll"], "step": 0, "state": None}

    def eval_fn(step):
        m = metrics_from(collect_logits(model, proc, small_val, **ev_kw))
        print("[eval step %d] %s" % (step, format_metrics(m)), flush=True)
        log["evals"].append({"step": step, **m})
        if a.no_keep_best:
            return
        nll = metrics_from(collect_logits(model, proc, calib_ex, **ev_kw))["all"]["nll"]  # NLL also sees overconfidence
        print("  calib nll %.4f (best %.4f at step %d)" % (nll, best["nll"], best["step"]), flush=True)
        log["evals"][-1]["calib_nll"] = nll
        if nll < best["nll"]:
            # only what training changes (heads, LoRA, unfrozen layers): a 30B state dict is ~60 GB of CPU copies
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            best.update(nll=nll, step=step, stale=0, state={k: v.detach().to("cpu", copy=True)
                                                            for k, v in model.state_dict().items() if k in trainable})
            return True
        best["stale"] = best.get("stale", 0) + 1
        if a.patience and best["stale"] >= a.patience:
            print("  no calibration improvement in %d evals: stopping (best step %d kept)" % (best["stale"], best["step"]),
                  flush=True)
            return "stop"
        return True

    augment_fn = None
    if a.augment:
        from laya.visionjev_augment import augment as augment_fn
        if a.abstain:
            import functools

            from laya.visionjev_augment import ABSTAIN_AUGMENTS, AUGMENTS
            augment_fn = functools.partial(augment_fn, probs=dict(AUGMENTS, **ABSTAIN_AUGMENTS))
    if a.notes_tag:
        print("notes %r attached to %d of %d training rows (shown with probability %.2f)"
              % (a.notes_tag, attach_notes(train_ex, a.root, a.notes_tag), len(train_ex), a.notes_prob), flush=True)
    train_stats = {}
    losses = train(model, proc, train_ex, steps=a.steps, batch_size=a.batch_size, freeze=a.freeze, n_last=a.n_last,
                   lr_head=a.lr_head, lr_backbone=a.lr_backbone, device="cuda", log_every=50, max_minutes=a.minutes,
                   num_workers=a.num_workers, warmup=100, eval_fn=eval_fn, eval_every=a.eval_every,
                   w_ce_schedule=a.w_ce_schedule, max_passes=a.max_passes or None, mix_alpha=a.mix_alpha,
                   mix_weights=mix_weights, augment=augment_fn,
                   train_act=a.train_act, w_rationale=a.rationale_weight,
                   rationale_max_tokens=a.rationale_max_tokens, w_abstain=a.abstain,
                   notes_prob=a.notes_prob if a.notes_tag else 0.0,
                   length_groups_n=a.length_groups, group_token_budget=a.group_token_budget,
                   ckpt_above_tokens=a.grad_checkpointing_above,
                   devices=[d for d in a.devices.split(",") if d] or None, stats=train_stats)
    log["train_stats"] = {k: v for k, v in train_stats.items() if k != "samples_per_dataset"}
    print("train stats: %s" % json.dumps(log["train_stats"]), flush=True)
    log["steps"], log["examples_seen"] = len(losses), len(losses) * a.batch_size
    log["loss_first50"] = float(np.mean(losses[:50]))
    log["loss_last50"] = float(np.mean(losses[-50:]))
    print("trained %d steps (%d examples); loss first50 %.4f -> last50 %.4f"
          % (log["steps"], log["examples_seen"], log["loss_first50"], log["loss_last50"]), flush=True)

    eval_fn(len(losses))  # the final weights compete for "best" too
    if best["state"] is not None:
        missing, unexpected = model.load_state_dict(best["state"], strict=False)  # trainable subset only
        assert not unexpected, unexpected
        print("restored best eval step %d (calib nll %.4f)" % (best["step"], best["nll"]), flush=True)
    log["best_step"], log["best_calib_nll"] = best["step"], best["nll"]
    model.eval()

    temps = fit_temperatures_from(collect_logits(model, proc, calib_ex, **ev_kw))
    val_records = collect_logits(model, proc, val_ex, **ev_kw)
    shuf_records = collect_logits(model, proc, shuffled_images(val_ex), **ev_kw)
    log["temperature"] = temps
    log["val_raw"] = metrics_from(val_records)
    log["val_calibrated"] = metrics_from(val_records, temps)
    log["val_shuffled_images"] = metrics_from(shuf_records, temps)
    log["selective_calibrated"] = selective(val_records, temps)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])
    print("[final val, T=1]         " + format_metrics(log["val_raw"]))
    print("[final val, calibrated]  " + format_metrics(log["val_calibrated"]))
    print("[shuffled-image control] " + format_metrics(log["val_shuffled_images"]))
    print("[selective accuracy at coverage, calibrated]\n" + format_selective(log["selective_calibrated"]))
    log["act_policies"] = act_policies(val_records, temps)
    print("[decide-or-escalate utility (+1 right, -3 wrong, -0.5 escalate)%s]\n"
          % ("" if a.train_act else "; act head UNTRAINED") + format_act(log["act_policies"]))

    out_dir = os.path.join(a.out, a.run_name)
    agent.temperature = temps
    agent.save(out_dir, include_backbone=a.freeze not in ("head", "lora"))
    log["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("saved %s (%.1f min total)" % (out_dir, (time.time() - t0) / 60))
    log_experiment("train", a.run_name, config=vars(a), datasets=names, checkpoint=out_dir,
                   metrics={k: log[k] for k in ("steps", "best_step", "temperature", "val_raw", "val_calibrated",
                                                "val_shuffled_images", "selective_calibrated", "act_policies",
                                                "minutes") if k in log},
                   artifacts=[os.path.join(out_dir, "metrics.json")])


if __name__ == "__main__":
    main()
