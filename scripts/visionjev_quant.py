"""Quantisation drift: accuracy and calibration of a checkpoint with its backbone's linear layers quantised, against
the same checkpoint unquantised, on the same val rows, logged.

Quantisation is *simulated* (quantise, then dequantise in place), which reproduces the numerics of real low-precision
weights without needing the kernels; it measures quality drift, not speed. Modes (the decision heads are never
quantised; every ``nn.Linear`` inside the backbone is, vision tower included):

* ``w8``    int8 weights, symmetric, one scale per output channel (weight-only, e.g. bitsandbytes / torchao int8)
* ``w4``    int4 weights, symmetric, one scale per 128-input group (GPTQ / AWQ-style weight-only)
* ``fp8``   float8 e4m3 weights (per output channel) and activations (per token, dynamic): the FP8 serving path

Reported per dataset and mode: accuracy, ECE, NLL with the checkpoint's temperatures (not refitted), the flip rate of
the argmax against the unquantised model, and the mean total-variation distance between their distributions.

    .venv/bin/python scripts/visionjev_quant.py ../checkpoints/mix-split1024 --datasets aokvqa,laya_typed --n 300
"""
import argparse
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

from laya.visionjev_log import log_experiment  # noqa: E402
from laya.vlm import VLMAgent  # noqa: E402
from laya.vlm_train import collect_logits, load_jsonl_examples, metrics_from  # noqa: E402

MODES = ("w8", "w4", "fp8")
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def _q_sym(w: torch.Tensor, bits: int, dim_groups: int = 0) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    shape = w.shape
    x = w.float()
    if dim_groups and shape[1] % dim_groups == 0:
        x = x.reshape(shape[0], shape[1] // dim_groups, dim_groups)
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return (torch.round(x / scale).clamp(-qmax - 1, qmax) * scale).reshape(shape).to(w.dtype)


def _q_fp8(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    scale = x.float().abs().amax(dim, keepdim=True).clamp_min(1e-8) / FP8_MAX
    return ((x.float() / scale).to(FP8).float() * scale).to(x.dtype)


def quantise(model, mode: str) -> int:
    """Quantise every backbone ``nn.Linear`` in place; returns how many layers were changed."""
    n = 0
    for name, mod in model.encoder.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        with torch.no_grad():
            if mode == "w8":
                mod.weight.copy_(_q_sym(mod.weight, 8))
            elif mode == "w4":
                mod.weight.copy_(_q_sym(mod.weight, 4, dim_groups=128))
            elif mode == "fp8":
                mod.weight.copy_(_q_fp8(mod.weight, dim=1))
                mod.register_forward_pre_hook(lambda m, args: (_q_fp8(args[0], dim=-1),) + tuple(args[1:]))
            else:
                raise ValueError(mode)
        n += 1
    return n


def _probs(recs, T):
    return [torch.softmax(r["logits"] / T[r["qtype"]], -1) for r in recs]


def drift(base, other, T, examples):
    pb, po = _probs(base, T), _probs(other, T)
    out = {}
    for name in sorted({ex["dataset"] for ex in examples}) + ["all"]:
        idx = [i for i, ex in enumerate(examples) if name == "all" or ex["dataset"] == name]
        flips = sum(int(pb[i].argmax() != po[i].argmax()) for i in idx)
        tv = sum(0.5 * float((pb[i] - po[i]).abs().sum()) for i in idx)
        out[name] = {"flip_rate": flips / max(1, len(idx)), "mean_tv": tv / max(1, len(idx)), "n": len(idx)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--n", type=int, default=300, help="val rows per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--tag", default="quant")
    ap.add_argument("--num-workers", type=int, default=6)
    a = ap.parse_args(argv)
    names = [d for d in a.datasets.split(",") if d]
    examples = []
    for name in names:
        exs = load_jsonl_examples(a.root, name, "val")
        if a.n and len(exs) > a.n:
            exs = random.Random(a.seed).sample(exs, a.n)
        examples += exs
    run = os.path.basename(os.path.normpath(a.checkpoint))
    results, base = {}, None
    for mode in ["none"] + [m for m in a.modes.split(",") if m]:
        agent = VLMAgent(a.checkpoint, device="cuda")
        T = list(agent.temperature)
        n_layers = quantise(agent.model, mode) if mode != "none" else 0
        bs = 4 if agent.model.is_qwen else max(2, 32 // agent.prep.max_tiles)
        recs = collect_logits(agent.model, agent.processor, examples, batch_size=bs, num_workers=a.num_workers)
        cal = metrics_from(recs, T)
        results[mode] = {"calibrated": cal, "layers_quantised": n_layers}
        if mode == "none":
            base = recs
        else:
            results[mode]["vs_unquantised"] = drift(base, recs, T, examples)
        print("%-5s layers %4d | " % (mode, n_layers) + " | ".join(
            "%s acc %.3f ece %.3f%s" % (d, cal[d]["acc"], cal[d]["ece"],
                                        "" if mode == "none" else " flip %.3f" % results[mode]["vs_unquantised"][d]["flip_rate"])
            for d in names + ["all"] if d in cal), flush=True)
        del agent
        torch.cuda.empty_cache()
    log_experiment("quant_drift", "%s:%s" % (a.tag, run), config=vars(a), metrics=results, datasets=names,
                   checkpoint=a.checkpoint,
                   notes="simulated quantisation (quantise-dequantise); temperatures not refitted")
    return results


if __name__ == "__main__":
    main()
