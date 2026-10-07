"""Measure the trainer's speed options on real batches, and check they leave the step unchanged.

For one checkpoint and mix: (1) in eval mode, the grouped forward (and the multi-device one) must equal the padded
forward row for row; (2) the same stream of batches is run for ``--steps`` steps with learning rate 0 under each
configuration, reporting seconds per step, padded tokens per step, peak GPU memory and the mean loss (dropout makes
the losses differ slightly between configurations; the weights never move, so every configuration sees the same model).

    .venv/bin/python scripts/visionjev_speedcheck.py ../checkpoints/qwen4b-v6b2 --mix-config configs/mix_v6b2.json \\
        --steps 30 --devices cuda:0,cuda:1
"""
import argparse
import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--mix-config", required=True)
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--per-dataset", type=int, default=200)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--devices", default="", help="e.g. cuda:0,cuda:1 to also time the multi-device step")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--freeze", default="lora")
    ap.add_argument("--rationale-weight", type=float, default=0.3)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--ckpt-above", type=int, default=12000)
    a = ap.parse_args(argv)

    from laya.visionjev_log import log_experiment
    from laya.vlm import collate_vlm, load_vlm, set_trainable
    from laya.vlm_train import GroupRunner, _collate_train, _forward, load_jsonl_examples, make_item, train
    from visionjev_finetune import mix_from_config

    names, weights, alpha, _, _ = mix_from_config(a.mix_config, a.root)
    agent = load_vlm(a.checkpoint, device=a.device)
    if a.max_len:
        agent.cfg["max_len"] = agent.processor.laya_max_len = a.max_len
    model, proc = agent.model, agent.processor
    cuda = a.device.startswith("cuda")
    if cuda:  # as visionjev_finetune.py --dtype bf16: trainable weights in fp32, frozen in bf16
        set_trainable(model, a.freeze)
        for p in model.parameters():
            p.data = p.data.float() if p.requires_grad else p.data.to(torch.bfloat16)
    examples = []
    for n in names:
        examples += load_jsonl_examples(a.root, n, "train", limit=a.per_dataset)
    devices = [d for d in a.devices.split(",") if d]
    res = {"checkpoint": a.checkpoint, "batch_size": a.batch_size, "groups": a.groups, "devices": devices}

    # (1) exactness on one real batch, eval mode
    rng = random.Random(0)
    items = []
    while len(items) < a.batch_size:
        ex = rng.choice(examples)
        try:
            it = make_item(proc, ex, rng, rationale_max_tokens=96 if a.rationale_weight else 0)
        except (OSError, ValueError):
            continue
        it["dataset"] = ex["dataset"]
        items.append(it)
    pad = proc.tokenizer.pad_token_id
    model.eval()
    dev = torch.device(a.device)
    with torch.no_grad():
        full = collate_vlm(items, pad)
        fb = {k: (v.to(dev, model.encoder.dtype) if k == "pixel_values" else v.to(dev)) if torch.is_tensor(v) else v
              for k, v in full.items()}
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
            plain, _ = _forward(model, fb)
        batch = _collate_train(items, pad, max_groups=a.groups)
        grouped = GroupRunner(model)(batch["parts"])["logits"]
        mask = full["marker_mask"].to(dev)
        diff = float(((plain.float() - grouped.float()).abs() * mask).max())
        same_top = float((plain.argmax(-1) == grouped.argmax(-1)).float().mean())
        mean_diff = float(((plain.float() - grouped.float()).abs() * mask).sum() / mask.sum())
        res["exactness"] = {"max_abs_logit_diff_grouped_vs_padded": round(diff, 5), "mean_abs_logit_diff": round(mean_diff, 5),
                            "same_argmax": same_top, "parts": len(batch["parts"])}
        if len(devices) > 1:
            multi = GroupRunner(model, devices)(batch["parts"])["logits"]
            res["exactness"]["max_abs_logit_diff_multi_device"] = round(float(((plain.float() - multi.float()).abs() * mask).max()), 5)
    print("exactness: %s" % json.dumps(res["exactness"]), flush=True)

    # (2) timing: identical batch stream, learning rate 0
    configs = [("padded + checkpointing (B2 setting)", dict(length_groups_n=1, ckpt_above_tokens=0)),
               ("groups, checkpointing above %d" % a.ckpt_above, dict(length_groups_n=a.groups, ckpt_above_tokens=a.ckpt_above))]
    if len(devices) > 1:
        configs.append(("groups + %d devices" % len(devices),
                        dict(length_groups_n=max(a.groups, 3 * len(devices)), ckpt_above_tokens=a.ckpt_above, devices=devices)))
    res["timing"] = {}
    for label, kw in configs:
        if cuda:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        stats = {}
        t0 = time.time()
        try:
            losses = train(model, proc, examples, steps=a.steps, batch_size=a.batch_size, freeze=a.freeze, lr_head=0.0,
                           lr_backbone=0.0, device=a.device, log_every=0, num_workers=a.num_workers, seed=1,
                           mix_weights=weights, mix_alpha=alpha, w_rationale=a.rationale_weight, stats=stats, **kw)
        except torch.OutOfMemoryError as e:
            res["timing"][label] = {"error": "out of memory: %s" % str(e)[:120]}
            print("%s: OUT OF MEMORY" % label, flush=True)
            torch.cuda.empty_cache()
            continue
        mem = [round(torch.cuda.max_memory_allocated(i) / 2 ** 30, 1) for i in range(torch.cuda.device_count())] if cuda else []
        res["timing"][label] = {"s_per_step": round(stats["train_minutes"] * 60 / max(1, stats["steps"]), 2),
                                "padded_tokens_per_step": round(stats["padded_tokens_per_step"]),
                                "data_wait": round(stats["data_wait_frac"], 3), "peak_gb": mem,
                                "mean_loss": round(sum(losses) / max(1, len(losses)), 4), "wall_s": round(time.time() - t0)}
        print("%s: %s" % (label, json.dumps(res["timing"][label])), flush=True)
    base = res["timing"].get(configs[0][0], {}).get("s_per_step")
    for label, _ in configs[1:]:
        t = res["timing"].get(label, {}).get("s_per_step")
        if base and t:
            res["timing"][label]["speedup_vs_padded"] = round(base / t, 2)
    print(json.dumps(res, indent=1), flush=True)
    log_experiment("speedcheck", "speedcheck:%s" % os.path.basename(os.path.normpath(a.checkpoint)),
                   config={k: v for k, v in vars(a).items()}, metrics=res, checkpoint=a.checkpoint)


if __name__ == "__main__":
    main()
