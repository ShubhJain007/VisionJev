"""Generate the student's OWN working ("notes") for training rows (think-if-unsure, Hypothesis/V6.3.md section 4).

For each dataset, sample training rows, let the checkpoint continue ``<question row> Because:`` greedily (the rationale
loss trains that continuation) and write ``<root>/<dataset>/notes/<tag>.jsonl`` rows ``{"id", "notes", "tokens"}``.
``visionjev_finetune.py --notes-tag <tag>`` then attaches them, so the read-out learns to use its working and to check
it against the state. The notes are whatever the student writes, right or wrong; the gold rationale is never used.

Rows are grouped by length and generated in batches with the key/value cache. ``--selfcheck`` (on by default) first
compares the cached batched generation with the plain token-by-token loop on two rows of this backbone and stops on a
mismatch (the Qwen path cannot be tested off-GPU).

    .venv/bin/python scripts/visionjev_notes.py ../checkpoints/qwen4b-v6b2 --mix-config configs/mix_v6b3.json \\
        --per-dataset 800 --tag v6b2 --max-tokens 96
"""
import argparse
import json
import os
import random
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))


def trim_notes(text: str, max_sentences: int = 4) -> str:
    """Keep the working up to the first repeated sentence or ``max_sentences`` (a checkpoint trained before rationales
    ended with an end-of-text token keeps writing)."""
    out, seen = [], set()
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if not key or key in seen or len(out) >= max_sentences:
            break
        seen.add(key)
        out.append(s)
    return " ".join(out)


def shard(names, spec: str):
    """``names[i::n]`` for ``spec`` "i/n" (all of them for an empty spec)."""
    if not spec:
        return list(names)
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise SystemExit("--shard i/n needs 0 <= i < n")
    return list(names)[i::n]


def batches(items, budget: int, max_rows: int):
    """Rows sorted by length, cut so one batch's padded tokens stay within ``budget``."""
    items = sorted(items, key=lambda it: len(it["ids"]))
    cur = []
    for it in items:
        if cur and (len(cur) >= max_rows or len(it["ids"]) * (len(cur) + 1) > budget):
            yield cur
            cur = []
        cur.append(it)
    if cur:
        yield cur


def generate(agent, items, max_tokens: int, budget: int, max_rows: int = 48):
    from laya.vlm import collate_vlm, end_of_text_id

    tok, model, dev = agent.processor.tokenizer, agent.model, agent.device
    stop = (end_of_text_id(tok),)
    for chunk in batches(items, budget, max_rows):
        b = collate_vlm(chunk, tok.pad_token_id)
        kw = {k: (b[k].to(dev, model.encoder.dtype) if k == "pixel_values" else b[k].to(dev))
              for k in ("pixel_values", "pixel_attention_mask", "raw_pixels", "image_mask", "image_grid_thw")
              if b.get(k) is not None}
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            new = model.generate_continuation(b["input_ids"].to(dev), b["attention_mask"].to(dev), max_tokens, stop, **kw)
        for it, ids in zip(chunk, new):
            yield it, ids


def selfcheck(agent, items, n_tokens: int = 8) -> bool:
    """Cached batched generation == plain loop (no cache, no padding) on two text rows of different length."""
    tok, model, dev = agent.processor.tokenizer, agent.model, agent.device
    text = sorted((it for it in items if not it.get("n_images")), key=lambda it: len(it["ids"]))
    if len(text) < 2:
        print("selfcheck: not enough text rows, skipped", flush=True)
        return True
    pair = [text[0], text[len(text) // 2]]
    fast = {id(it): ids for it, ids in generate(agent, pair, n_tokens, 10 ** 9)}
    emb = model.encoder.get_input_embeddings().weight
    ok = True
    for it in pair:
        cur = list(it["ids"])
        for _ in range(len(fast[id(it)])):
            x = torch.tensor([cur], device=dev)
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = model._backbone(x, torch.ones_like(x)).last_hidden_state[0, -1]
            cur.append(int((h.to(emb.dtype) @ emb.t()).float().argmax()))
        slow = cur[len(it["ids"]):]
        same = slow == fast[id(it)]
        ok &= same
        print("selfcheck row len %d: %s | cached %r | plain %r" % (len(it["ids"]), "OK" if same else "MISMATCH",
                                                                   tok.decode(fast[id(it)]), tok.decode(slow)), flush=True)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--mix-config", default="", help="take the dataset names from a configs/mix_*.json")
    ap.add_argument("--root", default=os.path.join(WORKSPACE, "data", "vqa"))
    ap.add_argument("--per-dataset", type=int, default=800)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--batch-tokens", type=int, default=60000, help="padded prompt tokens per generation batch")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--no-selfcheck", action="store_true")
    ap.add_argument("--shard", default="", help="i/n: take every n-th dataset starting at i (one process per GPU)")
    a = ap.parse_args(argv)

    from laya.vlm import THINK_PROMPT, load_vlm
    from laya.visionjev_log import log_experiment
    from laya.vlm_train import load_jsonl_examples, make_item

    if a.mix_config:
        cfg = json.load(open(a.mix_config))
        names = [d for dom in cfg["domains"].values() for d in dom["datasets"]
                 if os.path.exists(os.path.join(a.root, d, "train.jsonl"))]
    else:
        names = [n for n in a.datasets.split(",") if n]
    names = shard(names, a.shard)
    agent = load_vlm(a.checkpoint, device=a.device)
    if a.max_len:
        agent.cfg["max_len"] = agent.processor.laya_max_len = a.max_len
    agent.model.eval()
    tok = agent.processor.tokenizer
    think_ids = tok(THINK_PROMPT, add_special_tokens=False)["input_ids"]
    rng = random.Random(a.seed)
    stats, t0, checked = {}, time.time(), a.no_selfcheck
    for name in names:
        exs = [e for e in load_jsonl_examples(a.root, name, "train") if e.get("id")]
        rng.shuffle(exs)
        items = []
        for ex in exs[:a.per_dataset]:
            try:
                it = make_item(agent.processor, ex, rng, shuffle=False)
            except (OSError, ValueError) as e:
                print("skipping %s: %s" % (ex.get("id"), e), flush=True)
                continue
            it["ids"] = list(it["ids"]) + think_ids
            it["ex_id"] = ex["id"]
            items.append(it)
        if not items:
            continue
        if not checked:
            if not selfcheck(agent, items):
                raise SystemExit("selfcheck failed: cached generation differs from the plain loop")
            checked = True
        out_dir = os.path.join(a.root, name, "notes")
        os.makedirs(out_dir, exist_ok=True)
        n_tok = n = 0
        with open(os.path.join(out_dir, a.tag + ".jsonl"), "w") as f:
            for it, ids in generate(agent, items, a.max_tokens, a.batch_tokens):
                notes = trim_notes(tok.decode(ids, skip_special_tokens=True))
                if notes:
                    f.write(json.dumps({"id": it["ex_id"], "notes": notes, "tokens": len(ids)}, ensure_ascii=False) + "\n")
                    n, n_tok = n + 1, n_tok + len(ids)
        stats[name] = {"rows": n, "mean_tokens": round(n_tok / max(1, n), 1)}
        print("%s: %d notes, mean %.1f tokens (%.1f min)" % (name, n, n_tok / max(1, n), (time.time() - t0) / 60), flush=True)
    log_experiment("notes", "notes:%s:%s%s" % (a.tag, os.path.basename(os.path.normpath(a.checkpoint)),
                                              ":" + a.shard if a.shard else ""),
                   config={"per_dataset": a.per_dataset, "max_tokens": a.max_tokens, "tag": a.tag, "seed": a.seed},
                   metrics={"datasets": stats, "minutes": round((time.time() - t0) / 60, 1)}, datasets=list(stats),
                   checkpoint=a.checkpoint)


if __name__ == "__main__":
    main()
