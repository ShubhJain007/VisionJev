"""Training sketch for the VLM-backed decision model (``laya.vlm``), SmolVLM or ModernVBERT.

Fine-tunes on multiple-choice VQA with the same objective as the text model's notebooks: a soft
cross-entropy term plus a proper-scoring-rule policy-gradient term over noisy logits (``proper_reward``).
Options are shuffled per example so a causal backbone cannot learn a position prior (harmless for the
bidirectional one). Nothing here depends on the backbone family: the sequence builder and the model's
readout follow the processor and checkpoint (``laya.vlm.processor_readout``).

Data sources (adapters take one HF ``datasets`` row each):
  * A-OKVQA (``HuggingFaceM4/A-OKVQA``)      -> ``choice``
  * ScienceQA (``derek-thomas/ScienceQA``)   -> ``choice`` (image optional, hint as context)
  * VQAv2 yes/no (``HuggingFaceM4/VQAv2``)   -> ``noul`` with soft target = fraction of "yes" votes

Prepared datasets (``load_jsonl_examples``): ``<root>/<name>/<split>.jsonl`` + ``images/``, one record per line
``{"id", "image", "state_text", "question": {"type", "instructions", "criteria"}, "label"}``. ``modal_app.py``
runs ``finetune`` on these from the ``laya-datasets`` volume.

Smoke run on a tiny synthetic batch (no downloads beyond the backbone):
    python -m laya.vlm_train --synthetic --steps 3 --freeze head
    python -m laya.vlm_train --synthetic --steps 3 --freeze head --backbone ModernVBERT/modernvbert
"""
import argparse
import concurrent.futures
import copy
import functools
import json
import math
import os
import random
import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from .common import QTYPES, ece_score, proper_reward, render_options
from .vlm import (ABSTAIN_REASONS, VLMAgent, VLMDecisionModel, build_vlm_inputs, collate_vlm, end_of_text_id,
                  input_ids_sha256, sanitize_text, set_trainable, with_notes)

# ---------------------------------------------------------------------------------------------------------
# Examples: {"state": ..., "q": {"t", "ins", "crit"}, "target": [prob per option in label order]}
# ---------------------------------------------------------------------------------------------------------


def _one_hot(i: int, k: int) -> List[float]:
    return [1.0 if j == i else 0.0 for j in range(k)]


def _choice_example(state, question: str, choices: List[str], answer: int) -> Optional[Dict]:
    choices = [str(c) for c in choices]
    if len(set(choices)) != len(choices) or not 0 <= answer < len(choices):
        return None
    return {"state": state, "q": {"t": "choice", "ins": question, "crit": {c: None for c in choices}}, "target": _one_hot(answer, len(choices))}


def aokvqa_example(row: Dict) -> Optional[Dict]:
    return _choice_example({"image": row["image"]}, row["question"], row["choices"], int(row["correct_choice_idx"]))


def scienceqa_example(row: Dict) -> Optional[Dict]:
    state = {"context": row["hint"]} if row.get("hint") else {}
    if row.get("image") is not None:
        state["image"] = row["image"]
    return _choice_example(state or "", row["question"], row["choices"], int(row["answer"]))


def vqav2_yesno_example(row: Dict) -> Optional[Dict]:
    if row.get("answer_type") != "yes/no":
        return None
    votes = [a["answer"] if isinstance(a, dict) else a for a in row.get("answers") or []]
    votes = [v for v in votes if v in ("yes", "no")]
    if votes:
        p_yes = sum(v == "yes" for v in votes) / len(votes)
    elif row.get("multiple_choice_answer") in ("yes", "no"):
        p_yes = float(row["multiple_choice_answer"] == "yes")
    else:
        return None
    return {"state": {"image": row["image"]}, "q": {"t": "noul", "ins": row["question"], "crit": None}, "target": [1.0 - p_yes, p_yes]}


ADAPTERS = {
    "aokvqa": ("HuggingFaceM4/A-OKVQA", aokvqa_example),
    "scienceqa": ("derek-thomas/ScienceQA", scienceqa_example),
    "vqav2_yesno": ("HuggingFaceM4/VQAv2", vqav2_yesno_example),
}


def load_hf_examples(name: str, split: str = "train", limit: int = 1000) -> List[Dict]:
    """Stream ``limit`` usable examples from one of ``ADAPTERS`` (requires the ``datasets`` package)."""
    from datasets import load_dataset

    repo, fn = ADAPTERS[name]
    out = []
    for row in load_dataset(repo, split=split, streaming=True):
        ex = fn(row)
        if ex is not None:
            out.append(ex)
        if len(out) >= limit:
            break
    return out


def synthetic_examples(n: int = 8, seed: int = 0) -> List[Dict]:
    """Coloured squares on white: colour (choice), is-red (noul), size (score)."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    colors = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60)}
    sizes = ["small", "medium", "large"]
    out = []
    for _ in range(n):
        c, s = rng.choice(list(colors)), rng.randrange(3)
        img = Image.new("RGB", (96, 96), (255, 255, 255))
        half = (8, 20, 40)[s]
        ImageDraw.Draw(img).rectangle([48 - half, 48 - half, 48 + half, 48 + half], fill=colors[c])
        state = {"image": img}
        out.append({"state": state, "q": {"t": "choice", "ins": "What color is the square?", "crit": {k: None for k in colors}}, "target": _one_hot(list(colors).index(c), 3)})
        out.append({"state": state, "q": {"t": "noul", "ins": "Is the square red?", "crit": None}, "target": _one_hot(int(c == "red"), 2)})
        out.append({"state": state, "q": {"t": "score", "ins": "How large is the square?", "crit": sizes}, "target": _one_hot(s, 3)})
    return out


def make_item(
    processor, ex: Dict, rng: random.Random, shuffle: bool = True, max_len: Optional[int] = None,
    head_max_len: int = 256, order: Optional[List[int]] = None, rationale_max_tokens: int = 0,
    abstain_training: bool = False, notes_prob: float = 0.0,
) -> Dict:
    """Tokenize one example with a random (or the given) option order; the target is permuted to marker order.
    ``max_len`` defaults to the agent's (``laya.vlm.build_vlm_inputs``).

    With ``rationale_max_tokens > 0`` and an example ``"rationale"``, ``" Because: <rationale>"`` (at most that many
    tokens) is appended AFTER the option block and ``"rationale_start"`` marks its first token. The backbone is causal,
    so every read-out position comes before it and is computed exactly as at inference, where nothing is appended; the
    rationale only adds a language-model loss (``rationale_loss``) whose gradient reaches those positions. A rationale
    that fits is followed by the end-of-text token, so the model learns where its working stops.

    With ``notes_prob > 0`` and an example ``"notes"`` (the student's OWN generated working for this row, right or
    wrong; never the gold rationale), the notes are attached to the state with that probability (``with_notes``), so
    the read-out learns to use its working and to check it against the state (think-if-unsure, V6.3 section 4)."""
    k = len(render_options(ex["q"]))
    if order is None:
        order = list(range(k))
        if shuffle:
            rng.shuffle(order)
    state = ex["state"]
    if notes_prob > 0 and ex.get("notes") and rng.random() < notes_prob:
        state = with_notes(state, sanitize_text(processor.tokenizer, ex["notes"]))
    it = build_vlm_inputs(processor, state, ex["q"], max_len, head_max_len, option_order=order)
    if rationale_max_tokens > 0 and ex.get("rationale"):
        tok = processor.tokenizer
        r_ids = tok(" Because: " + sanitize_text(tok, ex["rationale"]), add_special_tokens=False)["input_ids"]
        eot = end_of_text_id(tok)
        if len(r_ids) < rationale_max_tokens and eot is not None:
            r_ids = r_ids + [eot]
        r_ids = r_ids[:rationale_max_tokens]
        if r_ids:
            it["rationale_start"] = len(it["ids"])
            it["ids"] = list(it["ids"]) + r_ids
    it["target"] = [ex["target"][i] for i in order]
    it["label"] = max(range(k), key=lambda j: it["target"][j])
    if abstain_training or ex.get("abstain") is not None:
        # abstain-head target for every row once abstain training is on (answerable rows are 0), with the reason index
        it["abstain"] = int(ex.get("abstain") or 0)
        r = ex.get("abstain_reason")
        it["abstain_reason"] = ABSTAIN_REASONS.index(r) if (it["abstain"] and r in ABSTAIN_REASONS) else -1
    it["qtype"] = QTYPES[ex["q"]["t"]]
    it["order"] = order
    if ex.get("value") is not None:  # optional value-head target in [0, 1] (order-free: one per state)
        v = float(ex["value"])
        if not 0.0 <= v <= 1.0:
            raise ValueError("example value must be in [0, 1], got %r" % ex["value"])
        it["value"] = v
    if ex.get("next_target") is not None:  # optional next-move target: a distribution over the same options
        nt = [float(p) for p in ex["next_target"]]
        if len(nt) != k or not all(math.isfinite(p) and p >= 0 for p in nt) or sum(nt) <= 0:
            raise ValueError("example next_target must be %d non-negative floats with a positive sum, got %r"
                             % (k, ex["next_target"]))
        total = sum(nt)
        it["next_target"] = [nt[i] / total for i in order]  # permuted to marker order, like the target
    return it


# ---------------------------------------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------------------------------------


ACT_COSTS = (1.0, -3.0, -0.5)  # utility of acting when right / wrong, and of escalating: act when P(right) > 0.625

# Writing a resumable state.pt means a full CPU copy of the weights plus optimizer, a torch.save and a volume
# commit: seconds of GPU time each. These bound the cost however small an interval a caller asks for.
MIN_STATE_MINUTES = 2.0     # never write state.pt more often than this
STATE_SAVE_OVERHEAD = 20.0  # and never more often than 20x the last write's duration (<= 5% of the wall clock)


def vlm_loss(logits, target, qtype, mask, sigma: float = 0.3, group_size: int = 4, w_ce: float = 1.0,
             w_sph: float = 0.75, act_logits=None, act_costs=ACT_COSTS, row_weight=None):
    """Proper-scoring-rule policy gradient over noisy logits + soft cross-entropy (as in the text notebooks).

    ``w_sph`` weights the spherical score in ``proper_reward`` (the original design used 0.5). ``w_ce`` can be
    annealed to 0 by the caller so training ends on proper scoring rules alone, which keeps calibration.
    ``act_logits`` (from the model's act head) adds the expected-utility term of the decide-or-escalate cost
    matrix ``act_costs``; without it the head gets no gradient. ``row_weight`` ([B], default all ones) weights each
    row's option loss: 0 masks a row out entirely (abstain examples have no correct option).
    """
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    ce_rows = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
    if row_weight is None:
        loss_rl = -(adv * logp).mean()
        loss_ce = ce_rows.mean()
    else:
        rw = row_weight.to(logits.device).float()
        denom = rw.sum().clamp(min=1.0)
        loss_rl = -(adv * logp * rw).sum() / (adv.shape[0] * denom)
        loss_ce = (ce_rows * rw).sum() / denom
    loss = loss_rl + w_ce * loss_ce
    if act_logits is not None:
        c_ok, c_bad, c_esc = act_costs
        p_act = torch.softmax(act_logits.float(), -1)[:, 0]
        with torch.no_grad():  # how right the model's own choice is, as a probability
            y = target.gather(1, logits.argmax(-1, keepdim=True)).squeeze(1)
        eu = p_act * (c_ok * y + c_bad * (1 - y)) + (1 - p_act) * c_esc
        loss = loss - eu.mean()
    return loss, r.mean()


def value_loss(value_logits: Optional[torch.Tensor], value: Optional[torch.Tensor], w_value: float = 1.0):
    """``w_value * BCE(value_logits, value)`` over the rows whose ``value`` is not NaN (``collate_vlm`` marks rows
    without a target that way); a plain ``0.0`` when the model has no value head, the batch no targets, or
    ``w_value`` is 0, so nothing about such a step changes."""
    if value_logits is None or value is None or not w_value:
        return 0.0
    value = value.to(value_logits.device).float()
    has = ~torch.isnan(value)
    if not bool(has.any()):
        return 0.0
    return w_value * torch.nn.functional.binary_cross_entropy_with_logits(value_logits.float()[has], value[has])


def abstain_loss(abstain_logits: Optional[torch.Tensor], reason_logits: Optional[torch.Tensor],
                 abstain: Optional[torch.Tensor], reason: Optional[torch.Tensor], w_abstain: float = 0.5,
                 w_reason: float = 0.2):
    """``w_abstain * BCE(abstain)`` over every row with a target, plus ``w_reason * CE(reason)`` over the abstain rows
    with a known reason (``reason`` -1 = none). ``0.0`` when the model has no abstain head or the batch no targets."""
    if abstain_logits is None or abstain is None or not w_abstain:
        return 0.0
    a = abstain.to(abstain_logits.device).float()
    loss = w_abstain * torch.nn.functional.binary_cross_entropy_with_logits(abstain_logits.float(), a)
    if reason is not None and w_reason:
        r = reason.to(abstain_logits.device)
        has = r >= 0
        if bool(has.any()):
            loss = loss + w_reason * torch.nn.functional.cross_entropy(reason_logits.float()[has], r[has])
    return loss


def rationale_loss(hidden: Optional[torch.Tensor], input_ids: torch.Tensor, rationale_mask: Optional[torch.Tensor],
                   emb_weight: torch.Tensor, w: float = 0.3, chunk: int = 512):
    """``w * CE(h_{p-1} E^T, x_p)`` over rationale token positions ``p`` (``collate_vlm``'s ``rationale_mask``), with the
    LM head tied to the input embeddings ``E`` (Qwen3-VL 2B / 4B tie them). ``0.0`` when there is nothing to train."""
    if hidden is None or rationale_mask is None or not w:
        return 0.0
    total, n = rationale_nll(hidden, input_ids, rationale_mask, emb_weight, chunk)
    return w * total / n if n else 0.0


def rationale_nll(hidden: torch.Tensor, input_ids: torch.Tensor, rationale_mask: torch.Tensor,
                  emb_weight: torch.Tensor, chunk: int = 512):
    """``(sum of token cross-entropies, number of rationale tokens)`` of ``rationale_loss``, so the forward passes of one
    step (``length_groups``) can be pooled into a single mean: ``w * sum(sums) / sum(counts)``. ``(0.0, 0)`` when the
    batch holds no rationale token."""
    m = rationale_mask.to(hidden.device)[:, 1:]
    if not bool(m.any()):
        return 0.0, 0
    h = hidden[:, :-1][m]
    tgt = input_ids.to(hidden.device)[:, 1:][m]
    total, n = 0.0, h.shape[0]
    for i in range(0, n, chunk):  # a 151k-way softmax per token: chunk to bound the logits' memory
        logits = (h[i:i + chunk].to(emb_weight.dtype) @ emb_weight.t()).float()
        total = total + torch.nn.functional.cross_entropy(logits, tgt[i:i + chunk], reduction="sum")
    return total, n


def next_loss(next_logits: Optional[torch.Tensor], next_target: Optional[torch.Tensor], w_next: float = 0.15):
    """``w_next * soft cross-entropy(next_logits, next_target)`` over the rows whose ``next_target`` is not NaN
    (``collate_vlm`` marks rows without one that way; padded options carry target 0 and logit -1e4). A plain
    ``0.0`` when the model has no next-move head, the batch no targets, or ``w_next`` is 0, so nothing about such a
    step changes."""
    if next_logits is None or next_target is None or not w_next:
        return 0.0
    next_target = next_target.to(next_logits.device).float()
    has = ~torch.isnan(next_target).any(-1)
    if not bool(has.any()):
        return 0.0
    logp = torch.log_softmax(next_logits.float()[has], -1)
    return w_next * -(next_target[has] * logp).sum(-1).mean()


def _to(b: Dict, device, dtype) -> Dict:
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
            if k == "pixel_values":
                v = v.to(dtype)
        out[k] = v
    return out


def _forward_kwargs(b) -> Dict:
    return dict(
        input_ids=b["input_ids"], attention_mask=b["attention_mask"], marker_pos=b["marker_pos"],
        marker_mask=b["marker_mask"], qtype=b["qtype"],
        pixel_values=b["pixel_values"], pixel_attention_mask=b["pixel_attention_mask"], option_span=b["option_span"],
        raw_pixels=b.get("raw_pixels"), image_mask=b.get("image_mask"), image_grid_thw=b.get("image_grid_thw"),
        # the appended rationale is for the LM loss only: hide it from the (bidirectional) head and the act head
        readout_mask=(b["attention_mask"] * (~b["rationale_mask"]).long()) if "rationale_mask" in b else None,
    )


def _forward(model, b):
    return model(**_forward_kwargs(b))


def length_groups(lengths: Sequence[int], max_groups: int = 4, token_budget: int = 0,
                  overhead: int = 256) -> List[List[int]]:
    """Split one batch's rows into at most ``max_groups`` groups of similar length, minimising the padded tokens
    ``sum(rows_g * longest_g)`` plus ``overhead`` tokens per group (a forward pass has a fixed cost, so tiny gains do
    not earn a split). Groups are contiguous in length order, which is optimal for this cost (dynamic programme).
    ``token_budget`` (> 0) forbids a group whose padded size exceeds it, unless it is a single row.

    Why: a batch is padded to its longest row, and with a few 4,000-token rows in the mix 75% of the tokens computed
    were padding (B2, 2026-09-30). Splitting *inside* the batch removes the padding without changing which rows a
    step trains on, their weights, or the number of updates.
    """
    n = len(lengths)
    order = sorted(range(n), key=lambda i: lengths[i])
    ls = [lengths[i] for i in order]
    if max_groups <= 1 or n <= 1:
        return [order]
    inf = float("inf")

    def cost(i, j):  # rows i..j-1 of the sorted order as one group
        padded = ls[j - 1] * (j - i)
        return padded + overhead if (not token_budget or padded <= token_budget or j - i == 1) else inf

    g_max = min(max_groups, n)
    best = [[inf] * (n + 1) for _ in range(g_max + 1)]
    cut = [[0] * (n + 1) for _ in range(g_max + 1)]
    best[0][0] = 0.0
    for g in range(1, g_max + 1):
        for j in range(1, n + 1):
            for i in range(g - 1, j):
                c = best[g - 1][i] + cost(i, j)
                if c < best[g][j]:
                    best[g][j], cut[g][j] = c, i
    g = min(range(1, g_max + 1), key=lambda k: best[k][n])
    if best[g][n] == inf:  # the budget cannot be met with max_groups groups: take the most groups allowed
        g = g_max
        best[g][n] = 0.0
        cut_pts = [round(k * n / g) for k in range(g + 1)]
        return [order[a:b] for a, b in zip(cut_pts, cut_pts[1:]) if b > a]
    groups, j = [], n
    while g > 0:
        i = cut[g][j]
        groups.append(order[i:j])
        g, j = g - 1, i
    return groups[::-1]


LABEL_KEYS = ("target", "qtype", "marker_mask", "label", "abstain", "abstain_reason", "value", "next_target")


def _collate_train(items, pad_id, max_groups: int = 1, token_budget: int = 0):
    """``{"parts": [(row indices, collated group)], "dataset": [...], label tensors for the whole batch}``. With
    ``max_groups`` 1 there is one part holding every row (the batch as it always was)."""
    if max_groups <= 1 or len(items) <= 1:
        b = collate_vlm(items, pad_id)
        parts = [(torch.arange(len(items)), b)]
        labels = b
    else:
        groups = length_groups([len(it["ids"]) for it in items], max_groups, token_budget)
        parts = [(torch.tensor(g), collate_vlm([items[i] for i in g], pad_id)) for g in groups]
        labels = parts[0][1] if len(parts) == 1 else collate_vlm(items, pad_id, with_pixels=False)
    out = {k: labels[k] for k in LABEL_KEYS if k in labels}
    out["parts"] = parts
    out["dataset"] = [it["dataset"] for it in items]
    return out


class GroupRunner:
    """Runs the forward passes of one training step (the length groups of ``_collate_train``) and returns the whole
    batch's outputs in row order, so the losses are computed once over the batch exactly as without groups.

    ``devices`` with more than one entry spreads the groups over several GPUs inside this process: every extra device
    holds a frozen copy of the model, and each step its trainable tensors are the primary model's parameters moved
    there (``torch.func.functional_call``), so gradients flow back to the one set of parameters the optimizer owns.
    One optimizer, one loss, the same rows per step as on a single device: nothing about the optimisation changes.
    """

    def __init__(self, model, devices: Optional[Sequence] = None, w_rationale: float = 0.0,
                 ckpt_above_tokens: Optional[int] = None):
        self.model, self.w_rationale, self.ckpt_above = model, w_rationale, ckpt_above_tokens
        primary = next(model.parameters()).device
        self.devices = [torch.device(d) for d in (devices or [primary])]
        self.dtype = model.encoder.dtype
        self.emb = model.encoder.get_input_embeddings().weight
        self.emb_name = next((n for n, p in model.named_parameters() if p is self.emb), None)
        self.trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.replicas = []
        for d in self.devices[1:]:
            for m in model.modules():  # outputs kept from an earlier step are not leaves and would break deepcopy
                for k in ("last_hidden", "last_value", "last_next", "last_abstain", "last_reason", "last_position_ids"):
                    if hasattr(m, k):
                        setattr(m, k, None)
            r = copy.deepcopy(model).to(d)
            for p in r.parameters():
                p.requires_grad_(False)
            r.train(model.training)
            self.replicas.append(r)
        self._ckpt_on = None
        self.pool = concurrent.futures.ThreadPoolExecutor(len(self.devices)) if len(self.devices) > 1 else None

    def _set_checkpointing(self, on: bool):
        if on == self._ckpt_on:
            return
        for m in [self.model] + self.replicas:
            if on:
                m.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                m.encoder.gradient_checkpointing_disable()
        self._ckpt_on = on

    def _checkpointing(self) -> bool:
        if self._ckpt_on is not None:
            return self._ckpt_on
        enc = self.model.encoder
        return bool(getattr(getattr(enc, "inner", enc), "is_gradient_checkpointing", False))

    def _one(self, module, overrides, mb, device):
        b = _to(mb, device, self.dtype)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if overrides is None:
                logits, act = _forward(module, b)
            else:
                logits, act = torch.func.functional_call(module, overrides, args=(), kwargs=_forward_kwargs(b))
        out = {"logits": logits, "act": act, "value": module.last_value, "next": module.last_next,
               "abstain": module.last_abstain, "reason": module.last_reason, "rat": None}
        if self.w_rationale and "rationale_mask" in b:
            emb = self.emb if overrides is None else overrides.get(self.emb_name, module.encoder.get_input_embeddings().weight)
            out["rat"] = rationale_nll(module.last_hidden, b["input_ids"], b["rationale_mask"], emb)
        module.last_hidden = None  # the largest tensor of the pass; nothing else needs it
        module.last_value = module.last_next = module.last_abstain = module.last_reason = None
        return out

    def __call__(self, parts) -> Dict:
        tokens = sum(mb["input_ids"].numel() for _, mb in parts)
        load = [0] * len(self.devices)
        plan = [[] for _ in self.devices]
        for j in sorted(range(len(parts)), key=lambda j: -parts[j][1]["input_ids"].numel()):
            d = min(range(len(load)), key=load.__getitem__)  # biggest group first onto the least loaded device
            plan[d].append(j)
            load[d] += parts[j][1]["input_ids"].numel()
        if self.ckpt_above is not None:
            # a device holds the graphs of all its passes until the backward: the per-device total has to fit (a 4B
            # model keeps ~3.5 MB of activations per token without checkpointing; image rows cost more: B3's smoke
            # ran out of memory on an 80 GB GPU at a 12k-token threshold, 2026-10-01)
            self._set_checkpointing(max(load) > self.ckpt_above)
        # Gradient checkpointing recomputes each layer during the backward pass, AFTER functional_call has put a
        # replica's own frozen weights back: the recomputation would use stale weights and send no gradient to the
        # live ones. So a checkpointed step runs entirely on the primary device.
        multi = self.pool is not None and not self._checkpointing()
        if not multi:
            outs = [self._one(self.model, None, mb, self.devices[0]) for _, mb in parts]
        else:

            def work(d):
                if d == 0:
                    return [(j, self._one(self.model, None, parts[j][1], self.devices[0])) for j in plan[0]]
                dev = self.devices[d]
                overrides = {n: p.to(dev) for n, p in self.trainable}  # differentiable copies of the live weights
                return [(j, self._one(self.replicas[d - 1], overrides, parts[j][1], dev)) for j in plan[d]]

            done = dict(x for res in self.pool.map(work, range(len(self.devices))) for x in res)
            outs = [done[j] for j in range(len(parts))]
        return self._merge(outs, [idx for idx, _ in parts], tokens)

    def _merge(self, outs, idxs, tokens) -> Dict:
        dev = self.devices[0]
        order = torch.cat(idxs).to(dev)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=dev)
        kmax = max(o["logits"].shape[1] for o in outs)

        def cat(key, pad_k=False):
            xs = [o[key] for o in outs]
            if any(x is None for x in xs):
                return None
            xs = [x.to(dev) for x in xs]
            if pad_k:
                xs = [torch.nn.functional.pad(x, (0, kmax - x.shape[1]), value=-1e4) for x in xs]
            return torch.cat(xs)[inv]

        res = {"logits": cat("logits", True), "act": cat("act"), "value": cat("value"), "next": cat("next", True),
               "abstain": cat("abstain"), "reason": cat("reason"), "tokens": tokens, "rationale": 0.0}
        rats = [o["rat"] for o in outs if o["rat"] is not None and o["rat"][1]]
        if rats:
            total = sum(t.to(dev) for t, _ in rats)
            res["rationale"] = self.w_rationale * total / sum(n for _, n in rats)
        return res


def _single_thread_worker(_):
    torch.set_num_threads(1)  # avoid CPU oversubscription across loader workers


def group_weights(groups: Dict[str, Sequence], weights: Optional[Dict[str, float]] = None,
                  size_alpha: float = 0.0) -> Dict[str, float]:
    """Unnormalised sampling weight per group: ``weights.get(k, 1.0) * len(group_k) ** size_alpha``.

    The defaults give every group the same weight (the equal sampling every earlier run used); ``size_alpha=1``
    draws in proportion to size (a plain shuffle of the union), ``0.5`` in proportion to its square root.
    """
    weights = weights or {}
    return {k: float(weights.get(k, 1.0)) * float(len(g)) ** size_alpha for k, g in groups.items()}


def mix_probabilities(groups: Dict[str, Sequence], weights: Optional[Dict[str, float]] = None,
                      size_alpha: float = 0.0) -> Dict[str, float]:
    """``group_weights`` normalised to sum to 1 (the effective mix before any ``max_passes`` cap)."""
    w = group_weights(groups, weights, size_alpha)
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


class ItemStream(torch.utils.data.IterableDataset):
    """Endless stream of shuffled-option items, sampling each ``balance_key`` group (dataset) by weight.

    Group ``k`` is drawn with probability proportional to ``weights.get(k, 1.0) * len(group_k) ** size_alpha``
    (``group_weights``); the defaults draw every group equally. ``max_passes`` caps how many times a group is
    sampled (in passes over it); a capped group leaves the mix and the others keep their relative shares. The
    stream ends when every group is capped. With several loader workers the cap is split evenly between them.
    ``consumed`` (samples already taken per group in an earlier attempt of a resumed run) counts against those
    caps.
    """

    def __init__(self, processor, examples: List[Dict], seed: int = 0, balance_key: str = "dataset",
                 max_passes: Optional[float] = None, consumed: Optional[Dict[str, int]] = None,
                 weights: Optional[Dict[str, float]] = None, size_alpha: float = 0.0,
                 augment: Optional[Callable[[Dict, random.Random], Dict]] = None, **item_kw):
        self.processor, self.seed, self.item_kw, self.max_passes = processor, seed, item_kw, max_passes
        self.augment = augment  # e.g. laya.visionjev_augment.augment: a label-exact variant of each drawn example
        self.consumed = consumed or {}
        self.groups: Dict[str, List[Dict]] = {}
        for ex in examples:
            self.groups.setdefault(ex.get(balance_key, "_"), []).append(ex)
        self.keys = sorted(self.groups)
        self.weights = group_weights(self.groups, weights, size_alpha)
        unknown = sorted(set(weights or {}) - set(self.groups))
        if unknown:
            print("mix weights for groups not in the data (ignored): %s" % unknown)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        n_workers = wi.num_workers if wi else 1
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        left = {k: (self.max_passes * len(g) / n_workers if self.max_passes else math.inf) for k, g in self.groups.items()}
        for k in left:  # a resumed run continues the caps where the previous attempt stopped
            left[k] -= self.consumed.get(k, 0) / n_workers
        while True:
            keys = [k for k in self.keys if left[k] >= 1]
            if not keys:
                return
            key = rng.choices(keys, weights=[self.weights[k] for k in keys])[0]
            left[key] -= 1
            ex = rng.choice(self.groups[key])
            try:
                if self.augment is not None:
                    ex = self.augment(ex, rng)
                it = make_item(self.processor, ex, rng, **self.item_kw)
            except (OSError, ValueError) as e:  # unreadable image / over-long question
                print("skipping %s: %s" % (ex.get("id"), e))
                continue
            it["dataset"] = key
            yield it


def train(
    model: VLMDecisionModel,
    processor,
    examples: List[Dict],
    steps: int = 100,
    batch_size: int = 2,
    freeze: str = "head",
    n_last: int = 4,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-5,
    sigma: float = 0.3,
    device: Optional[str] = None,
    seed: int = 0,
    log_every: int = 1,
    max_minutes: Optional[float] = None,
    num_workers: int = 0,
    prefetch_factor: int = 4,
    warmup: int = 0,
    eval_fn: Optional[Callable[[int], object]] = None,
    eval_every: int = 0,
    max_passes: Optional[float] = None,
    stats: Optional[Dict] = None,
    group_size: int = 4,
    w_sph: float = 0.75,
    w_ce: float = 1.0,
    w_ce_schedule: str = "const",
    sigma_end: Optional[float] = None,
    train_act: bool = False,
    balance_key: str = "dataset",
    resume: Optional[Dict] = None,
    save_state_fn: Optional[Callable[[int, Dict], None]] = None,
    save_state_every_min: float = 0.0,
    mix_weights: Optional[Dict[str, float]] = None,
    mix_alpha: float = 0.0,
    w_value: float = 1.0,
    w_next: float = 0.15,
    augment: Optional[Callable[[Dict, random.Random], Dict]] = None,
    w_rationale: float = 0.0,
    rationale_max_tokens: int = 96,
    w_abstain: float = 0.0,
    w_reason: float = 0.2,
    notes_prob: float = 0.0,
    length_groups_n: int = 1,
    group_token_budget: int = 0,
    ckpt_above_tokens: Optional[int] = None,
    devices: Optional[Sequence[str]] = None,
) -> List[float]:
    """One optimizer, one loss; stops at ``steps``, ``max_minutes``, or when every dataset hits ``max_passes``.
    ``augment(example, rng)`` (optional) replaces each drawn training example with a variant (``ItemStream``).

    Speed options that leave the optimisation unchanged (same rows per step, same loss, same number of updates):
    ``length_groups_n`` > 1 runs each batch as up to that many forward passes of similar-length rows
    (``length_groups``; ``group_token_budget`` caps one pass's padded tokens) instead of padding every row to the
    longest; ``devices`` (e.g. ``["cuda:0", "cuda:1"]``) spreads those passes over several GPUs in this process
    (``GroupRunner``); ``ckpt_above_tokens`` turns gradient checkpointing on only for steps whose padded tokens
    exceed it (``None`` leaves the encoder's setting alone).

    bf16 autocast on CUDA. Returns per-step losses; ``stats`` (if given) receives samples per dataset, steps/s
    and the fraction of training time spent waiting on the data loader (both excluding eval time).
    The LR follows linear warmup then cosine decay to 10%, on whichever of step or wall-clock progress is further.

    Objective options (all default to the previous behaviour): ``group_size`` and ``sigma`` (annealed to
    ``sigma_end`` over training when given) control the exploration noise, ``w_sph`` the spherical score, and
    ``w_ce_schedule="anneal"`` holds the cross-entropy weight at ``w_ce`` for the first 30% of progress then
    decays it linearly to 0 by 80%. ``train_act`` trains the act/escalate head on its cost matrix.

    Value head: when the model has one (config ``"value_head": true``) and a batch carries examples with a
    ``"value"`` (a float in [0, 1], e.g. whether the episode this game state came from was won), the loss adds
    ``w_value * BCE(value_logit, value)`` averaged over those examples only (``value_loss``); examples without one
    contribute nothing to it, and a batch with none (or a model without the head) trains exactly as before.

    Next-move head (auxiliary, never used for play; KataGo's opponent-move target, arXiv 1902.10565 section 3.4):
    when the model has one (config ``"next_head": true``) and a batch carries examples with a ``"next_target"``
    (a distribution over the same options: the expert's move at the next step of the same trajectory), the loss
    adds ``w_next * soft cross-entropy(next_logits, next_target)`` over those examples only (``next_loss``); as for
    the value head, examples without one contribute nothing and a batch with none trains exactly as before.

    Sampling mix: dataset ``k`` is drawn with probability proportional to
    ``mix_weights.get(k, 1.0) * n_k ** mix_alpha`` (``ItemStream``); the defaults draw every dataset equally.

    ``eval_fn(step)`` is called every ``eval_every`` steps and should return a truthy value when it really
    evaluated, so a caller can use it as a cheap progress probe at a small ``eval_every`` without paying for a
    state write on every call.

    Durability: ``save_state_fn(step, state)`` is called every ``save_state_every_min`` minutes and after every
    eval that ran with ``{"step", "opt", "rng", "losses", "seen", "elapsed_s", "eval_s"}``, so the caller can
    write a resumable checkpoint; passing that dict back as ``resume`` continues the run (optimizer, LR schedule,
    loss history, per-dataset sample counts and the time budget, which then counts across attempts). The sampler
    is an endless random stream, so a resumed run re-seeds it rather than replaying the same order.
    ``save_state_every_min`` is clamped to ``MIN_STATE_MINUTES`` and backed off further when writes are slow
    (``STATE_SAVE_OVERHEAD``), so no caller can spend most of its GPU time checkpointing.
    """
    device = torch.device(device or next(model.parameters()).device)
    torch.manual_seed(seed)
    n_train = set_trainable(model, freeze, n_last=n_last)
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": head, "lr": lr_head}] + ([{"params": enc, "lr": lr_backbone}] if enc else [])
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    if resume:
        opt.load_state_dict(resume["opt"])
    model.to(device).train()
    dtype = model.encoder.dtype
    amp = device.type == "cuda"
    stream = ItemStream(processor, examples, seed + (resume["step"] if resume else 0), balance_key=balance_key,
                        max_passes=max_passes, consumed=resume["seen"] if resume else None,
                        weights=mix_weights, size_alpha=mix_alpha, augment=augment,
                        rationale_max_tokens=rationale_max_tokens if w_rationale else 0,
                        abstain_training=bool(w_abstain), notes_prob=notes_prob)
    probs = mix_probabilities(stream.groups, mix_weights, mix_alpha)
    print("sampling mix (alpha=%g, before max_passes): %s"
          % (mix_alpha, ", ".join("%s %.3f" % (k, probs[k]) for k in stream.keys)), flush=True)
    loader = torch.utils.data.DataLoader(
        stream,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=functools.partial(_collate_train, pad_id=processor.tokenizer.pad_token_id,
                                     max_groups=length_groups_n, token_budget=group_token_budget),
        pin_memory=amp,
        persistent_workers=num_workers > 0,
        worker_init_fn=_single_thread_worker,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    runner = GroupRunner(model, devices, w_rationale=w_rationale, ckpt_above_tokens=ckpt_above_tokens)
    print("training %d params (freeze=%s) on %s, batch %d, amp=%s, length groups %d, checkpointing %s"
          % (n_train, freeze, "+".join(str(d) for d in runner.devices), batch_size, amp, length_groups_n,
             "as configured" if ckpt_above_tokens is None else "above %d tokens" % ckpt_above_tokens))
    tokens_run = 0
    losses, t0, step, wait, t_eval = [], time.time(), 0, 0.0, 0.0
    seen: Dict[str, int] = {}
    if resume:
        losses, step, seen, t_eval = list(resume["losses"]), resume["step"], dict(resume["seen"]), resume["eval_s"]
        t0 = time.time() - resume["elapsed_s"]  # the budget counts training time across attempts
        random.setstate(resume["rng"]["py"])
        np.random.set_state(resume["rng"]["np"])
        torch.set_rng_state(resume["rng"]["torch"])
        print("resuming at step %d (%.1f min already spent, %d samples seen)"
              % (step, resume["elapsed_s"] / 60, sum(seen.values())), flush=True)
    budget = max_minutes * 60 if max_minutes else None
    t_state, save_s = time.time(), 0.0
    min_state_gap = 0.0
    if save_state_every_min:
        min_state_gap = max(save_state_every_min, MIN_STATE_MINUTES) * 60
        if save_state_every_min < MIN_STATE_MINUTES:
            print("save_state_every_min=%.3f min is below the %.1f min floor (a state write costs seconds of GPU "
                  "time); using %.1f min" % (save_state_every_min, MIN_STATE_MINUTES, MIN_STATE_MINUTES), flush=True)

    def state(step_):
        return {"step": step_, "opt": opt.state_dict(), "losses": losses, "seen": seen,
                "elapsed_s": time.time() - t0, "eval_s": t_eval,
                "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}}

    def do_save(step_):
        nonlocal t_state, save_s
        ts = time.time()
        model.eval()
        save_state_fn(step_, state(step_))
        model.train()
        save_s, t_state = time.time() - ts, time.time()

    def save_due():
        # Beyond the floor, back off when writes are slow: state.pt never eats more than ~1/20 of the wall clock,
        # however often it is asked for.
        return time.time() - t_state >= max(min_state_gap, STATE_SAVE_OVERHEAD * save_s)

    t_fetch = time.time()
    for batch in loader:
        wait += time.time() - t_fetch
        for name in batch["dataset"]:
            seen[name] = seen.get(name, 0) + 1
        progress = step / max(1, steps)
        if budget:
            progress = max(progress, (time.time() - t0) / budget)
        if progress >= 1.0:
            break
        f = min(1.0, (step + 1) / warmup) if warmup else 1.0
        f *= 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
        for g, lr in zip(groups, base_lrs):
            g["lr"] = lr * f
        w_ce_now = w_ce
        if w_ce_schedule == "anneal":  # hold, then decay to 0 between 30% and 80% of progress
            w_ce_now = w_ce * min(1.0, max(0.0, (0.8 - progress) / 0.5))
        sigma_now = sigma if sigma_end is None else sigma + (sigma_end - sigma) * min(1.0, progress)
        out = runner(batch["parts"])  # the batch's forward passes, merged back into row order
        tokens_run += out["tokens"]
        b = _to({k: batch[k] for k in LABEL_KEYS if k in batch}, device, dtype)
        logits, act = out["logits"], out["act"]
        row_w = (1.0 - b["abstain"].float()) if (w_abstain and "abstain" in b) else None  # abstain rows: no option loss
        loss, reward = vlm_loss(logits, b["target"], b["qtype"], b["marker_mask"], sigma=sigma_now,
                                group_size=group_size, w_ce=w_ce_now, w_sph=w_sph,
                                act_logits=act if train_act else None, row_weight=row_w)
        if not train_act:
            loss = loss + 0.0 * act.float().sum()
        loss = loss + value_loss(out["value"], b.get("value"), w_value)
        loss = loss + next_loss(out["next"], b.get("next_target"), w_next)
        if w_abstain and "abstain" in b:
            loss = loss + abstain_loss(out["abstain"], out["reason"], b["abstain"], b.get("abstain_reason"),
                                       w_abstain, w_reason)
        loss = loss + out["rationale"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0))
        opt.step()
        losses.append(loss.item())
        if len(losses) > 200:  # a batch far worse than the recent run: say what it held (collapse forensics)
            base_avg = sum(losses[-201:-1]) / 200
            if losses[-1] > 3 * max(0.2, base_avg):
                from collections import Counter

                print("loss spike step %d: %.3f (recent avg %.3f, grad norm %.2f) batch: %s" % (
                    step, losses[-1], base_avg, gnorm, dict(Counter(batch["dataset"]))), flush=True)
        if log_every and step % log_every == 0:
            recent = losses[-log_every:]
            print("step %d | %.1f min | loss %.4f (avg %.4f) | reward %.3f | lr %.2e | w_ce %.2f | sigma %.2f | "
                  "grad %.2f | data wait %.0f%%" % (step, (time.time() - t0) / 60, losses[-1], sum(recent) / len(recent),
                                                    reward.item(), groups[0]["lr"], w_ce_now, sigma_now, gnorm,
                                                    100 * wait / max(1e-6, time.time() - t0)), flush=True)
        step += 1
        if eval_fn is not None and eval_every and step % eval_every == 0:
            te = time.time()
            model.eval()
            ran = eval_fn(step)
            model.train()
            t_eval += time.time() - te
            if ran == "stop":  # eval_fn's early stop (e.g. no calibration improvement for --patience evals)
                print("early stop at step %d" % step, flush=True)
                step_stop = True
            else:
                step_stop = False
            # Only a real eval earns a state write. ``eval_fn`` is often a cheap probe called every few steps
            # that decides for itself whether to evaluate (it returns falsey when it did not), and writing
            # state.pt costs seconds of GPU time.
            if save_state_fn is not None and ran:
                do_save(step)
        if save_state_fn is not None and min_state_gap and save_due():
            do_save(step)
        if eval_fn is not None and eval_every and step % eval_every == 0 and step_stop:
            break
        t_fetch = time.time()
    model.eval()
    if stats is not None:
        train_s = max(1e-6, time.time() - t0 - t_eval)
        stats.update(steps=step, samples_per_dataset=seen, train_minutes=train_s / 60, eval_minutes=t_eval / 60,
                     steps_per_s=step / train_s, data_wait_frac=wait / train_s,
                     padded_tokens_per_step=tokens_run / max(1, step))
    return losses


# ---------------------------------------------------------------------------------------------------------
# Prepared VQA datasets: <root>/<name>/{images/, <split>.jsonl, _READY}
# ---------------------------------------------------------------------------------------------------------


def jsonl_example(rec: Dict, root: str, dataset: str = "") -> Optional[Dict]:
    """``{"id", "image", "state_text", "question": {type, instructions, criteria}, "label"}`` -> training example.

    ``"images"`` (a list of paths) replaces ``"image"`` for a multi-image record, e.g. NLVR2's pairs.
    ``label`` indexes the rendered options (choice: criteria order; score: level; noul: 0=false, 1=true).
    An optional ``"target"`` (a probability per option, same order) replaces the one-hot target, e.g. an expert
    policy's action distribution; ``label`` is still used for accuracy. An optional ``"value"`` (in [0, 1]) is the
    value-head target for the state, an optional ``"next_target"`` (a probability per option, same order) the
    next-move head's (``train``).
    """
    qdef = rec["question"]
    q = VLMAgent._to_internal(qdef)
    if q["t"] == "choice" and len(q["crit"]) != len(qdef["criteria"]):
        return None  # duplicate choice strings collapse in the dict form
    k = len(render_options(q))
    label = int(rec["label"])
    if not 0 <= label < k:
        return None
    state = {}
    if rec.get("image"):
        state["image"] = os.path.join(root, rec["image"])
    elif rec.get("images"):
        state["images"] = [os.path.join(root, p) for p in rec["images"]]
    if rec.get("state_text"):
        state["context"] = rec["state_text"]
    target = _one_hot(label, k)
    soft = rec.get("target")
    if soft is not None and len(soft) == k and min(soft) >= 0 and sum(soft) > 0:
        target = [float(p) / sum(soft) for p in soft]
    ex = {"state": state or "", "q": q, "target": target, "label": label, "dataset": dataset, "id": rec.get("id")}
    if rec.get("value") is not None:  # optional value-head target in [0, 1]
        ex["value"] = float(rec["value"])
    if rec.get("rationale"):  # optional worked reason, trained after the read-out span (rationale loss)
        ex["rationale"] = str(rec["rationale"])
    if rec.get("abstain") is not None:  # optional abstain-head target (1 = the evidence does not decide)
        ex["abstain"] = int(rec["abstain"])
        ex["abstain_reason"] = rec.get("abstain_reason")
    if rec.get("next_target") is not None:  # optional next-move-head target, checked by make_item
        ex["next_target"] = [float(p) for p in rec["next_target"]]
    return ex


_ID_EPISODE_STEP = re.compile(r"^(?P<episode>.+)-(?P<step>\d+)$")


def episode_step(rec: Dict) -> Optional[tuple]:
    """``(episode, step)`` of a recorded game frame, or None when the record does not say.

    Records with ``"episode"`` and ``"step"`` fields (the Atari expert sets, ``laya.atari_data``) use them. Otherwise
    the id is ``<episode>-<step>`` with a decimal step, e.g. ViZDoom basic's ``"train-000000-001"`` (``modal_app.py``
    ``prepare_doom_basic`` writes ``"%s-%06d-%03d" % (split, episode, t)``): episode ``"train-000000"``, step 1.
    """
    if rec.get("episode") is not None and rec.get("step") is not None:
        return (rec["episode"], int(rec["step"]))
    m = _ID_EPISODE_STEP.match(str(rec.get("id") or ""))
    return (m.group("episode"), int(m.group("step"))) if m else None


def with_next_targets(recs: List[Dict]) -> List[Dict]:
    """The records, each copied with ``"next_target"`` = the training target of the next step of the same recorded
    episode (the frame at ``(episode, step + 1)``, see ``episode_step``): its ``"target"`` (e.g. the expert's action
    distribution) normalised, or the one-hot of its ``"label"``, exactly the target ``jsonl_example`` builds for it.

    A record keeps no ``next_target`` (and is returned unchanged) when its successor was not recorded (an episode's
    last frame, or a step the recording skipped), when its successor's options differ from its own (a different
    question type or option list / order: the target is only meaningful over the same options), when either record
    is not a valid example, or when ``episode_step`` cannot place it. Duplicate ``(episode, step)`` keys raise.
    """
    keys = [episode_step(r) for r in recs]
    at: Dict = {}
    for i, key in enumerate(keys):
        if key is None:
            continue
        if key in at:
            raise ValueError("duplicate game frame %r (records %d and %d)" % (key, at[key], i))
        at[key] = i
    parsed: Dict[int, Optional[Dict]] = {}

    def example(i):
        if i not in parsed:
            parsed[i] = jsonl_example(recs[i], "")
        return parsed[i]

    out = []
    for i, (rec, key) in enumerate(zip(recs, keys)):
        j = at.get((key[0], key[1] + 1)) if key is not None else None
        ex, nxt = (example(i), example(j)) if j is not None else (None, None)
        same = ex is not None and nxt is not None and (
            (ex["q"]["t"], render_options(ex["q"])) == (nxt["q"]["t"], render_options(nxt["q"])))
        if not same:
            out.append(rec)
            continue
        out.append(dict(rec, next_target=list(nxt["target"])))
    return out


def load_jsonl_examples(root: str, name: str, split: str, limit: Optional[int] = None,
                        next_targets: bool = False) -> List[Dict]:
    """Load ``<root>/<name>/<split>.jsonl``; ``limit`` keeps the first records in file order. ``next_targets`` adds
    each recorded game frame's ``next_target`` from the whole file (``with_next_targets``) before ``limit`` applies;
    off, the examples are exactly as before."""
    base = os.path.join(root, name)
    with open(os.path.join(base, split + ".jsonl")) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if next_targets:
        recs = with_next_targets(recs)
    if limit:
        recs = recs[:limit]
    out = [jsonl_example(r, base, name) for r in recs]
    return [ex for ex in out if ex is not None]


# ---------------------------------------------------------------------------------------------------------
# Evaluation and calibration
# ---------------------------------------------------------------------------------------------------------


class _EvalItems(torch.utils.data.Dataset):
    def __init__(self, processor, examples, pairs, transform=None):
        self.processor, self.examples, self.pairs, self.transform = processor, examples, pairs, transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, j):
        i, order = self.pairs[j]
        ex = self.transform(self.examples[i]) if self.transform else self.examples[i]
        it = make_item(self.processor, ex, random.Random(0), shuffle=False, order=order)
        it["index"] = i
        return it


def _collate_eval(items, pad_id):
    b = collate_vlm(items, pad_id)
    b["index"] = [it["index"] for it in items]
    b["order"] = [it["order"] for it in items]
    b["ids_sha256"] = [input_ids_sha256([it["ids"]]) for it in items]
    return b


@torch.no_grad()
def collect_logits(
    model: VLMDecisionModel,
    processor,
    examples: List[Dict],
    batch_size: int = 16,
    num_workers: int = 0,
    device=None,
    orders: Optional[Callable[[int], List[List[int]]]] = None,
    transform: Optional[Callable[[Dict], Dict]] = None,
) -> List[Dict]:
    """Label-order logits per example.

    ``orders(k)`` gives the option orders to score each k-option example under (default: the example's own
    ``"order"`` if it has one, else identity). ``transform(example)`` is applied in the loader just before
    tokenizing, e.g. ``laya.robustness.realize`` to perturb the images without storing them.
    ``"logits"`` is the mean over orders (as in ``VLMAgent.predict(n_permutations=...)``); ``"logits_per_order"``
    keeps each order's logits (label order), aligned with ``orders(k)``, and ``"input_ids_sha256"`` the hash of
    each order's input ids (``laya.vlm.input_ids_sha256``), for row-level evidence files. A model with a value
    head adds ``"value_logit"`` (mean over orders) and, for examples that carry one, ``"value_target"``.
    """
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    amp = device.type == "cuda"
    pairs = []
    for i, ex in enumerate(examples):
        k = len(ex["target"])
        for order in (orders(k) if orders else [ex.get("order") or list(range(k))]):
            pairs.append((i, order))
    loader = torch.utils.data.DataLoader(
        _EvalItems(processor, examples, pairs, transform), batch_size=batch_size, num_workers=num_workers,
        collate_fn=functools.partial(_collate_eval, pad_id=processor.tokenizer.pad_token_id),
        worker_init_fn=_single_thread_worker,
    )
    per_ex: Dict[int, List[torch.Tensor]] = {}
    hashes: Dict[int, List[str]] = {}
    per_value: Dict[int, List[float]] = {}
    per_act: Dict[int, List[float]] = {}
    for batch in loader:
        b = _to(batch, device, model.encoder.dtype)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, act = _forward(model, b)
        logits = logits.float().cpu()
        p_act = torch.softmax(act.float(), -1)[:, 0].cpu()  # the act head's P(act) (vs escalate)
        vlog = model.last_value.float().cpu() if model.last_value is not None else None
        for r, (i, order) in enumerate(zip(batch["index"], batch["order"])):
            k = len(order)
            z = torch.empty(k)
            z[torch.tensor(order)] = logits[r, :k]  # marker j scored option order[j]
            per_ex.setdefault(i, []).append(z)
            per_act.setdefault(i, []).append(float(p_act[r]))
            if vlog is not None:
                per_value.setdefault(i, []).append(float(vlog[r]))
            hashes.setdefault(i, []).append(batch["ids_sha256"][r])
    out = []
    for i, ex in enumerate(examples):
        zs = per_ex[i]
        out.append({"logits": torch.stack(zs).mean(0), "logits_per_order": zs, "input_ids_sha256": hashes[i],
                    "target": torch.tensor(ex["target"]),
                    "qtype": QTYPES[ex["q"]["t"]], "dataset": ex.get("dataset", "_"),
                    "label": ex.get("label", int(np.argmax(ex["target"]))),
                    "act_probability": float(np.mean(per_act[i]))})
        if i in per_value:  # value head: mean logit over orders, and the example's target if it has one
            out[-1]["value_logit"] = float(np.mean(per_value[i]))
            if ex.get("value") is not None:
                out[-1]["value_target"] = float(ex["value"])
    return out


def cyclic_orders(n: int) -> Callable[[int], List[List[int]]]:
    """The first ``n`` cyclic shifts of the options (fewer if an example has fewer options)."""
    return lambda k: [[(i + s) % k for i in range(k)] for s in range(min(n, k))]


def fit_temperatures_from(records: List[Dict]) -> List[float]:
    """Per-type temperature scaling by LBFGS (same as the text training notebook)."""
    temps = []
    for t in range(3):
        sel = [r for r in records if r["qtype"] == t]
        if len(sel) < 10:
            temps.append(1.0)
            continue
        kmax = max(len(r["logits"]) for r in sel)
        Z = torch.full((len(sel), kmax), -1e4)
        T = torch.zeros((len(sel), kmax))
        for i, r in enumerate(sel):
            Z[i, : len(r["logits"])], T[i, : len(r["target"])] = r["logits"], r["target"]
        log_t = torch.zeros(1, requires_grad=True)
        lbfgs = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            lbfgs.zero_grad()
            loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss

        lbfgs.step(closure)
        temps.append(float(torch.clamp(log_t.exp(), 0.1, 10.0)))
    return temps


def fit_temperatures(model: VLMDecisionModel, processor, examples: List[Dict], **kw) -> List[float]:
    return fit_temperatures_from(collect_logits(model, processor, examples, **kw))


def metrics_from(records: List[Dict], temperatures: Sequence[float] = (1.0, 1.0, 1.0),
                 ece_floor_sims: int = 0) -> Dict[str, Dict[str, float]]:
    """Accuracy, ECE (max-prob confidence, 15 bins), and NLL overall and per dataset.

    ``ece_floor_sims > 0`` adds ``ece_floor`` / ``ece_floor_p95`` to every group (``all`` included): the ECE a
    perfectly calibrated model scores on the same confidences and row count (``laya.robustness.floor``, that many
    simulated draws, seeded per group name), computed from exactly the confidences and 15 bins its ``ece`` uses.
    The definition is the same for every question type (max probability; argmax == label, the most-voted answer
    on vote sets), so the floor is the null for the ``ece`` next to it everywhere. Rows are taken as independent;
    sets with several questions per image have a somewhat higher true floor. Off by default (training loops).

    Groups with ``score`` records also get two ordinal metrics over those records: ``mae``, the absolute
    difference between the expected level under the model and under the target (``|E_p[i] - E_t[i]|``, in
    levels; with a one-hot target that is the distance from the label), and ``xent``, the cross-entropy against
    the (possibly soft) target, which is what a vote-histogram target like AVA's is actually trained on. Argmax
    accuracy and NLL against the argmax label understate such a model: it is trained to spread probability.

    ``choice`` and ``noul`` records with a soft target (human vote shares, e.g. CIFAR-10H or VQAv2 yes/no) add
    ``n_soft`` and ``soft_xent``, the same cross-entropy for those records. Next to each cross-entropy is a
    model-free reference, ``prior_xent`` / ``prior_soft_xent``: the cross-entropy of the same records' targets
    against their mean target (the dataset's average vote histogram), where they all have the same option count.
    """
    groups: Dict[str, List] = {"all": []}
    ordinal: Dict[str, List] = {}
    soft: Dict[str, List] = {}
    targets: Dict[str, Dict[str, List]] = {"xent": {}, "soft_xent": {}}
    for r in records:
        p = torch.softmax(r["logits"] / temperatures[r["qtype"]], -1)
        row = (float(p.max()), float(int(p.argmax()) == r["label"]), -float(torch.log(p[r["label"]].clamp_min(1e-12))))
        groups["all"].append(row)
        groups.setdefault(r["dataset"], []).append(row)
        if r["qtype"] == QTYPES["score"] and r.get("target") is not None:
            t = torch.as_tensor(r["target"], dtype=torch.float32)
            t = t / t.sum().clamp_min(1e-12)
            levels = torch.arange(len(p), dtype=torch.float32)
            orow = (abs(float((p * levels).sum() - (t * levels).sum())), -float((t * torch.log(p.clamp_min(1e-12))).sum()))
            ordinal.setdefault("all", []).append(orow)
            ordinal.setdefault(r["dataset"], []).append(orow)
            for g in ("all", r["dataset"]):
                targets["xent"].setdefault(g, []).append(t)
        elif r["qtype"] != QTYPES["score"] and float(r["target"].max()) < 1.0 - 1e-6:
            t = torch.as_tensor(r["target"], dtype=torch.float32)
            t = t / t.sum().clamp_min(1e-12)
            xent = -float((t * torch.log(p.clamp_min(1e-12))).sum())
            for g in ("all", r["dataset"]):
                soft.setdefault(g, []).append(xent)
                targets["soft_xent"].setdefault(g, []).append(t)
    out = {}
    for name, rows in groups.items():
        a = np.array(rows) if rows else np.zeros((0, 3))
        out[name] = {"n": len(rows), "acc": float(a[:, 1].mean()) if rows else float("nan"),
                     "ece": ece_score(a[:, 0], a[:, 1]), "nll": float(a[:, 2].mean()) if rows else float("nan")}
        if ece_floor_sims > 0:
            from .robustness.floor import ece_floor_fields
            out[name].update(ece_floor_fields(a[:, 0], name, n_sim=ece_floor_sims))
        if name in ordinal:
            o = np.array(ordinal[name])
            out[name].update(n_score=len(o), mae=float(o[:, 0].mean()), xent=float(o[:, 1].mean()))
        if name in soft:
            out[name].update(n_soft=len(soft[name]), soft_xent=float(np.mean(soft[name])))
        for key, by_group in targets.items():
            ts = by_group.get(name)
            if ts and key in out[name] and len({len(t) for t in ts}) == 1:
                T = torch.stack(ts)
                out[name]["prior_" + key] = -float((T * torch.log(T.mean(0).clamp_min(1e-12))).sum(-1).mean())
    return out


def evaluate(model: VLMDecisionModel, processor, examples: List[Dict], temperatures=(1.0, 1.0, 1.0), **kw) -> Dict:
    return metrics_from(collect_logits(model, processor, examples, **kw), temperatures)


def format_metrics(m: Dict) -> str:
    def one(k, v):
        s = "%s n=%d acc=%.3f ece=%.3f nll=%.3f" % (k, v["n"], v["acc"], v["ece"], v["nll"])
        for key in ("xent", "soft_xent"):
            if key == "xent" and "mae" in v:
                s += " mae=%.3f" % v["mae"]
            if key in v:
                s += " %s=%.3f" % (key, v[key])
                if "prior_" + key in v:
                    s += " (prior %.3f)" % v["prior_" + key]
        return s
    return " | ".join(one(k, v) for k, v in m.items())


def main(argv: Optional[Iterable[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--init", default=None, help="saved VLM agent dir to continue from")
    ap.add_argument("--synthetic", action="store_true", help="train on coloured-square toy data")
    ap.add_argument("--dataset", action="append", choices=sorted(ADAPTERS), default=[])
    ap.add_argument("--limit", type=int, default=1000, help="examples per dataset")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--freeze", choices=["head", "last_n", "full"], default="head")
    ap.add_argument("--n-last", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    agent = VLMAgent(args.init, backbone=None if args.init else args.backbone, device=args.device)
    examples = synthetic_examples() if args.synthetic else []
    for name in args.dataset:
        examples += load_hf_examples(name, limit=args.limit)
    if not examples:
        ap.error("no training data: pass --synthetic and/or --dataset")
    losses = train(
        agent.model, agent.processor, examples, steps=args.steps, batch_size=args.batch_size,
        freeze=args.freeze, n_last=args.n_last, device=str(agent.device),
    )
    print("final loss %.4f (finite=%s)" % (losses[-1], math.isfinite(losses[-1])))
    if args.out:
        agent.temperature = fit_temperatures(agent.model, agent.processor, examples)
        agent.save(args.out, include_backbone=args.freeze != "head")
        print("saved to", args.out)


if __name__ == "__main__":
    main()
