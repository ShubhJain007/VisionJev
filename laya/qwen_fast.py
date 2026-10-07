"""Serving fast path for Qwen3-VL checkpoints: the same decision function with less work per request.

Two changes, both made once at load (``VLMAgent.enable_fast`` / ``VLMAgent(..., fast=True)``), following the design of
Imajev's ``--fast --merge-lora`` server (results/serving/fast-path-2026-09-27 in that repo):

* ``merge_lora``: every LoRA adapter is folded into its base weight (float32 sum, rounded once to the weight's dtype) and
  the wrapper is removed, so each projection is one matmul instead of three.
* ``DecoderGraphs``: at batch 1 the language model is bound by kernel launches, not arithmetic, so its forward is recorded
  as CUDA graphs at a ladder of padded lengths and replayed. A prompt is right-padded to the next length (causal
  attention: the padding cannot change earlier positions) and the hidden states of its own positions are returned.
  DeepStack image features, which the eager model adds with a boolean-indexed write a graph cannot hold, are scattered into
  dense buffers before the replay and added inside it.

The vision tower is left eager: a graph of it (one per image grid) was tried on 2026-10-02 and removed. Its 24 blocks are
bound by arithmetic (about 1.3 ms each for ~2,000 patches on an RTX 4070 Ti SUPER), so there was no launch overhead to save.

Inference only: a merged model has no adapters left to save or train. Longer prompts, batches above 1 and cached
generation fall back to the eager model.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

#: padded lengths recorded by default: fine steps where launch overhead dominates, coarser where arithmetic does
DEFAULT_LENGTHS = tuple(range(64, 513, 64)) + tuple(range(640, 1025, 128)) + tuple(range(1280, 2049, 256))
N_DEEPSTACK = 3


def merge_lora(module: nn.Module) -> int:
    """Fold peft LoRA layers under ``module`` into their base ``nn.Linear`` weights in place; returns how many."""
    from peft.tuners.lora.layer import LoraLayer

    merged = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if not isinstance(child, LoraLayer):
                continue
            base = child.get_base_layer()
            if not isinstance(base, nn.Linear):
                raise ValueError("merge_lora: %s wraps %s, only nn.Linear is supported" % (name, type(base).__name__))
            w = base.weight.data.float()
            for adapter in child.lora_A:
                if child.r[adapter] <= 0:
                    continue
                a, b = child.lora_A[adapter].weight.data.float(), child.lora_B[adapter].weight.data.float()
                w += (b @ a) * child.scaling[adapter]
            base.weight.data = w.to(base.weight.dtype)
            setattr(parent, name, base)
            merged += 1
    return merged


class DecoderGraphs:
    """CUDA graphs of ``lm`` (a transformers ``Qwen3VLTextModel``) at fixed padded lengths, batch 1."""

    def __init__(self, lm: nn.Module, lengths: Sequence[int], device: torch.device, dtype: torch.dtype,
                 position_dims: int = 3):
        from transformers.masking_utils import create_causal_mask

        self.lm, self.graphs = lm, {}
        hidden = lm.config.hidden_size
        pool = None
        with torch.no_grad():  # not inference_mode: the static buffers are written in place on every replay
            for n in sorted(set(int(x) for x in lengths), reverse=True):  # longest first: the others reuse its pool
                emb = torch.zeros(1, n, hidden, device=device, dtype=dtype)
                pos = torch.arange(n, device=device).view(1, 1, n).expand(position_dims, 1, n).contiguous()
                deep = [torch.zeros(1, n, hidden, device=device, dtype=dtype) for _ in range(N_DEEPSTACK)]
                mask = create_causal_mask(config=lm.config, inputs_embeds=emb, attention_mask=None, past_key_values=None,
                                          position_ids=None)

                def step(emb=emb, pos=pos, deep=deep, mask=mask):
                    text_pos, rope_pos = (pos[0], pos[1:]) if pos.shape[0] == 4 else (None, pos)
                    pe = lm.rotary_emb(emb, rope_pos)
                    h = emb
                    for i, layer in enumerate(lm.layers):
                        h = layer(h, attention_mask=mask, position_ids=text_pos, past_key_values=None,
                                  position_embeddings=pe)
                        if i < N_DEEPSTACK:
                            h = h + deep[i]
                    return lm.norm(h)

                for _ in range(3):  # warm-up outside the capture (lazy initialisation, cudnn / cublas handles)
                    step()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    out = step()
                pool = graph.pool()
                self.graphs[n] = (graph, emb, pos, deep, out)
        self.lengths: List[int] = sorted(self.graphs)

    def fits(self, n: int) -> bool:
        return bool(self.lengths) and n <= self.lengths[-1]

    def run(self, emb: torch.Tensor, position_ids: torch.Tensor, visual_pos: Optional[torch.Tensor] = None,
            deep: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        """Last hidden states ``[1, n, d]`` for one prompt (``emb`` ``[1, n, d]``, ``position_ids`` ``[3 | 4, 1, n]``)."""
        n = emb.shape[1]
        size = next(x for x in self.lengths if x >= n)
        graph, s_emb, s_pos, s_deep, out = self.graphs[size]
        s_emb.zero_()
        s_emb[:, :n].copy_(emb)
        s_pos[:, :, :n].copy_(position_ids)
        if size > n:  # padded positions continue each axis; nothing before them can see them
            s_pos[:, :, n:].copy_(position_ids[:, :, -1:] + torch.arange(1, size - n + 1, device=s_pos.device))
        idx = visual_pos[0].nonzero(as_tuple=True)[0] if visual_pos is not None and deep is not None else None
        for i, d in enumerate(s_deep):
            d.zero_()
            if idx is not None and i < len(deep):
                d[0, idx] = deep[i].to(d.dtype)
        graph.replay()
        return out[:, :n].clone()
