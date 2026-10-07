"""Experimental: fewer image tokens at the same input resolution (Qwen3-VL checkpoints). NOT part of the fast path; off by
default; every variant here changes the model's inputs, so its accuracy has to be measured (scripts/visionjev_tokens.py).

* ``TokenMerging`` (after ToMe, Bolya et al. 2023): inside the vision tower, at a few layers, the most similar tokens are
  averaged together (bipartite matching on the hidden states), so the later blocks run on fewer tokens. Before every
  read-out (the three DeepStack mergers and the final 2 x 2 patch merger) each original patch gets its merged token back,
  so the tower still returns one feature per image token and the language model sees the usual grid. Saves vision time only.
* ``pool_features``: the tower runs unchanged and its output tokens are average-pooled ``k x k`` before the language model
  (``k = 2``: a quarter of the image tokens), so the language model is what gets cheaper. ``VLMAgent.token_pool = k``.
"""
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


class TokenMerging:
    """``encode(pixel_values, grid_thw)`` like ``QwenVLBackbone.encode`` for one image. At each layer of ``layers`` a
    ``fraction`` (at most half) of the current tokens is merged away."""

    def __init__(self, visual, layers: Sequence[int] = (4, 10, 16), fraction: float = 0.3):
        self.visual, self.layers, self.fraction = visual, tuple(layers), float(fraction)

    @staticmethod
    def _merge(h: torch.Tensor, size: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge ``r`` tokens of ``h`` [n, d] into their most similar partner. Returns the new tokens, their sizes, the
        indices (into the old sequence) whose rotary positions they keep, and the old -> new index map."""
        n = h.shape[0]
        a_idx, b_idx = torch.arange(0, n, 2, device=h.device), torch.arange(1, n, 2, device=h.device)
        r = min(r, a_idx.numel())
        metric = F.normalize(h.float(), dim=-1)
        best, partner = (metric[a_idx] @ metric[b_idx].T).max(-1)
        order = best.argsort(descending=True)
        src, keep = order[:r], order[r:].sort().values  # merged away / kept, both as positions within ``a``
        w = size.unsqueeze(-1).to(h.dtype)
        hb, sb = (h[b_idx] * w[b_idx]).float(), size[b_idx].clone()
        hb.index_add_(0, partner[src], (h[a_idx[src]] * w[a_idx[src]]).float())
        sb.index_add_(0, partner[src], size[a_idx[src]])
        new_h = torch.cat([h[a_idx[keep]], (hb / sb.unsqueeze(-1)).to(h.dtype)], 0)
        new_size = torch.cat([size[a_idx[keep]], sb], 0)
        mapping = torch.empty(n, dtype=torch.long, device=h.device)
        mapping[a_idx[keep]] = torch.arange(keep.numel(), device=h.device)
        mapping[a_idx[src]] = keep.numel() + partner[src]
        mapping[b_idx] = keep.numel() + torch.arange(b_idx.numel(), device=h.device)
        return new_h, new_size, torch.cat([a_idx[keep], b_idx], 0), mapping

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        import transformers.models.qwen3_vl.modeling_qwen3_vl as M

        vis = self.visual
        idx, wts = M.get_vision_interpolation_indices_and_weights(
            grid_thw, num_grid_per_side=vis.num_grid_per_side, mode=vis.interpolation_mode,
            align_corners=vis.interpolation_align_corners, spatial_merge_size=vis.config.spatial_merge_size, kwargs={})
        pos_ids = M.get_vision_position_ids(grid_thw, vis.spatial_merge_size, kwargs={})
        h = vis.patch_embed(pixel_values.type(vis.dtype))
        h = h + (vis.pos_embed(idx) * wts[:, :, None]).sum(1).to(h.dtype)
        pe = vis.rotary_pos_emb(h, pos_ids)
        pe = tuple(pe) if isinstance(pe, (tuple, list)) else (pe,)
        h = h.reshape(h.shape[0], -1)
        n0 = h.shape[0]
        assign = torch.arange(n0, device=h.device)  # original patch -> current token
        size = torch.ones(n0, device=h.device)
        deep_at, deep = list(vis.deepstack_visual_indexes), []
        for i, blk in enumerate(vis.blocks):
            if i in self.layers and h.shape[0] > 8:
                h, size, kept, mapping = self._merge(h, size, int(self.fraction * h.shape[0]))
                pe = tuple(p[kept] for p in pe)
                assign = mapping[assign]
            cu = torch.tensor([0, h.shape[0]], dtype=torch.int32)
            h = blk(h, cu_seqlens=cu, max_seqlen=h.shape[0], position_embeddings=pe if len(pe) > 1 else pe[0])
            if i in deep_at:
                deep.append(vis.deepstack_merger_list[deep_at.index(i)](h[assign]))
        return vis.merger(h[assign]), deep


def pool_features(feats, grid_thw: torch.Tensor, k: int, merge: int = 2):
    """Average-pool the image tokens ``k x k`` (ragged edges averaged over what exists). ``feats`` = (tokens [n, d],
    DeepStack list) of ONE image in raster order over ``(h / merge, w / merge)``; returns the pooled features and the
    grid to tell the language model (patch units)."""
    t, hh, ww = (int(x) for x in grid_thw[0].tolist())
    h2, w2 = hh // merge, ww // merge

    def pool(x: torch.Tensor) -> torch.Tensor:
        y = x.float().reshape(t, h2, w2, -1).permute(0, 3, 1, 2)
        y = F.avg_pool2d(y, k, ceil_mode=True, count_include_pad=False)
        return y.permute(0, 2, 3, 1).reshape(-1, x.shape[-1]).to(x.dtype)

    embeds, deep = feats
    nh, nw = -(-h2 // k), -(-w2 // k)
    grid = torch.tensor([[t, nh * merge, nw * merge]], dtype=grid_thw.dtype)
    return (pool(embeds), [pool(d) for d in deep]), grid


def repack_prefix(prefix: dict, image_token_id: int, n_tokens: int, grid: torch.Tensor) -> dict:
    """The prefix of a ONE-image state with its run of image placeholders resized to ``n_tokens`` and its grid replaced."""
    ids: List[int] = list(prefix["ids"])
    first = ids.index(image_token_id)
    last = len(ids) - 1 - ids[::-1].index(image_token_id)
    return dict(prefix, ids=ids[:first] + [image_token_id] * n_tokens + ids[last + 1:], image_grid_thw=grid)
