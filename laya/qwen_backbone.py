"""Qwen3-VL as a ``VLMDecisionModel`` backbone (causal ``"terminator"`` readout), e.g. ``Qwen/Qwen3-VL-2B-Instruct``.

Qwen3-VL differs from the Idefics3 family (SmolVLM, ModernVBERT) in four ways this module absorbs:

* images are native-resolution patch sequences, not fixed 512 px tiles: the processor returns ``pixel_values``
  ``[n_patches, 3 * 2 * 16 * 16]`` for all images of a sample plus ``image_grid_thw`` ``[n_images, 3]``, and each
  image costs ``t * h * w / 4`` tokens (32 px per token after the 2 x 2 merge);
* positions are 3-D (MRoPE: text / height / width), computed from the token ids and the grids (``get_rope_index``);
* DeepStack: features from three vision-tower layers are added to the hidden states of the first three decoder
  layers at the image positions;
* the prompt uses Qwen's chat template (``QWEN_PREFIX_TEXT`` / ``QWEN_QUESTION_TEXT``).

``QwenVLBackbone`` wraps ``Qwen3VLModel`` behind the attributes the rest of ``laya.vlm`` uses (``config``,
``vision_model``, ``text_model`` with ``layers`` and ``norm``, ``get_input_embeddings``, ``dtype``, ``device``) and a
``forward`` that takes either pixels or precomputed image features. Image features (``encode``) are a pair
``(embeds [n_tokens, d], deepstack [3 x [n_tokens, d]])`` for all images in order, so ``predict`` encodes an image
once and repeats the pair per question row (``repeat_features``).

Not supported yet: ``option_attention="block"`` and the shared-prefix cache (both assume 1-D positions).
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

QWEN_MODEL_TYPES = ("qwen3_vl", "qwen3_vl_moe")  # dense 2B/4B/8B/32B and MoE 30B-A3B share the interface
QWEN_PREFIX_TEXT = "<|im_start|>user\n"
QWEN_IMAGE_TEXT = "<|vision_start|><|image_pad|><|vision_end|>"
QWEN_QUESTION_TEXT = "\n%s question: %s<|im_end|>\n<|im_start|>assistant\nOptions:\n"
QWEN_EOT = "<|im_end|>"
#: default pixel budget per image: 1024 x 1024 px -> up to 1024 image tokens (32 px per token)
QWEN_MAX_PIXELS = 1024 * 1024
QWEN_MIN_PIXELS = 256 * 256

Features = Tuple[torch.Tensor, List[torch.Tensor]]


def is_qwen(config) -> bool:
    return getattr(config, "model_type", None) in QWEN_MODEL_TYPES


def configure_processor(processor, max_pixels: int = QWEN_MAX_PIXELS, min_pixels: int = QWEN_MIN_PIXELS) -> None:
    """Bound the image-token budget: the processor resizes each image to at most ``max_pixels`` (aspect kept)."""
    processor.image_processor.size = {"longest_edge": int(max_pixels), "shortest_edge": int(min_pixels)}
    processor.laya_family = "qwen"
    processor.laya_max_pixels = int(max_pixels)


def max_image_tokens(max_pixels: int = QWEN_MAX_PIXELS) -> int:
    return int(max_pixels) // (32 * 32) + 2  # plus vision_start / vision_end


class QwenVLBackbone(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner  # transformers Qwen3VLModel (no LM head)
        self.image_token_id = inner.config.image_token_id

    # -- the attributes laya.vlm reads -----------------------------------------------------------------------
    @property
    def config(self):
        return self.inner.config

    @property
    def vision_model(self) -> nn.Module:
        return self.inner.visual

    @property
    def text_model(self) -> nn.Module:
        return self.inner.language_model

    @property
    def dtype(self) -> torch.dtype:
        return next(self.inner.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.inner.parameters()).device

    def get_input_embeddings(self) -> nn.Module:
        return self.inner.get_input_embeddings()

    def gradient_checkpointing_enable(self, **kw) -> None:
        self.inner.gradient_checkpointing_enable(**kw)

    def gradient_checkpointing_disable(self) -> None:
        self.inner.gradient_checkpointing_disable()

    # -- images ------------------------------------------------------------------------------------------------
    def encode(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> Features:
        """Patches of all images -> (merged image tokens [n_tokens, d], DeepStack features [3 x [n_tokens, d]])."""
        tm = self.__dict__.get("token_merging")  # experimental (laya.qwen_tokens): one image, merged inside the tower
        if tm is not None and image_grid_thw.shape[0] == 1 and not self.training:
            return tm.encode(pixel_values, image_grid_thw)
        out = self.inner.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        return torch.cat(out.pooler_output, 0), list(out.deepstack_features)

    @staticmethod
    def repeat_features(feats: Features, n: int) -> Features:
        """The features of one sample's images, for ``n`` rows that each carry those images in the same order."""
        embeds, deep = feats
        return embeds.repeat(n, 1), [d.repeat(n, 1) for d in deep]

    # -- forward -----------------------------------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values=None,
                image_grid_thw: Optional[torch.Tensor] = None, image_feats: Optional[Features] = None,
                use_cache: bool = False, **_):
        """Hidden states for right-padded rows. Image features (given, or encoded from ``pixel_values``) fill the
        image-pad positions in row order; ``image_grid_thw`` lists every image of every row, in the same order."""
        if attention_mask.dim() != 2:
            raise ValueError("Qwen3-VL backbone: option_attention='block' (4-D masks) is not supported yet")
        emb = self.get_input_embeddings()(input_ids)
        img_mask = input_ids == self.image_token_id
        visual_pos, deep = None, None
        if img_mask.any():
            if image_feats is None:
                if pixel_values is None:
                    raise ValueError("image tokens in the input but neither pixel_values nor image_feats given")
                image_feats = self.encode(pixel_values, image_grid_thw)
            embeds, deep = image_feats
            if embeds.shape[0] != int(img_mask.sum()):
                raise ValueError("%d image features for %d image tokens" % (embeds.shape[0], int(img_mask.sum())))
            emb = emb.masked_scatter(img_mask[..., None], embeds.to(emb.dtype))
            visual_pos = img_mask
            deep = [d.to(emb.dtype) for d in deep]
        position_ids, _ = self.inner.get_rope_index(input_ids, mm_token_type_ids=img_mask.int(),
                                                    image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        self.last_position_ids = position_ids  # [3, B, L]: generation continues each row from its last position
        graphs = self.__dict__.get("graphs")  # serving fast path (laya.qwen_fast): batch 1, no cache, recorded lengths
        if graphs is not None and not use_cache and emb.shape[0] == 1 and graphs.fits(emb.shape[1]) and not self.training:
            from transformers.modeling_outputs import BaseModelOutputWithPast

            return BaseModelOutputWithPast(last_hidden_state=graphs.run(emb, position_ids, visual_pos, deep))
        return self.inner.language_model(input_ids=None, inputs_embeds=emb, attention_mask=attention_mask,
                                         position_ids=position_ids, visual_pos_masks=visual_pos,
                                         deepstack_visual_embeds=deep, use_cache=use_cache)


def qwen_prefix(processor, images: Sequence) -> dict:
    """``vlm_prefix`` for Qwen: ``<|im_start|>user\\n`` + one expanded image run per image, and its patches."""
    text = QWEN_PREFIX_TEXT + QWEN_IMAGE_TEXT * len(images)
    if not images:
        ids = processor.tokenizer(QWEN_PREFIX_TEXT, add_special_tokens=False)["input_ids"]
        return {"ids": list(ids), "pixel_values": None, "pixel_attention_mask": None, "image_grid_thw": None,
                "raw_images": None, "n_images": 0}
    out = processor(text=[text], images=list(images), return_tensors="pt")
    return {"ids": out["input_ids"][0].tolist(), "pixel_values": out["pixel_values"], "pixel_attention_mask": None,
            "image_grid_thw": out["image_grid_thw"], "raw_images": None, "n_images": len(images)}
