"""Tests for the SmolVLM2-500M backbone (VisionJev default). Downloads HuggingFaceTB/SmolVLM2-500M-Video-Instruct (~1 GB).

It is SmolVLM2-256M's architecture at a larger language model (SmolLM2-360M, 32 layers, d = 960) with the same
vision tower and processor, so the checks here are that it builds through the same code: the causal readout, the
64-token image run, the shared-prefix path, a full (language model trainable, vision frozen) training step, and a
save/load round trip.
"""
import math

import pytest
import torch
from test_vlm import DEVICE, QUESTIONS, check_schema, frame, square

from laya.preprocess import prefix_ids
from laya.vlm import PREFIX_TEXT, SMOLVLM2_500M_BACKBONE, VLMAgent, set_trainable
from laya.vlm_train import synthetic_examples, train


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return VLMAgent(backbone=SMOLVLM2_500M_BACKBONE, device=DEVICE)


def test_builds_as_a_causal_terminator_backbone(agent):
    assert agent.model.readout == "terminator"
    tc = agent.model.encoder.config.text_config
    assert (tc.hidden_size, tc.num_hidden_layers) == (960, 32)
    assert agent.prep.image_seq_len == 64
    check_schema(agent.predict({"image": square((220, 20, 20))}, QUESTIONS), QUESTIONS)


def test_fast_path_ids_match_the_processor(agent):
    proc = agent.processor
    out = proc(text=[PREFIX_TEXT + proc.image_token], images=[[frame()]], do_image_splitting=False, return_tensors="pt",
               add_special_tokens=False)
    assert prefix_ids(proc, PREFIX_TEXT, 1) == out["input_ids"][0].tolist()


@pytest.mark.parametrize("attention", ["causal", "block"])
def test_prefix_cache_matches_full_path(agent, attention):
    state = {"image": square((220, 20, 20)), "caption": "a test card"}
    agent.model.option_attention = attention
    try:
        ref = agent.predict(state, QUESTIONS, n_permutations=2, prefix_cache=False)
        got = agent.predict(state, QUESTIONS, n_permutations=2, prefix_cache=True)
    finally:
        agent.model.option_attention = "causal"
    for qid, a in ref["answers"].items():
        b = got["answers"][qid]
        pa, pb = a.get("probabilities", {"p": a.get("noul")}), b.get("probabilities", {"p": b.get("noul")})
        assert all(abs(pa[k] - pb[k]) <= 2e-4 for k in pa), (attention, qid, a, b)


def test_full_freeze_trains_language_model_not_vision(agent):
    """``freeze="full"``, the mode the published runs use: the language model trains, the vision tower stays put."""
    model = agent.model
    n = set_trainable(model, "full")
    assert n > 300e6  # the whole language model plus the head
    assert not any(p.requires_grad for p in model.encoder.vision_model.parameters())
    params = dict(model.encoder.named_parameters())
    vision = "vision_model.embeddings.patch_embedding.weight"
    text = "text_model.layers.31.mlp.down_proj.weight"
    before = {k: params[k].detach().clone() for k in (vision, text)}
    losses = train(model, agent.processor, synthetic_examples(2), steps=2, batch_size=2, freeze="full", device=DEVICE,
                   log_every=0)
    assert all(math.isfinite(x) for x in losses)
    assert torch.equal(params[vision], before[vision])
    assert not torch.equal(params[text], before[text])


def test_save_load_roundtrip(agent, tmp_path):
    state = {"image": square((20, 20, 220)), "caption": "a test card"}
    before = agent.predict(state, QUESTIONS)
    agent.save(str(tmp_path), include_backbone=True)
    loaded = VLMAgent(str(tmp_path), device=DEVICE)
    assert loaded.predict(state, QUESTIONS)["answers"] == before["answers"]
