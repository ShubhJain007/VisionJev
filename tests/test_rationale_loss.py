"""Rationale loss (Hypothesis/V6.1.md §4.6): an example's worked rationale is appended AFTER the option block during
training only, so the read-out positions are unchanged and ``rationale_loss`` adds a tied-embedding LM loss on it. CPU is
enough; downloads HuggingFaceTB/SmolVLM-256M-Instruct like tests/test_vlm.py."""
import random

import pytest
import torch

from laya.visionjev_augment import augment, negate
from laya.vlm import VLMAgent, collate_vlm
from laya.vlm_train import _forward, make_item, rationale_loss, synthetic_examples, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device=DEVICE)


def _ex():
    ex = synthetic_examples(1)[0]
    return dict(ex, rationale="The square is red, so the answer is red.")


def test_rationale_is_appended_after_the_readout_and_changes_nothing_before_it(agent):
    ex = _ex()
    plain = make_item(agent.processor, ex, random.Random(0))
    with_r = make_item(agent.processor, ex, random.Random(0), rationale_max_tokens=32)
    assert with_r["rationale_start"] == len(plain["ids"])
    assert with_r["ids"][: len(plain["ids"])] == plain["ids"]
    assert with_r["markers"] == plain["markers"] and with_r["target"] == plain["target"]
    assert 0 < len(with_r["ids"]) - len(plain["ids"]) <= 32
    capped = make_item(agent.processor, ex, random.Random(0), rationale_max_tokens=3)
    assert len(capped["ids"]) - len(plain["ids"]) == 3
    assert "rationale_start" not in make_item(agent.processor, synthetic_examples(1)[0], random.Random(0),
                                              rationale_max_tokens=32)


def test_collate_marks_exactly_the_rationale_tokens(agent):
    items = [make_item(agent.processor, _ex(), random.Random(0), rationale_max_tokens=16),
             make_item(agent.processor, synthetic_examples(2)[1], random.Random(1))]
    b = collate_vlm(items, agent.processor.tokenizer.pad_token_id)
    m = b["rationale_mask"]
    assert int(m[0].sum()) == len(items[0]["ids"]) - items[0]["rationale_start"]
    assert not bool(m[1].any())
    assert not bool(m[0, : items[0]["rationale_start"]].any())


def test_readout_logits_are_identical_with_and_without_the_rationale(agent):
    ex = _ex()
    items = [make_item(agent.processor, ex, random.Random(0)),
             make_item(agent.processor, ex, random.Random(0), rationale_max_tokens=24)]
    model = agent.model.eval()
    outs = []
    with torch.no_grad():
        for it in items:
            b = collate_vlm([it], agent.processor.tokenizer.pad_token_id)
            b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
            logits, _ = _forward(model, b)  # the training forward: it hides the rationale from the head
            outs.append(logits)
    assert torch.allclose(outs[0], outs[1], atol=1e-4)


def test_rationale_loss_matches_manual_cross_entropy():
    torch.manual_seed(0)
    V, d = 50, 8
    emb = torch.randn(V, d)
    h = torch.randn(2, 6, d, requires_grad=True)
    ids = torch.randint(0, V, (2, 6))
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 3:] = True
    got = rationale_loss(h, ids, mask, emb, w=0.5, chunk=2)
    want = 0.5 * torch.nn.functional.cross_entropy(h[0, 2:5] @ emb.t(), ids[0, 3:6])
    assert torch.allclose(got, want, atol=1e-5)
    got.backward()
    assert h.grad[0, 2:5].abs().sum() > 0 and h.grad[1].abs().sum() == 0
    assert rationale_loss(h, ids, torch.zeros_like(mask), emb) == 0.0 and rationale_loss(None, ids, mask, emb) == 0.0


def test_answer_changing_augmentations_drop_the_rationale():
    ex = {"state": {"context": "The door is open."}, "q": {"t": "noul", "ins": "Is the door open?", "crit": None},
          "target": [0.0, 1.0], "label": 1, "dataset": "toy", "rationale": "It says open."}
    assert negate(ex, random.Random(0)).get("rationale")  # the op itself does not know about rationales
    probs = {k: 0.0 for k in ("state_format", "noul_to_choice", "choice_to_noul", "none_of_above", "option_subset",
                              "inject", "image_ops", "describe_labels")}
    out = augment(ex, random.Random(0), probs=dict(probs, negate=1.0))
    assert "negate" in out["augments"] and "rationale" not in out
    kept = augment(ex, random.Random(0), probs=dict(probs, negate=0.0, inject=1.0))
    assert kept.get("rationale") == "It says open."


def test_training_step_runs_with_the_rationale_loss(agent):
    exs = [dict(e, rationale="Look at the square's colour.") for e in synthetic_examples(6)]
    losses = train(agent.model, agent.processor, exs, steps=2, batch_size=2, freeze="head", device=DEVICE,
                   w_rationale=0.3, rationale_max_tokens=16)
    assert len(losses) == 2 and all(l == l for l in losses)
