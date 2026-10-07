"""Abstain + reason head (Hypothesis/V6.2.md section 3): off by default and then invisible, round-trips through
checkpoints, is added fresh to a checkpoint that lacks it, trains through make_item / collate / train with the option loss
masked on abstain rows, and is exported by ``predict`` as ``unknown`` + ``abstain_reason``. CPU is enough; downloads
HuggingFaceTB/SmolVLM-256M-Instruct like tests/test_vlm.py."""
import random
import warnings

import pytest
import torch

from laya.visionjev_augment import ABSTAIN_AUGMENTS, AUGMENTS, CBD_TEXT, augment, cbd_removed, gold_removed
from laya.vlm import ABSTAIN_REASONS, VLMAgent, collate_vlm
from laya.vlm_train import abstain_loss, make_item, synthetic_examples, train, vlm_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"
Q = {"q": {"type": "choice", "instructions": "Which colour is the sky?", "criteria": ["blue", "green", "red"]}}


def fresh(**kw):
    torch.manual_seed(0)
    return VLMAgent(backbone=BACKBONE, device=DEVICE, **kw)


@pytest.fixture(scope="module")
def plain():
    return fresh()


@pytest.fixture(scope="module")
def abst():
    return fresh(abstain_head=True)


def test_off_by_default_and_invisible(plain):
    assert plain.model.abstain_head is None
    assert not any(k.startswith("abstain_head.") for k in plain.model.state_dict())
    out = plain.predict("The sky is blue.", Q)
    assert "unknown" not in out["answers"]["q"] and "abstain_reason" not in out["answers"]["q"]


def test_predict_exports_unknown_and_reason_without_touching_probabilities(abst):
    out = abst.predict("The sky is blue.", Q)["answers"]["q"]
    assert 0.0 <= out["unknown"] <= 1.0 and out["abstain_reason"] in ABSTAIN_REASONS
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-3


def test_round_trip_and_fresh_head_on_an_old_checkpoint(abst, plain, tmp_path):
    abst.save(str(tmp_path / "a"))
    back = VLMAgent(str(tmp_path / "a"), device=DEVICE)
    for k, v in abst.model.abstain_head.state_dict().items():
        assert torch.equal(v.cpu(), back.model.abstain_head.state_dict()[k].cpu())
    plain.save(str(tmp_path / "p"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        added = VLMAgent(str(tmp_path / "p"), device=DEVICE, abstain_head=True)
    assert added.model.abstain_head is not None and any("abstain head" in str(x.message) for x in w)


def _choice_ex(gold=0, crit=("blue", "green", "red", "yellow")):
    t = [0.0] * len(crit)
    t[gold] = 1.0
    return {"state": "The sky is blue.", "q": {"t": "choice", "ins": "Which colour?", "crit": {c: None for c in crit}},
            "target": t, "label": gold, "dataset": "toy", "rationale": "The text says blue."}


def test_gold_removed_and_cbd_removed_make_exact_abstain_rows():
    ex = gold_removed(_choice_ex(gold=0), random.Random(0))
    assert "blue" not in ex["q"]["crit"] and ex["abstain"] == 1 and ex["abstain_reason"] == "not_listed"
    assert abs(sum(ex["target"]) - 1.0) < 1e-9 and len(ex["target"]) == 3
    cbd = _choice_ex(gold=3, crit=("5 May", "6 May", "7 May", CBD_TEXT))
    out = cbd_removed(cbd, random.Random(0))
    assert CBD_TEXT not in out["q"]["crit"] and out["abstain_reason"] == "insufficient_evidence"
    probs = {k: 0.0 for k in AUGMENTS}
    probs.update(ABSTAIN_AUGMENTS, cbd_removed=1.0, gold_removed=0.0)
    assert augment(_choice_ex(gold=0), random.Random(0), probs).get("abstain") is None  # gold is not the CBD option
    both = augment(cbd, random.Random(0), probs)
    assert both["abstain"] == 1 and "rationale" not in both  # the worked reason argued for the removed option


def test_make_item_and_collate_carry_abstain_targets(abst):
    exs = [dict(_choice_ex(), abstain=1, abstain_reason="not_listed"), synthetic_examples(1)[0]]
    items = [make_item(abst.processor, ex, random.Random(i), abstain_training=True) for i, ex in enumerate(exs)]
    b = collate_vlm(items, abst.processor.tokenizer.pad_token_id)
    assert b["abstain"].tolist() == [1.0, 0.0]
    assert b["abstain_reason"].tolist() == [ABSTAIN_REASONS.index("not_listed"), -1]
    assert "abstain" not in make_item(abst.processor, exs[1], random.Random(0))  # off unless asked for


def test_masked_rows_get_no_option_gradient():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    loss, _ = vlm_loss(logits, target, torch.tensor([0, 0]), mask, row_weight=torch.tensor([1.0, 0.0]))
    loss.backward()
    assert logits.grad[1].abs().sum() == 0 and logits.grad[0].abs().sum() > 0


def test_abstain_loss_matches_manual():
    a_logits = torch.tensor([0.3, -1.2, 2.0])
    r_logits = torch.randn(3, len(ABSTAIN_REASONS))
    a = torch.tensor([1.0, 0.0, 1.0])
    r = torch.tensor([2, -1, 0])
    got = abstain_loss(a_logits, r_logits, a, r, w_abstain=0.5, w_reason=0.2)
    want = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(a_logits, a) + \
        0.2 * torch.nn.functional.cross_entropy(r_logits[[0, 2]], r[[0, 2]])
    assert torch.allclose(got, want, atol=1e-6)
    assert abstain_loss(None, None, a, r) == 0.0


def test_training_step_runs_with_the_abstain_loss(abst):
    exs = synthetic_examples(4) + [dict(_choice_ex(), abstain=1, abstain_reason="not_listed")] * 2
    losses = train(abst.model, abst.processor, exs, steps=2, batch_size=3, freeze="head", device=DEVICE, w_abstain=0.5)
    assert len(losses) == 2 and all(l == l for l in losses)
