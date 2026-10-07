"""Speed options of ``laya.vlm_train.train`` that must not change what is learned: splitting a batch into length groups
(``length_groups`` / ``_collate_train`` / ``GroupRunner``) and spreading the groups over several devices. The merged
outputs have to equal the plain padded forward row for row, and the gradients on a second "device" have to equal the
single-device ones. CPU only (two CPU "devices" stand in for two GPUs); downloads HuggingFaceTB/SmolVLM-256M-Instruct
like tests/test_vlm.py."""
import itertools
import random

import pytest
import torch

from laya.vlm import VLMAgent, collate_vlm, set_trainable
from laya.vlm_train import GroupRunner, _collate_train, _forward, length_groups, make_item, synthetic_examples, train

BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"


def padded(lengths, groups):
    return sum(len(g) * max(lengths[i] for i in g) for g in groups)


def test_length_groups_is_optimal_and_covers_every_row():
    rng = random.Random(0)
    for _ in range(60):
        n, g_max = rng.randint(1, 7), rng.randint(1, 4)
        ls = [rng.choice([40, 90, 300, 900, 4000]) + rng.randint(0, 30) for _ in range(n)]
        groups = length_groups(ls, g_max, overhead=0)
        assert sorted(i for g in groups for i in g) == list(range(n)) and len(groups) <= g_max
        order = sorted(range(n), key=lambda i: ls[i])
        best = min(padded(ls, [order[a:b] for a, b in zip((0,) + cuts, cuts + (n,))])
                   for k in range(min(g_max, n)) for cuts in itertools.combinations(range(1, n), k))
        assert padded(ls, groups) == best
    assert length_groups([5, 900, 7], 1) == [[0, 2, 1]]


def test_overhead_stops_pointless_splits_and_budget_forces_them():
    assert len(length_groups([100, 101, 102, 103], 4, overhead=256)) == 1
    groups = length_groups([4000] * 6, 3, token_budget=9000, overhead=0)
    assert len(groups) == 3 and all(len(g) == 2 for g in groups)
    assert len(length_groups([4000] * 6, 2, token_budget=9000)) == 2  # cannot be met: still every row, two groups


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    a = VLMAgent(backbone=BACKBONE, device="cpu")
    a.model.eval()
    return a


def _items(agent, with_rationale=False):
    exs = synthetic_examples(6)
    fill = ["", " very" * 40, "", " quite" * 120, " so" * 15, ""]
    out = []
    for i, (ex, f) in enumerate(zip(exs, fill)):
        ex = dict(ex, q=dict(ex["q"], ins=ex["q"]["ins"] + f))
        if with_rationale and i % 2 == 0:
            ex["rationale"] = "The square has that colour."
        it = make_item(agent.processor, ex, random.Random(i), rationale_max_tokens=12 if with_rationale else 0)
        it["dataset"] = "toy"
        out.append(it)
    return out


def test_grouped_forward_equals_the_padded_forward_row_for_row(agent):
    items = _items(agent)
    pad = agent.processor.tokenizer.pad_token_id
    with torch.no_grad():
        plain, plain_act = _forward(agent.model, collate_vlm(items, pad))
        batch = _collate_train(items, pad, max_groups=3)
        out = GroupRunner(agent.model)(batch["parts"])
    assert len(batch["parts"]) > 1
    assert torch.allclose(out["logits"], plain, atol=1e-4) and torch.allclose(out["act"], plain_act, atol=1e-4)
    assert torch.equal(batch["marker_mask"], collate_vlm(items, pad)["marker_mask"])
    assert out["tokens"] < collate_vlm(items, pad)["input_ids"].numel()  # less padding was computed


def test_two_devices_give_the_same_outputs_and_gradients(agent):
    items = _items(agent, with_rationale=True)
    pad = agent.processor.tokenizer.pad_token_id
    model = agent.model
    set_trainable(model, "head")
    model.eval()  # no dropout: the two runs must match exactly
    batch = _collate_train(items, pad, max_groups=3)
    params = [p for p in model.parameters() if p.requires_grad]

    def grads(devices):
        runner = GroupRunner(model, devices, w_rationale=0.3)
        out = runner(batch["parts"])
        loss = out["logits"].clamp(min=-50).sum() + out["act"].sum() + out["rationale"]
        for p in params:
            p.grad = None
        loss.backward()
        return out, [p.grad.clone() for p in params]

    one, g1 = grads(None)
    two, g2 = grads(["cpu", "cpu"])
    assert torch.allclose(one["logits"], two["logits"], atol=1e-5)
    assert torch.allclose(torch.as_tensor(one["rationale"]), torch.as_tensor(two["rationale"]), atol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g1, g2)) and any(g.abs().sum() > 0 for g in g1)


def test_training_runs_with_groups_and_devices_and_reports_padding(agent):
    exs = [dict(e, q=dict(e["q"], ins=e["q"]["ins"] + (" really" * 80 if i % 3 == 0 else "")))
           for i, e in enumerate(synthetic_examples(9))]
    plain, grouped = {}, {}
    train(agent.model, agent.processor, exs, steps=2, batch_size=6, freeze="head", device="cpu", stats=plain)
    losses = train(agent.model, agent.processor, exs, steps=2, batch_size=6, freeze="head", device="cpu", stats=grouped,
                   length_groups_n=3, devices=["cpu", "cpu"])
    assert len(losses) == 2 and all(l == l for l in losses)
    assert grouped["padded_tokens_per_step"] < plain["padded_tokens_per_step"]


def test_group_runner_copies_a_model_that_ran_a_step():
    torch.manual_seed(0)
    agent = VLMAgent(backbone=BACKBONE, device="cpu", abstain_head=True)  # heads whose outputs the model keeps
    agent.model.eval()
    items = _items(agent)
    pad = agent.processor.tokenizer.pad_token_id
    set_trainable(agent.model, "head")
    batch = _collate_train(items, pad, max_groups=2)
    out = GroupRunner(agent.model)(batch["parts"])
    (out["logits"].clamp(min=-50).sum() + out["act"].sum()).backward()  # leaves non-leaf outputs on the model
    two = GroupRunner(agent.model, ["cpu", "cpu"], ckpt_above_tokens=10 ** 9)  # deepcopy must still work
    assert len(two.replicas) == 1 and torch.allclose(two(batch["parts"])["logits"], out["logits"].detach(), atol=1e-5)


def test_checkpointed_steps_stay_on_the_primary_device(agent):
    """Checkpointing recomputes layers in the backward pass with the replica's own (stale) weights, so a step that
    needs it must not use a replica: gradients with checkpointing on and two devices equal the single-device ones."""
    items = _items(agent)
    pad = agent.processor.tokenizer.pad_token_id
    model = agent.model
    set_trainable(model, "head")
    model.eval()
    batch = _collate_train(items, pad, max_groups=3)
    params = [p for p in model.parameters() if p.requires_grad]

    def grads(devices, ckpt_above):
        runner = GroupRunner(model, devices, ckpt_above_tokens=ckpt_above)
        out = runner(batch["parts"])
        for p in params:
            p.grad = None
        (out["logits"].clamp(min=-50).sum() + out["act"].sum()).backward()
        return runner, [p.grad.clone() for p in params]

    _, ref = grads(None, None)
    runner, got = grads(["cpu", "cpu"], 0)  # threshold 0: every step is checkpointed
    assert runner._checkpointing() and all(torch.allclose(a, b, atol=1e-5) for a, b in zip(ref, got))
    GroupRunner(model, None, ckpt_above_tokens=10 ** 9)(batch["parts"])  # leave checkpointing off for other tests
