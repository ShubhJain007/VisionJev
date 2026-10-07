"""Think-if-unsure (Hypothesis/V6.3.md section 4): greedy generation with the key/value cache equals the plain
token-by-token loop, notes reach the state only when asked, the rationale ends with the end-of-text token, and
``predict(thinking=...)`` re-scores unsure questions with the model's own working. CPU; downloads
HuggingFaceTB/SmolVLM-256M-Instruct like tests/test_vlm.py."""
import random

import pytest
import torch

from laya.vlm import NOTES_KEY, NOTES_TEXT, VLMAgent, collate_vlm, end_of_text_id, top_probability, with_notes
from laya.vlm_train import make_item, synthetic_examples

Q = {"q": {"type": "choice", "instructions": "Which colour is the sky?", "criteria": ["blue", "green", "red"]}}


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    a = VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu")
    a.model.eval()
    return a


def test_with_notes_by_state_kind():
    assert with_notes({"context": "a"}, "n") == {"context": "a", NOTES_KEY: "n"}
    assert with_notes("a", "n") == "a" + NOTES_TEXT + "n"
    assert with_notes("a", "") == "a" and with_notes(["x"], "n").endswith(NOTES_TEXT + "n")


def test_cached_generation_equals_the_plain_loop_on_padded_rows(agent):
    tok, model = agent.processor.tokenizer, agent.model
    rows = [tok("The sky is blue because", add_special_tokens=False)["input_ids"],
            tok("One, two, three, four, five, six, seven, eight,", add_special_tokens=False)["input_ids"]]
    L = max(map(len, rows))
    ids = torch.full((2, L), tok.pad_token_id)
    att = torch.zeros((2, L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)], att[i, :len(r)] = torch.tensor(r), 1
    fast = model.generate_continuation(ids, att, max_new_tokens=6)
    emb = model.encoder.get_input_embeddings().weight
    for r, got in zip(rows, fast):
        cur = list(r)
        for _ in range(6):  # no cache, no padding: the reference
            x = torch.tensor([cur])
            with torch.no_grad():
                h = model._backbone(x, torch.ones_like(x)).last_hidden_state[0, -1]
            cur.append(int((h @ emb.t()).argmax()))
        assert got == cur[len(r):]
    assert model.generate_continuation(ids, att, max_new_tokens=6, stop_ids=(fast[0][2],))[0] == fast[0][:2]


def test_notes_enter_the_state_only_with_notes_prob_and_rationale_ends_with_eot(agent):
    ex = dict(synthetic_examples(1)[0], notes="working: it looks red", rationale="It is red.")
    proc, tok = agent.processor, agent.processor.tokenizer
    plain = make_item(proc, ex, random.Random(0))
    noted = make_item(proc, ex, random.Random(0), notes_prob=1.0)
    assert plain["ids"] == make_item(proc, dict(ex, notes=None), random.Random(0))["ids"]
    assert len(noted["ids"]) > len(plain["ids"]) and "working: it looks red" in tok.decode(noted["ids"])
    assert noted["markers"] != plain["markers"] and noted["target"] == plain["target"]
    with_r = make_item(proc, ex, random.Random(0), rationale_max_tokens=32)
    assert with_r["ids"][-1] == end_of_text_id(tok) and with_r["ids"][:with_r["rationale_start"]] == plain["ids"]
    assert make_item(proc, ex, random.Random(0), rationale_max_tokens=3)["ids"][-1] != end_of_text_id(tok)


def test_predict_thinking_modes(agent):
    base = agent.predict("The sky is blue.", Q)
    assert "thinking" not in base["answers"]["q"] and base["usage"]["output_tokens"] == 0
    never = agent.predict("The sky is blue.", Q, thinking={"mode": "auto", "threshold": 0.0})
    assert never["answers"]["q"] == base["answers"]["q"]  # confident enough: the single pass, untouched
    always = agent.predict("The sky is blue.", Q, thinking={"mode": "always", "max_tokens": 8, "return_notes": True})
    t = always["answers"]["q"]["thinking"]
    assert t["first_pass"] == pytest.approx(top_probability(base["answers"]["q"]), abs=1e-4)
    assert 0 < t["notes_tokens"] <= 8 and isinstance(t["notes"], str) and always["usage"]["output_tokens"] == t["notes_tokens"]
    assert abs(sum(always["answers"]["q"]["probabilities"].values()) - 1.0) < 1e-3
    notes, n = agent.generate_notes("The sky is blue.", Q["q"], 8)
    assert notes == t["notes"] and n == t["notes_tokens"]
    with pytest.raises(ValueError):
        agent.predict("x", Q, thinking={"mode": "sometimes"})


def test_notes_script_writes_notes_and_finetune_attaches_them(agent, tmp_path, monkeypatch):
    import json
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import visionjev_finetune
    import visionjev_notes

    assert visionjev_notes.trim_notes("A is 3. So B is 4. A is 3. More.") == "A is 3. So B is 4."
    assert visionjev_notes.trim_notes("One. Two. Three. Four. Five.", 3) == "One. Two. Three."
    chunks = list(visionjev_notes.batches([{"ids": [0] * n} for n in (5, 50, 6, 48)], budget=100, max_rows=8))
    assert [[len(it["ids"]) for it in c] for c in chunks] == [[5, 6], [48, 50]]

    root = tmp_path / "vqa"
    (root / "toy").mkdir(parents=True)
    rows = [{"id": "toy-train-%d" % i, "state_text": "The box holds %d apples and %d pears." % (i + 2, i + 1),
             "question": {"type": "choice", "instructions": "How many apples?", "criteria": [str(i + 1), str(i + 2), str(i + 3)]},
             "label": 1} for i in range(5)]
    (root / "toy" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ck = str(tmp_path / "ck")
    agent.save(ck)
    monkeypatch.setattr("laya.visionjev_log.log_experiment", lambda *a, **k: None)
    visionjev_notes.main([ck, "--datasets", "toy", "--root", str(root), "--per-dataset", "4", "--max-tokens", "6",
                          "--tag", "t", "--device", "cpu"])
    notes = [json.loads(l) for l in (root / "toy" / "notes" / "t.jsonl").read_text().splitlines()]
    assert 0 < len(notes) <= 4 and all(n["notes"] and 0 < n["tokens"] <= 6 for n in notes)
    from laya.vlm_train import load_jsonl_examples

    exs = load_jsonl_examples(str(root), "toy", "train")
    assert visionjev_finetune.attach_notes(exs, str(root), "t") == len(notes)
    assert sum("notes" in e for e in exs) == len(notes) and visionjev_finetune.attach_notes(exs, str(root), "missing") == 0


def test_thinking_does_not_recurse_when_predict_is_bound_with_thinking(agent):
    import functools

    agent.predict = functools.partial(agent.predict, thinking={"mode": "always", "max_tokens": 6})
    try:
        out = agent.predict("The sky is blue.", Q)  # the second pass must call the class method, not the partial
    finally:
        del agent.predict
    assert out["answers"]["q"]["thinking"]["notes_tokens"] > 0
