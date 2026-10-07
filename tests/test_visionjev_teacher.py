"""Teacher-lite pipeline (laya.visionjev_teacher) without any model: plan invariants, parsing, writer validation, and
assemble's agree / disagree / held-out-domain routing and soft targets, with the endpoint replaced by a stub."""
import json
import random

import pytest

from laya import visionjev_teacher as T


def _by_opt(r):
    """{option text: target mass} for a choice row (options are shuffled at assemble time)."""
    return dict(zip(r["question"]["criteria"], r["target"]))


def _plan(unknown=None, domain="payroll"):
    return {"doc_id": "t00001", "domain": domain, "families": ["routing", "extraction", "ambiguity"],
            "types": ["choice", "noul", "choice"], "state_shape": "text", "unknown_question": unknown, "length_words": 250}


def _written(unknown=None):
    return {"document": "Payroll runs on the 25th. Overtime is paid at 1.5x.",
            "questions": [
                {"family": "routing", "type": "choice", "instructions": "Which team?", "options": ["payroll", "support", "finance"],
                 "answer": "payroll", "justification": "It is a payroll question."},
                {"family": "extraction", "type": "noul", "instructions": "Is overtime paid at 1.5x?", "options": [],
                 "answer": "yes", "justification": "The document says 1.5x."},
                {"family": "ambiguity", "type": "choice", "instructions": "What is the bonus rate?",
                 "options": ["5%", "10%", T.CBD], "answer": T.CBD if unknown == 2 else "5%",
                 "justification": "No bonus rate is given."}]}


def test_plan_is_deterministic_and_unknown_questions_are_choices():
    a, b = T.make_plan(400, seed=3), T.make_plan(400, seed=3)
    assert a == b
    for p in a:
        assert len(set(p["families"])) == 3 and len(p["types"]) == 3
        if p["unknown_question"] is not None:
            assert p["types"][p["unknown_question"]] == "choice"
        for f, t in zip(p["families"], p["types"]):
            assert f != "judge_rubric_score" or t == "score"
    assert 0.2 < sum(p["unknown_question"] is not None for p in a) / len(a) < 0.8


def test_parse_strips_thinking_and_normalises_probabilities():
    txt = '<think>{"probabilities": {"1": 1}}</think> answer: {"probabilities": {"1": 2, "2": 6}}'
    assert T.parse_probs(T.parse_json(txt), 2) == [0.25, 0.75]
    assert T.parse_probs(T.parse_json("no json"), 2) is None
    assert T.parse_probs({"probabilities": {"1": 0, "2": 0}}, 2) is None


def test_gold_index_and_writer_validation():
    w = _written(unknown=2)
    assert [T.gold_index(q) for q in w["questions"]] == [0, 0, 2]
    assert T.valid_written(w, _plan(unknown=2)) is None
    assert T.valid_written(w, _plan(unknown=None)) == "unknown question mismatch"
    assert T.valid_written(_written(), _plan(unknown=2)) == "unknown question mismatch"
    bad = _written()
    bad["questions"][0]["answer"] = "legal"
    assert T.valid_written(bad, _plan()) == "answer not among options"
    score = {"type": "score", "options": ["low", "mid", "high"], "answer": "2"}
    assert T.gold_index(score) == 2 and T.gold_index(dict(score, answer=5)) is None


def test_assemble_routes_rows_and_builds_soft_targets(tmp_path):
    wp = tmp_path / "writer.jsonl"
    recs = [dict(_plan(unknown=2), doc_id="d1", written=_written(unknown=2), model="writer-m", ok=True),
            dict(_plan(unknown=2, domain="hospitality"), doc_id="d2", written=_written(unknown=2), model="writer-m", ok=True),
            dict(_plan(), doc_id="d3", ok=False, reject="bad json")]
    wp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    ans = {"a": {"d1-q0": [0.9, 0.05, 0.05], "d1-q1": [0.8, 0.2], "d1-q2": [0.1, 0.1, 0.8],
                 "d2-q0": [0.9, 0.05, 0.05], "d2-q1": [0.8, 0.2], "d2-q2": [0.1, 0.1, 0.8]},
           "b": {"d1-q0": [0.7, 0.2, 0.1], "d1-q1": [0.6, 0.4], "d1-q2": [0.6, 0.1, 0.3],  # b disagrees on q2
                 "d2-q0": [0.7, 0.2, 0.1], "d2-q1": [0.6, 0.4], "d2-q2": [0.2, 0.1, 0.7]}}
    paths = []
    for name, qs in ans.items():
        p = tmp_path / ("answers-%s.jsonl" % name)
        p.write_text("".join(json.dumps({"qid": q, "answerer": name, "model": name + "-m", "probs": v}) + "\n"
                             for q, v in qs.items()))
        paths.append(str(p))
    stats = T.assemble(str(wp), paths, "", str(tmp_path / "out"))
    assert stats["rejected_writer"] == 1

    def rows(name):
        out = []
        for split in ("train", "val"):
            f = tmp_path / "out" / name / (split + ".jsonl")
            if f.exists():
                out += [json.loads(l) for l in f.read_text().splitlines()]
        return out

    hard, contested, calib = rows("teacher_hard"), rows("teacher_hard_eval"), rows("teacher_calib")
    assert len(hard) == 2 and len(contested) == 1 and len(calib) == 3
    assert all(r["provenance"]["domain"] == "hospitality" for r in calib)
    choice = next(r for r in hard if r["question"]["type"] == "choice")
    assert _by_opt(choice)["payroll"] == 0.8 and _by_opt(choice)["support"] == 0.125 and choice["rationale"]
    assert choice["question"]["criteria"][choice["label"]] == "payroll"
    noul = next(r for r in hard if r["question"]["type"] == "noul")
    assert noul["label"] == 1 and noul["target"] == [0.3, 0.7]  # [P(no), P(yes)], label 1 = yes
    assert "target" not in contested[0] and contested[0]["question"]["criteria"][contested[0]["label"]] == T.CBD
    man = json.loads((tmp_path / "out" / "teacher_hard_eval" / "manifest.json").read_text())
    assert man["eval_only"] and man["license"]["commercial"]


def test_write_and_answer_resume_with_a_stub_endpoint(tmp_path, monkeypatch):
    plan = tmp_path / "plan.jsonl"
    plan.write_text(json.dumps(_plan(unknown=2)) + "\n")
    calls = []

    def fake_chat(url, model, messages, max_tokens, temperature, timeout=600, extra=None):
        calls.append(model)
        if model == "writer":
            return "<think>hmm</think>" + json.dumps(_written(unknown=2))
        return json.dumps({"probabilities": {"1": 0.7, "2": 0.2, "3": 0.1}})

    monkeypatch.setattr(T, "chat", fake_chat)
    out = tmp_path / "writer.jsonl"
    T.run_write(str(plan), "http://x", "writer", str(out), workers=2)
    T.run_write(str(plan), "http://x", "writer", str(out), workers=2)  # resumes: nothing left to do
    assert calls.count("writer") == 1 and json.loads(out.read_text())["ok"]
    ans = tmp_path / "a.jsonl"
    T.run_answer(str(out), "http://x", "ans", "a", str(ans), workers=2)
    got = [json.loads(l) for l in ans.read_text().splitlines()]
    assert len(got) == 3 and all(g["probs"] for g in got)
    assert got[[g["qid"] for g in got].index("t00001-q1")]["probs"] == pytest.approx([0.7 / 0.9, 0.2 / 0.9])


def _lp(tok, alts):
    import math
    return {"token": tok, "logprob": 0.0, "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in alts]}


def test_probs_from_logprobs_uses_the_last_option_token_and_merges_variants():
    toks = [_lp("The", [("The", 1.0)]), _lp("2", [("2", 0.9), ("1", 0.1)]), _lp(" answer", [(" answer", 1.0)]),
            _lp("3", [("3", 0.6), (" 3", 0.1), ("1", 0.2), ("9", 0.05), ("x", 0.05)])]
    got = T.probs_from_logprobs(toks, 3)
    assert got["probs"] == pytest.approx([0.2 / 0.9, 0.0, 0.7 / 0.9], abs=1e-5)
    assert got["other_mass"] == pytest.approx(0.1, abs=1e-5)
    assert T.probs_from_logprobs([_lp("yes", [("yes", 1.0)])], 2) is None
    assert T.ensemble([[1.0, 0.0], [0.5, 0.5]]) == [0.75, 0.25]


def test_logprobs_mode_ensembles_samples(tmp_path, monkeypatch):
    wp = tmp_path / "writer.jsonl"
    wp.write_text(json.dumps(dict(_plan(unknown=2), written=_written(unknown=2), model="w", ok=True)) + "\n")
    outs = [[_lp("1", [("1", 0.9), ("2", 0.1)])], [_lp("2", [("2", 0.8), ("1", 0.2)])], [_lp("1", [("1", 1.0)])]]
    monkeypatch.setattr(T, "chat_logprobs", lambda *a, **k: outs)
    out = tmp_path / "lp.jsonl"
    T.run_answer(str(wp), "http://x", "m", "a", str(out), workers=1, mode="logprobs", samples=3, temperature=1.0)
    rec = [json.loads(l) for l in out.read_text().splitlines()][0]
    assert rec["samples"] == 3 and rec["votes"] == [0, 1, 0]
    assert rec["probs"][:2] == pytest.approx([(0.9 + 0.2 + 1.0) / 3, (0.1 + 0.8) / 3], abs=1e-5)


def test_soft_target_keeps_gold_argmax_and_carries_distractor_mass():
    t = T.soft_target(0, [[0.1, 0.7, 0.2], [0.3, 0.5, 0.2]], 0.5)
    assert t == pytest.approx([0.6, 0.3, 0.1]) and max(range(3), key=t.__getitem__) == 0
    assert T.soft_target(1, [[0.0, 1.0]], 0.5) == [0.0, 1.0]


def test_assemble_uses_direct_answers_for_soft_targets(tmp_path):
    wp = tmp_path / "writer.jsonl"
    wp.write_text(json.dumps(dict(_plan(unknown=2), doc_id="d1", written=_written(unknown=2), model="w", ok=True)) + "\n")
    think = tmp_path / "a.jsonl"
    think.write_text("".join(json.dumps({"qid": q, "answerer": "a", "model": "m", "probs": p}) + "\n" for q, p in
                             {"d1-q0": [1, 0, 0], "d1-q1": [1, 0], "d1-q2": [0, 0, 1]}.items()))
    direct = tmp_path / "d.jsonl"
    direct.write_text("".join(json.dumps({"qid": q, "answerer": "d", "model": "m", "probs": p}) + "\n" for q, p in
                              {"d1-q0": [0.2, 0.8, 0.0], "d1-q1": [0.6, 0.4], "d1-q2": [0.5, 0.0, 0.5]}.items()))
    stats = T.assemble(str(wp), [str(think)], "", str(tmp_path / "out"), soft_paths=[str(direct)])
    assert stats["soft_direct"] == 3
    rows = []
    for sp in ("train", "val"):
        f = tmp_path / "out" / "teacher_hard" / (sp + ".jsonl")
        if f.exists():
            rows += [json.loads(l) for l in f.read_text().splitlines()]
    by = {r["question"]["instructions"]: r for r in rows}
    assert _by_opt(by["Which team?"])["payroll"] == pytest.approx(0.6) and _by_opt(by["Which team?"])["support"] == pytest.approx(0.4)
    assert by["Is overtime paid at 1.5x?"]["target"] == pytest.approx([0.2, 0.8]) and by["Is overtime paid at 1.5x?"]["label"] == 1
    assert _by_opt(by["What is the bonus rate?"])[T.CBD] == pytest.approx(0.75)


def test_parse_reasoned_renormalises_without_unknown():
    got = T.parse_reasoned({"probabilities": {"1": 0.6, "2": 0.2, "unknown": 0.2}, "rationale": " Clause 4 wins. "}, 2)
    assert got["probs"] == pytest.approx([0.75, 0.25]) and got["unknown_mass"] == pytest.approx(0.2)
    assert got["rationale"] == "Clause 4 wins."
    assert T.parse_reasoned({"probabilities": {"unknown": 1.0}}, 2) is None
    msgs = T.reasoned_messages("doc", {"type": "noul", "instructions": "Q?", "options": []})
    assert msgs[0]["role"] == "system" and "1. yes" in msgs[1]["content"] and "unknown" in msgs[1]["content"]


def test_combine_target_mixes_and_keeps_gold_on_top():
    assert T.combine_target(0, [[0.9, 0.1]], [[0.5, 0.5]]) == pytest.approx([0.7, 0.3])
    t = T.combine_target(0, [[0.55, 0.45]], [[0.0, 1.0]])  # confident one-pass slip: blended with the gold
    assert max(range(2), key=t.__getitem__) == 0 and t == pytest.approx([0.6375, 0.3625])
    assert T.combine_target(1, [], [[0.4, 0.6]]) == pytest.approx([0.2, 0.8])


def test_assemble_with_reasoned_rationale_and_disagreement(tmp_path):
    wp = tmp_path / "writer.jsonl"
    wp.write_text(json.dumps(dict(_plan(unknown=2), doc_id="d1", written=_written(unknown=2), model="w", ok=True)) + "\n")
    rows = lambda d, extra=None: "".join(json.dumps(dict({"qid": q, "answerer": "x", "model": "m", "probs": p}, **(extra or {}))) + "\n"
                                         for q, p in d.items())  # noqa: E731
    think, reas, direct = tmp_path / "a.jsonl", tmp_path / "r.jsonl", tmp_path / "d.jsonl"
    think.write_text(rows({"d1-q0": [1, 0, 0], "d1-q1": [1, 0], "d1-q2": [0, 0, 1]}))
    reas.write_text(rows({"d1-q0": [0.8, 0.2, 0.0], "d1-q1": [0.3, 0.7], "d1-q2": [0.0, 0.1, 0.9]},
                         {"rationale": "Because the ledger says so.", "unknown_mass": 0.05}))
    direct.write_text(rows({"d1-q0": [0.4, 0.6, 0.0], "d1-q1": [0.5, 0.5], "d1-q2": [0.2, 0.0, 0.8]}))
    stats = T.assemble(str(wp), [str(think)], "", str(tmp_path / "out"), soft_paths=[str(direct)], reasoned_paths=[str(reas)])
    got = {}
    for name in ("teacher_hard", "teacher_hard_eval"):
        for sp in ("train", "val"):
            f = tmp_path / "out" / name / (sp + ".jsonl")
            if f.exists():
                for l in f.read_text().splitlines():
                    r = json.loads(l)
                    got[r["question"]["instructions"]] = (name, r)
    name, r = got["Which team?"]
    assert name == "teacher_hard" and _by_opt(r)["payroll"] == pytest.approx(0.6) and r["rationale"] == "Because the ledger says so."
    assert got["Is overtime paid at 1.5x?"][0] == "teacher_hard_eval"  # the reasoned teacher disagrees with the gold
    assert _by_opt(got["What is the bonus rate?"][1])[T.CBD] == pytest.approx(0.85)
    assert stats["soft_reasoned+direct"] == 3


def test_finish_choice_shuffles_filters_length_and_adds_cbd():
    q = {"type": "choice", "instructions": "Q", "criteria": ["a", "bb", "ccc"]}
    assert T.finish_choice(q, 2, [0.1, 0.1, 0.8], random.Random(0)) is None  # gold longest by > 15%
    seen_orders, seen_cbd = set(), 0
    for seed in range(40):
        got = T.finish_choice({"type": "choice", "instructions": "Q", "criteria": ["aaa", "bbb", "ccc"]}, 1, [0.1, 0.8, 0.1],
                              random.Random(seed))
        q2, lab, tgt, order = got
        assert q2["criteria"][lab] == "bbb" and tgt[lab] == 0.8 and abs(sum(tgt) - 1) < 1e-9
        seen_orders.add(tuple(order)); seen_cbd += T.CBD in q2["criteria"]
    assert len(seen_orders) > 3 and 0 < seen_cbd < 40
    sc = T.finish_choice({"type": "score", "instructions": "Q", "criteria": ["low", "mid", "top"]}, 2, None, random.Random(1))
    assert sc[0]["criteria"] == ["low", "mid", "top"] and sc[1] == 2  # score levels keep their order, no CBD
    cbd = T.finish_choice({"type": "choice", "instructions": "Q", "criteria": ["5%", "10%", T.CBD]}, 2, None, random.Random(1))
    assert cbd is not None and cbd[0]["criteria"][cbd[1]] == T.CBD  # a CBD gold is never dropped for its length


def test_round2_plan_and_writer_prompt():
    plans = T.make_plan2(400, seed=1)
    assert all(p["doc_id"].startswith("u") and len(p["families"]) == 5 == len(set(p["families"])) for p in plans)
    assert not {p["domain"] for p in plans} & T.CALIB_DOMAINS
    assert {p["state_kind"] for p in plans} == set(T.STATE_KINDS)
    fams = [f for p in plans for f in p["families"]]
    assert fams.count("multi_hop_chain") > fams.count("routing")  # weighted to what the model misses
    for p in plans:
        u = p["unknown_question"]
        assert u is None or (p["types"][u] == "choice" and not p["families"][u].startswith("judge_"))
    p = plans[0]
    text = T.writer_messages(p)[0]["content"]
    assert "exactly 5 hard questions" in text and T.STATE_KINDS[p["state_kind"]][1] in text
    q = {"type": "noul", "instructions": "x?", "options": [], "answer": "yes"}
    assert T.valid_written({"document": "d", "questions": [q] * 3}, dict(p, unknown_question=None)) == "bad json"
    assert T.valid_written({"document": "d", "questions": [q] * 5}, dict(p, unknown_question=None)) is None
    assert "exactly 3 hard questions" in T.writer_messages(T.make_plan(1)[0])[0]["content"]  # round 1 unchanged
