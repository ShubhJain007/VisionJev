"""Teacher states rendered as page images: layout helpers and the dataset build."""
import json
import random

import pytest

pytest.importorskip("PIL")
from laya import visionjev_teacher_img as TI
from laya.visionjev_synth import _font


def test_json_state_becomes_key_value_lines_and_wrap_keeps_every_word():
    lines = TI.state_lines(json.dumps({"case": "A-1", "items": [{"sku": "x", "qty": 2}], "tags": ["a", "b"], "note": None}))
    assert lines[0] == "case: A-1" and "tags: a, b" in lines and any(l.strip() == "sku: x" for l in lines)
    assert TI.state_lines("plain\ntext") == ["plain", "text"]
    multi = TI.state_lines(json.dumps({"log": "first\nsecond\n\nthird", "n": 1}))
    assert multi == ["log: first", "    second", "    third", "n: 1"] and not any("\n" in l for l in multi)
    font = _font(14)
    text = "    " + " ".join("word%d" % i for i in range(120)) + " " + "x" * 400
    wrapped = TI.wrap([text], font, 300)
    assert all(font.getlength(l) <= 300 for l in wrapped) and all(l.startswith("    ") for l in wrapped)
    assert "".join(l.strip() for l in wrapped).replace(" ", "") == text.replace(" ", "")


def test_pages_fit_the_pixel_budget_and_long_states_are_skipped():
    short = TI.render_pages("Refund policy.\n" + "Clause text here. " * 40, random.Random(0))
    assert len(short) == 1 and short[0].size[0] == TI.PAGE_W and short[0].size[1] <= TI.PAGE_H
    two = TI.render_pages("A sentence of the long policy document. " * 110, random.Random(0))
    assert len(two) == 2 and all(p.size[0] * p.size[1] <= 589824 for p in two)
    assert TI.render_pages("A sentence of the long policy document. " * 400, random.Random(0)) is None


def test_build_keeps_labels_targets_and_rationales(tmp_path):
    src = tmp_path / "teacher_hard"
    src.mkdir()
    rows = [{"id": "t-%d" % i, "state_text": "Doc %d. " % (i // 2) + "fact " * (30 if i < 4 else 3000),
             "question": {"type": "noul", "instructions": "q%d?" % i, "criteria": None}, "label": i % 2,
             "target": [0.2, 0.8], "rationale": "because", "provenance": {"doc_id": "d%d" % (i // 2)}} for i in range(6)]
    (src / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts, stats = TI.build(str(tmp_path), sources=("teacher_hard",), log=lambda *_: None)
    out = [json.loads(l) for l in open(tmp_path / TI.NAME / "train.jsonl")]
    assert counts["train"] == 4 == len(out) and stats["rows_left_text_only"] == 2  # the 15,000-character state stays text
    assert all(r["state_text"] is None and r["target"] == [0.2, 0.8] and r["rationale"] == "because" for r in out)
    assert out[0]["image"] == out[1]["image"] != out[2]["image"]  # one render per state, shared by its questions
    assert all((tmp_path / TI.NAME / r["image"]).exists() for r in out)
