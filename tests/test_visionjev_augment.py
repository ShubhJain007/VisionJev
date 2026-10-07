import copy
import json
import random

from PIL import Image

from laya import visionjev_augment as A
from laya.common import render_options


def ex_choice(label=1, target=None):
    q = {"t": "choice", "ins": "Which fruit is red?", "crit": {"banana": None, "apple": None, "lime": None, "kiwi": None}}
    return {"state": {"context": json.dumps({"basket": ["apple", "banana"], "count": 2})}, "q": q,
            "target": target or [0.0, 1.0, 0.0, 0.0], "label": label, "dataset": "toy"}


def ex_noul(label=1):
    return {"state": {"context": "The door is open."}, "q": {"t": "noul", "ins": "Is the door open?", "crit": None},
            "target": [0.0, 1.0] if label else [1.0, 0.0], "label": label, "dataset": "toy"}


def gold_text(ex):
    return render_options(ex["q"])[ex["label"]]


def test_negate_flips_label_and_reverses_target():
    for lab in (0, 1):
        e = A.negate(ex_noul(lab), random.Random(0))
        assert e["label"] == 1 - lab and e["target"] == list(reversed(ex_noul(lab)["target"]))
        assert e["q"]["ins"] != "Is the door open?" and "Decide whether the answer" not in e["q"]["ins"]


def test_choice_to_noul_label_is_whether_proposed_option_is_gold():
    for seed in range(40):
        src = ex_choice(target=[0.1, 0.7, 0.1, 0.1])
        e = A.choice_to_noul(src, random.Random(seed))
        proposed = [o for o in src["q"]["crit"] if '"%s"' % o in e["q"]["ins"] or ": %s\n" % o in e["q"]["ins"]
                    or "answer: %s" % o in e["q"]["ins"]]
        assert e["q"]["t"] == "noul" and e["label"] == int(proposed == ["apple"])
        assert abs(sum(e["target"]) - 1) < 1e-9 and e["target"][1] == (0.7 if e["label"] else 0.1)
        assert "Is the answer to this question" not in e["q"]["ins"]


def test_noul_to_choice_keeps_label_order():
    e = A.noul_to_choice(ex_noul(1), random.Random(0))
    assert e["q"]["t"] == "choice" and e["label"] == 1 and e["target"] == [0.0, 1.0]
    assert list(e["q"]["crit"])[1] in ("correct", "true", "holds")


def test_none_of_above_both_ways():
    seen = set()
    for seed in range(30):
        e = A.none_of_above(ex_choice(), random.Random(seed))
        opts = list(e["q"]["crit"])
        assert opts[-1] in A.NOTA and len(e["target"]) == len(opts) and e["target"][e["label"]] == 1.0
        if "apple" in opts:
            assert gold_text(e) == "apple"
            seen.add("kept")
        else:
            assert e["label"] == len(opts) - 1
            seen.add("removed")
    assert seen == {"kept", "removed"}


def test_option_subset_keeps_gold_and_renormalises():
    for seed in range(20):
        e = A.option_subset(ex_choice(target=[0.1, 0.6, 0.2, 0.1]), random.Random(seed))
        assert gold_text(e) == "apple" and 2 <= len(e["q"]["crit"]) < 4 and abs(sum(e["target"]) - 1) < 1e-9


def test_state_format_keeps_content():
    for seed in range(30):
        e = A.state_format(ex_choice(), random.Random(seed))
        ctx = e["state"]["context"]
        assert "apple" in ctx and "banana" in ctx and "2" in ctx and e["label"] == 1


def test_inject_leaves_label_and_facts():
    src = ex_choice()
    e = A.inject(src, random.Random(3))
    obj = json.loads(e["state"]["context"])
    assert obj["basket"] == ["apple", "banana"] and e["label"] == 1 and e["target"] == src["target"]
    assert any(w in json.dumps(obj) for w in ("banana", "lime", "kiwi"))  # the planted answer is a wrong one


def test_augment_does_not_mutate_and_respects_exclusions(tmp_path):
    src = ex_choice()
    frozen = copy.deepcopy(src)
    for seed in range(200):
        A.augment(src, random.Random(seed), probs={k: 1.0 for k in A.AUGMENTS})
    assert src == frozen
    ex = dict(ex_noul(), dataset="prompt_injection")
    for seed in range(50):
        assert "inject" not in A.augment(ex, random.Random(seed), probs={"inject": 1.0}).get("augments", [])
    p = tmp_path / "x.jpg"
    Image.new("RGB", (64, 48), (200, 10, 10)).save(p)
    img_ex = {"state": {"image": str(p)}, "q": {"t": "noul", "ins": "Is it red?", "crit": None},
              "target": [0.0, 1.0], "label": 1, "dataset": "synth_quality"}
    assert "image_ops" not in A.augment(img_ex, random.Random(0), probs={"image_ops": 1.0}).get("augments", [])
    out = A.augment(dict(img_ex, dataset="aokvqa"), random.Random(0), probs={"image_ops": 1.0})
    assert out["augments"] == ["image_ops"] and not isinstance(out["state"]["image"], str)


def test_augmented_examples_render():
    rng = random.Random(0)
    for seed in range(100):
        for src in (ex_choice(), ex_noul(0), ex_noul(1)):
            e = A.augment(src, random.Random(seed), probs={k: 0.5 for k in A.AUGMENTS})
            k = len(render_options(e["q"]))
            assert len(e["target"]) == k and 0 <= e["label"] < k and abs(sum(e["target"]) - 1) < 1e-6
            assert e["target"][e["label"]] == max(e["target"])
    assert rng


def test_describe_labels_keys_options_and_keeps_label():
    ex = {"dataset": "banking77", "state": {"context": "my card never arrived"}, "label": 2, "target": [0.0, 0.0, 1.0, 0.0],
          "q": {"t": "choice", "ins": "Which intent?",
                "crit": {"card arrival": None, "lost or stolen card": None, "card not delivered yet": None,
                         "card arrival": None, "exchange rate": None}}}
    out = A.describe_labels(ex, random.Random(1))
    keys, descs = list(out["q"]["crit"].keys()), list(out["q"]["crit"].values())
    assert len(keys) == len(set(keys)) == 4
    assert all(" " not in k and k == k.lower() for k in keys)
    assert "card not delivered yet" in descs[2]  # order and gold position preserved
    assert out["label"] == 2 and out["target"] == ex["target"]


def test_describe_labels_leaves_described_criteria_alone():
    ex = {"dataset": "x", "state": {}, "label": 0, "target": [1.0, 0.0],
          "q": {"t": "choice", "ins": "?", "crit": {"refund": "Wants money back", "track": "Where is my parcel"}}}
    assert A.describe_labels(ex, random.Random(0)) is ex


def test_cut_state_keeps_head_and_tail():
    from laya.vlm import _cut_state
    st = list(range(100))
    cut = _cut_state(st, 10)
    assert cut == list(range(6)) + list(range(96, 100))
    assert _cut_state(st, 10, truncate_left=True) == list(range(90, 100))
    assert _cut_state(st, 200) == st and _cut_state(st, 0) == []
