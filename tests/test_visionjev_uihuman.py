"""laya.visionjev_uihuman: box drawing and numbering on a synthetic image, candidate / distractor selection on
hand-made records, the exact "option set first, gold second" samplers, OmniACT script variants, the split-side rule,
and, for the built sets under data/vqa, split disjointness (files, Rico screens, also against rico_screenqa) and that
every image opens with PIL. No network."""
import json
import os
import random
import struct
from collections import Counter, defaultdict

import pytest
from PIL import Image

from laya import visionjev_uihuman as U

ROOT = os.environ.get("VISIONJEV_DATA", os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "vqa")))
BUILT = [n for n in U.BUILDERS if os.path.exists(os.path.join(ROOT, n, "_READY"))]


# ---- TFRecord / protobuf reader -------------------------------------------------------------------------------------

def _varint(n):
    out = b""
    while True:
        b, n = n & 0x7F, n >> 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _field(num, wire, payload):
    key = _varint((num << 3) | wire)
    return key + (_varint(len(payload)) + payload if wire == 2 else payload)


def _example(features):
    """A tf.train.Example serialized by hand: {name: (kind, values)} with kind bytes / float / int."""
    entries = b""
    for name, (kind, vals) in features.items():
        if kind == "bytes":
            lst = b"".join(_field(1, 2, v) for v in vals)
            feat = _field(1, 2, lst)
        elif kind == "float":
            feat = _field(2, 2, _field(1, 2, b"".join(struct.pack("<f", v) for v in vals)))
        else:
            feat = _field(3, 2, _field(1, 2, b"".join(_varint(v) for v in vals)))
        entries += _field(1, 2, _field(1, 2, name.encode()) + _field(2, 2, feat))
    return _field(1, 2, entries)


def test_parse_example_reads_all_three_list_kinds():
    ex = _example({"image/id": ("bytes", [b"123"]), "image/ref_exp/label": ("int", [7]),
                   "image/object/bbox/xmin": ("float", [0.25, 0.5])})
    out = U.parse_example(ex)
    assert out["image/id"] == [b"123"] and out["image/ref_exp/label"] == [7]
    assert out["image/object/bbox/xmin"] == pytest.approx([0.25, 0.5])


def test_read_tfrecord_roundtrip(tmp_path):
    p = tmp_path / "x.tfrecord"
    with open(p, "wb") as f:
        for i in range(3):
            data = _example({"image/id": ("bytes", [str(i).encode()])})
            f.write(struct.pack("<Q", len(data)) + b"\0\0\0\0" + data + b"\0\0\0\0")
    assert [e["image/id"][0] for e in U.read_tfrecord(str(p))] == [b"0", b"1", b"2"]


# ---- drawing --------------------------------------------------------------------------------------------------------

def test_draw_boxes_numbers_outside_and_legible():
    im = Image.new("RGB", (576, 1024), (250, 250, 250))
    boxes = [(0.1, 0.2, 0.3, 0.25), (0.5, 0.5, 0.8, 0.6), (0.1, 0.8, 0.9, 0.9)]
    out, tags, outside = U.draw_boxes(im, boxes, numbered=True)
    assert out.size == im.size and im.getpixel((100, 300)) == (250, 250, 250)  # the original is untouched
    assert len(tags) == 3 and all(outside)
    for i, (b, t) in enumerate(zip(boxes, tags)):
        x0, y0, x1, y1 = round(b[0] * 576), round(b[1] * 1024), round(b[2] * 576), round(b[3] * 1024)
        # the coloured line sits just outside the element, the element's own pixels are not painted
        assert out.getpixel((x0 - 3, (y0 + y1) // 2)) == U.PALETTE[i]
        assert out.getpixel(((x0 + x1) // 2, (y0 + y1) // 2)) == (250, 250, 250)
        assert not U._rects_touch(t, (x0, y0, x1, y1))  # tag outside the box
        assert t[2] - t[0] >= 20 and t[3] - t[1] >= 20  # big enough to read
        assert out.getpixel((t[0] + 2, t[1] + 2)) == U.PALETTE[i]  # tag filled with the box colour
        assert any(out.getpixel((x, y)) == (255, 255, 255) for x in range(t[0], t[2]) for y in range(t[1], t[3]))  # white digit
    for i in range(3):
        for j in range(i + 1, 3):
            assert not U._rects_touch(tags[i], tags[j])


def test_draw_boxes_tag_falls_back_inside_when_nothing_fits():
    im = Image.new("RGB", (200, 200), "white")
    out, tags, outside = U.draw_boxes(im, [(0.0, 0.0, 1.0, 1.0)], numbered=True)
    assert outside == [False] and tags[0][0] >= 0 and tags[0][1] >= 0


def test_single_highlight_is_red_with_white_rim():
    im = Image.new("RGB", (400, 400), (200, 30, 30))  # a red background: the white rim keeps the box visible
    out, tags, _ = U.draw_boxes(im, [(0.25, 0.25, 0.75, 0.75)], numbered=False)
    assert tags == [None]
    row = [out.getpixel((x, 200)) for x in range(80, 100)]
    assert (255, 255, 255) in row and U.HIGHLIGHT in row


def test_render_writes_jpeg_max_side_1024(tmp_path):
    src = tmp_path / "s.png"
    Image.new("RGB", (1080, 1920), "white").save(src)
    dst = tmp_path / "d.jpg"
    U._render((str(src), str(dst), [(0.1, 0.1, 0.2, 0.2)], True))
    with Image.open(dst) as im:
        assert im.format == "JPEG" and max(im.size) == 1024


# ---- RefExp candidates ----------------------------------------------------------------------------------------------

def _grid(n=8, cols=4):
    """n well separated boxes in a grid (normalised)."""
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append((0.05 + c * 0.24, 0.1 + r * 0.2, 0.2 + c * 0.24, 0.2 + r * 0.2))
    return out


def test_refexp_uses_other_targets_first_and_keeps_boxes_apart():
    boxes = _grid()
    others = [boxes[1], boxes[2], boxes[3], boxes[4], (0.05, 0.1, 0.2, 0.2)]  # last one == the target: must not be drawn twice
    rng = random.Random(0)
    drawn, label, n_other, tier = U.refexp_candidates(boxes, 0, [""] * 8, rng, others, k=4)
    assert drawn[label] == boxes[0] and len(drawn) == 4 and n_other == 3 and tier == -1
    assert all(b in others for i, b in enumerate(drawn) if i != label)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not U.touches(drawn[i], drawn[j])


def test_refexp_fills_from_candidates_same_class_first():
    boxes = _grid()
    types = ["BTN", "TXT", "BTN", "BTN", "TXT", "TXT", "BTN", "TXT"]
    rng = random.Random(1)
    drawn, label, n_other, tier = U.refexp_candidates(boxes, 0, types, rng, [boxes[1]], tiers=[[1.0] * 8])
    assert n_other == 1 and tier == 0 and len(drawn) == 4 and drawn[label] == boxes[0]
    fillers = [boxes.index(b) for i, b in enumerate(drawn) if i != label and b != boxes[1]]
    assert all(types[i] == "BTN" for i in fillers)  # same class as the target preferred


def test_refexp_returns_none_when_four_separate_boxes_do_not_exist():
    boxes = [(0.1, 0.1, 0.3, 0.3), (0.15, 0.15, 0.35, 0.35), (0.6, 0.6, 0.8, 0.8)]
    assert U.refexp_candidates(boxes, 0, [""] * 3, random.Random(0), [], tiers=[[1.0] * 3]) is None
    assert U.refexp_candidates(boxes, 0, [""] * 3, random.Random(0), [boxes[2]] * 5, tiers=[[1.0] * 3]) is None


def test_refexp_gold_position_uniform():
    boxes = _grid()
    pos = Counter()
    for seed in range(600):
        drawn, label, _, _ = U.refexp_candidates(boxes, 0, [""] * 8, random.Random(seed), boxes[1:], k=4)
        pos[label] += 1
    assert min(pos.values()) > 0.7 * 150 and max(pos.values()) < 1.3 * 150


def test_fill_tiers_thin_by_propensity_and_last_tier_drops_status_bar():
    boxes = [(0.1, 0.0, 0.2, 0.03), (0.1, 0.5, 0.2, 0.55)]
    types = ["", "IMAGE_BUTTON"]
    prop = {U._feature(boxes[0], ""): 0.001, U._feature(boxes[1], "IMAGE_BUTTON"): 0.35}
    tiers = U.fill_tiers(boxes, types, prop, fulls=(0.35,))
    assert tiers[0][1] == 1.0 and tiers[0][0] < 0.01 and tiers[-1] == [0.0, 1.0]


# ---- widget captions ------------------------------------------------------------------------------------------------

def test_captions_conflict():
    assert U.captions_conflict(["go back"], ["back button"])            # same content words
    assert U.captions_conflict(["search"], ["search for music"])        # subset
    assert U.captions_conflict(["open the settings menu"], ["settings menu"])
    assert not U.captions_conflict(["go back"], ["open settings"])
    assert not U.captions_conflict(["share the photo"], ["delete the photo"])


def _w(screen, node, caps, box=(0.1, 0.1, 0.3, 0.15), app="app"):
    return {"screen": screen, "node": node, "captions": caps, "box": box, "app": app}


def test_caption_options_same_screen_first_never_identical_or_conflicting():
    t = _w("s", "0", ["go back", "back button"])
    same = [t, _w("s", "1", ["open settings"]), _w("s", "2", ["back"]), _w("s", "3", ["share photo", "share"])]
    app = [_w("o", "9", ["delete item"]), _w("o", "8", ["go back"]), _w("o", "7", ["play video"])]
    rng = random.Random(0)
    opts, label, n_same = U.caption_options(t, same, app, rng)
    assert len(opts) == 4 and opts[label] in t["captions"]
    assert len({o.lower() for o in opts}) == 4
    assert "back" not in opts and "go back" not in [o for i, o in enumerate(opts) if i != label]
    assert n_same == 2 and set(opts) >= {"open settings"} and ("share photo" in opts or "share" in opts)
    assert U.caption_options(t, same[:2], [], rng) is None  # fewer than three usable widgets


def test_find_candidates_only_distinct_non_touching_widgets():
    boxes = _grid()
    t = _w("s", "0", ["go back"], boxes[0])
    others = [_w("s", "1", ["back"], boxes[1]), _w("s", "2", ["open settings"], boxes[2]),
              _w("s", "3", ["share"], (0.06, 0.11, 0.19, 0.19)), _w("s", "4", ["play"], boxes[4]),
              _w("s", "5", ["stop"], boxes[5]), _w("s", "6", ["settings"], boxes[6])]
    chosen, label = U.find_candidates(t, [t] + others, random.Random(0), k=5)
    assert chosen[label] is t and len(chosen) == 4  # only three usable distractors exist: 4 boxes, not 5
    names = {c["node"] for c in chosen}
    assert "1" not in names and "3" not in names  # caption conflict, box overlap
    assert not ("2" in names and "6" in names)     # "open settings" vs "settings" conflict with each other
    assert U.find_candidates(t, [t, others[1], others[3]], random.Random(0)) is None


# ---- screen_annotation ----------------------------------------------------------------------------------------------

SA = ("TOOLBAR 0 999 30 109 (PICTOGRAM arrow backward 20 80 40 100, TEXT Settings (beta) 100 500 40 100), "
      "LIST_ITEM 45 955 116 179 (RADIO_BUTTON 93 160 129 166, LABEL USD 205 305 131 163), BUTTON SKIP 0 167 858 935")


def test_parse_screen_annotation_nesting_and_text_with_parentheses():
    els = U.parse_screen_annotation(SA)
    assert [e["type"] for e in els] == ["TOOLBAR", "PICTOGRAM", "TEXT", "LIST_ITEM", "RADIO_BUTTON", "LABEL", "BUTTON"]
    assert els[2]["text"] == "Settings (beta)" and els[2]["depth"] == 1 and els[2]["parent"] == 0
    assert els[6]["depth"] == 0 and els[6]["box"] == pytest.approx((0, 858 / 999, 167 / 999, 935 / 999))
    assert U.parse_screen_annotation("TEXT hello 1 2 3") is None
    assert U.parse_screen_annotation("TEXT a 1 2 3 4 (TEXT b 1 2 3 4") is None


def test_type_targets_skip_ambiguous_and_tiny_boxes():
    els = [{"type": "PICTOGRAM", "box": (0.4, 0.1, 0.6, 0.2)}, {"type": "IMAGE", "box": (0.41, 0.1, 0.6, 0.2)},
           {"type": "BUTTON", "box": (0.1, 0.5, 0.5, 0.6)}, {"type": "TEXT", "box": (0.1, 0.7, 0.11, 0.71)},
           {"type": "LABEL", "box": (0.1, 0.8, 0.5, 0.9)}]
    assert U.type_targets(els) == [2]


def test_draw_uniform_gold_is_exactly_uniform_given_the_option_set():
    """Whatever the pool sizes, P(gold = o | option set) = 1 / k: a text-only prior cannot pick the answer."""
    rng = random.Random(0)
    pools = {c: [("s", i) for i in range(n)] for c, n in
             {"TEXT": 3000, "BUTTON": 3000, "LIST_ITEM": 3000, "IMAGE": 3000, "SWITCH": 60, "CHECKBOX": 200}.items()}
    by_set = defaultdict(Counter)
    for opts, gold, inst in U.draw_uniform_gold(pools, rng, U.type_option_set, 4000):
        assert gold in opts and len(set(opts)) == 4
        by_set[tuple(sorted(opts))][gold] += 1
    n_rows = sum(sum(c.values()) for c in by_set.values())
    assert n_rows == 4000
    big = [c for c in by_set.values() if sum(c.values()) >= 200]
    assert big
    for c in big:  # every option of a frequent set is gold about a quarter of the time
        for cls in c:
            assert 0.15 < c[cls] / sum(c.values()) < 0.35
    # and a value that ran dry is never offered again
    dry = [(opts, gold) for opts, gold, _ in U.draw_uniform_gold({0: [1], 1: list(range(50)), 2: list(range(50)), 3: list(range(50)), 4: list(range(50))},
                                                                  random.Random(1), U.count_option_set, 200)]
    seen0 = [i for i, (o, g) in enumerate(dry) if 0 in o]
    assert seen0 and all(i <= [i for i, (o, g) in enumerate(dry) if g == 0][0] for i in seen0)


def test_count_option_set_is_a_window_of_neighbours_ascending():
    for seed in range(50):
        opts = U.count_option_set(range(13), random.Random(seed))
        assert opts == sorted(opts) and len(set(opts)) == 4 and opts[-1] - opts[0] <= 5


# ---- OmniACT --------------------------------------------------------------------------------------------------------

BOXES = {"e0": {"top_left": [100, 100], "bottom_right": [200, 150], "label": "save"},
         "e1": {"top_left": [400, 100], "bottom_right": [500, 150], "label": "open"},
         "e2": {"top_left": [100, 400], "bottom_right": [200, 450], "label": "close"},
         "e3": {"top_left": [400, 400], "bottom_right": [500, 450], "label": "help"},
         "e4": {"top_left": [700, 400], "bottom_right": [800, 450], "label": "save"}}  # same label as e0: never a swap target
SCRIPT = 'pyautogui.click(150.0,125.0)\npyautogui.write("hello")\npyautogui.press("enter")'


def test_parse_and_render_script():
    lines = U.parse_script("Pyautogui.click(150.0,125.0)\n pyautogui.press(\"enter\") ")
    assert lines and lines[0]["point"] == (150.0, 125.0) and lines[1]["point"] is None
    assert U.parse_script("click somewhere") is None
    assert U.render_script(U.parse_script(SCRIPT)) == SCRIPT


def test_swap_target_and_key_change_make_exactly_one_change():
    lines = U.parse_script(SCRIPT)
    swapped = U.render_script(U.swap_target(lines, (150.0, 125.0), (450.0, 125.0))).split("\n")
    assert swapped[0] == "pyautogui.click(450.0,125.0)" and swapped[1:] == SCRIPT.split("\n")[1:]
    assert U.render_script(U.swap_target(U.parse_script("pyautogui.click(150, 125)"), (150.0, 125.0), (450.5, 125.0))) == "pyautogui.click(450, 125)"
    keyed = U.render_script(U.change_key(lines, random.Random(0))).split("\n")
    assert keyed[:2] == SCRIPT.split("\n")[:2] and keyed[2].startswith('pyautogui.press("') and keyed[2] != 'pyautogui.press("enter")'
    assert U.render_script(U.change_key(U.parse_script("pyautogui.scroll(10)"), random.Random(0))) == "pyautogui.scroll(-10)"
    assert U.change_key(U.parse_script("pyautogui.click(1,2)"), random.Random(0)) is None


def test_script_options_designs_are_symmetric_and_distinct():
    centers = U.box_centers(BOXES)
    lines = U.parse_script(SCRIPT)
    other = U.parse_script("pyautogui.click(450.0,425.0)")
    for design in ("targets", "keys", "tasks"):
        opts, label, d = U.script_options(lines, centers, (1000, 600), [other], {}, random.Random(3), design=design)
        assert d == design and len(set(opts)) == 4 and opts[label] == SCRIPT
        assert all("750.0,425.0" not in o for o in opts)  # e4 carries the gold's label "save": never a distractor
        if design == "targets":
            assert all(o.split("\n")[1:] == SCRIPT.split("\n")[1:] for o in opts)  # only the click differs
        if design == "keys":
            firsts = Counter(o.split("\n")[0] for o in opts)
            assert sorted(firsts.values()) == [2, 2]  # two targets x two keys
    pos = Counter(U.script_options(lines, centers, (1000, 600), [other], {}, random.Random(s))[1] for s in range(400))
    assert min(pos.values()) > 60


def test_other_targets_keeps_a_visible_gap_and_prefers_popular_elements():
    centers = U.box_centers(BOXES)
    lines = U.parse_script("pyautogui.click(150.0,125.0)")
    alts = U.other_targets(lines, centers, (1000, 600), {"e3": 50}, random.Random(0), 1)
    assert alts and alts[0]["key"] == "e3"
    near = [{"key": "n", "label": "x", "center": (160.0, 125.0), "rect": (150, 100, 170, 150)}]
    assert U.other_targets(lines, centers + near, (1000, 600), {}, random.Random(0), 5) == [c for c in U.other_targets(lines, centers, (1000, 600), {}, random.Random(0), 5)]


def test_omniact_val_apps_ranked_by_val_to_train_ratio():
    data = {"train": [{"app": "a"}] * 10 + [{"app": "b"}] * 100 + [{"app": "c"}] * 20,
            "val": [{"app": "a"}] * 5 + [{"app": "b"}] * 5 + [{"app": "c"}] * 8}
    assert U.omniact_val_apps(data, target=12) == ["a", "c"]


# ---- split sides ----------------------------------------------------------------------------------------------------

def test_assign_sides_direct_matches_only():
    ids = ["t1", "d1", "x", "y", "z", "sq"]
    splits = {"src": {"train": {"t1", "x", "y", "z"}, "val": {"d1"}, "test": {"t1"}}}
    sq = {"train": {"sq"}, "val": set()}
    pairs = [(0, 2), (1, 3), (3, 4), (5, 1)]  # x~t1, y~d1, z~y (chain: z is NOT near d1), sq~d1
    side = U.assign_sides(ids, pairs, splits, sq)
    assert side == {"t1": "test", "x": "test", "d1": "none", "y": "none", "z": "train", "sq": "none"}
    side = U.assign_sides(ids, [(1, 3)], splits, {"train": set(), "val": set()})
    assert side["d1"] == "val" and side["y"] == "none" and side["z"] == "train"


def test_near_pairs_matches_bytes_and_near_thumbnails():
    import numpy as np

    t0 = np.random.RandomState(0).randint(0, 255, 4096).astype(np.uint8)
    t1 = np.clip(t0.astype(int) + 2, 0, 255).astype(np.uint8)
    t2 = np.random.RandomState(1).randint(0, 255, 4096).astype(np.uint8)
    pairs = U.near_pairs(["a", "b", "c", "a"], [1, 1 ^ 3, 1 << 40, None], np.stack([t0, t1, t2, t2]), 8, 0.98)
    assert pairs == [(0, 1), (0, 3)]


# ---- text-only baseline ---------------------------------------------------------------------------------------------

def _write_set(root, name, gold_longest, n=600):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    rng = random.Random(0)
    words = "alpha beta gamma delta eps zeta eta theta iota kappa".split()
    for split, m in (("train", n), ("val", n // 3)):
        with open(os.path.join(d, split + ".jsonl"), "w") as f:
            for i in range(m):
                opts = [" ".join(rng.sample(words, rng.randint(1, 3))) for _ in range(4)]
                if gold_longest:
                    lab = max(range(4), key=lambda j: len(opts[j]))
                else:
                    lab = rng.randrange(4)
                f.write(json.dumps({"id": "%s-%d" % (split, i), "state_text": None, "label": lab,
                                    "question": {"type": "choice", "instructions": "pick", "criteria": opts}}) + "\n")


def test_text_only_baseline_finds_a_length_leak_and_stays_at_chance_otherwise(tmp_path):
    _write_set(str(tmp_path), "leaky", True)
    _write_set(str(tmp_path), "clean", False)
    leaky = U.text_only_baseline("leaky", str(tmp_path), epochs=6)
    clean = U.text_only_baseline("clean", str(tmp_path), epochs=6)
    assert leaky["text_only_val"] > 0.8 and abs(clean["text_only_val"] - 0.25) < 0.12


# ---- the built sets -------------------------------------------------------------------------------------------------

def _rows(name, split):
    with open(os.path.join(ROOT, name, split + ".jsonl")) as f:
        return [json.loads(l) for l in f]


@pytest.mark.skipif(not BUILT, reason="no built ui_* / omniact_* set under data/vqa")
@pytest.mark.parametrize("name", BUILT)
def test_built_splits_share_no_image_or_screen(name):
    rows = {s: _rows(name, s) for s in ("train", "val")}
    files = {s: {r["image"] for r in rows[s]} for s in rows}
    assert not files["train"] & files["val"]
    ids = [r["id"] for s in rows for r in rows[s]]
    assert len(ids) == len(set(ids))
    key = "rico_screen" if name.startswith("ui_") else "screen"
    screens = {s: {r["attribution"][key] for r in rows[s]} for s in rows}
    assert not screens["train"] & screens["val"]
    if name.startswith("ui_"):
        apps = {s: {r["attribution"]["app"] for r in rows[s]} for s in rows} if name.startswith("omniact") else None
        sq = U.screenqa_screens(ROOT)
        assert not screens["train"] & sq["val"] and not screens["val"] & sq["train"]
    else:
        apps = {s: {r["attribution"]["app"] for r in rows[s]} for s in rows}
        assert not apps["train"] & apps["val"]
    with open(os.path.join(ROOT, name, "manifest.json")) as f:
        m = json.load(f)
    assert m["license"]["commercial"] is True and m["revision"] and m["attribution"]
    for s in rows:
        pos = Counter((len(r["question"]["criteria"]), r["label"]) for r in rows[s])
        for r in rows[s]:
            assert 0 <= r["label"] < len(r["question"]["criteria"]) and len(set(r["question"]["criteria"])) == len(r["question"]["criteria"])
        for k in {k for k, _ in pos}:
            counts = [pos[(k, i)] for i in range(k)]
            if sum(counts) >= 200:
                assert min(counts) > 0.6 * sum(counts) / k and max(counts) < 1.4 * sum(counts) / k, (name, s, k, counts)


@pytest.mark.skipif(not BUILT, reason="no built ui_* / omniact_* set under data/vqa")
@pytest.mark.parametrize("name", BUILT)
def test_built_images_open_with_pil(name):
    base = os.path.join(ROOT, name)
    files = sorted({r["image"] for s in ("train", "val") for r in _rows(name, s)})
    assert files
    for i, rel in enumerate(files):  # header of every image, full decode of a sample
        with Image.open(os.path.join(base, rel)) as im:
            assert im.format == "JPEG" and max(im.size) <= U.MAX_SIDE
            if i % 200 == 0:
                im.load()


def test_specific_caption_prefers_distinguishing_words_and_rejects_generic_only():
    assert U.specific_caption(["toggle setting on/off", "turn on SMS log", "switch"]) == "turn on SMS log"
    assert U.specific_caption(["toggle setting on/off", "select option"]) is None
    assert U.specific_caption(["go back", "back button"]) == "go back"  # short, names a distinct function
    assert U.caption_options({"captions": ["toggle on/off"]}, [], [], random.Random(0)) is None
