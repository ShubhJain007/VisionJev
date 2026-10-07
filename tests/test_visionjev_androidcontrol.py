"""AndroidControl builder (laya.visionjev_androidcontrol) without the 50 GB download: the TFRecord / tf.train.Example /
accessibility-forest readers on records encoded here by hand, gold-element selection and box drawing on a hand-made
tree, option construction (the gold never in the shown history, positions uniform), an end-to-end build from two tiny
synthetic shards, and the invariants of the real built files when they are on disk (episodes disjoint between splits,
images exist)."""
import gzip
import io
import json
import os
import random
import struct
from collections import Counter

import pytest

from laya import visionjev_androidcontrol as AC
from laya.visionjev_data import DATA_ROOT

# ---- a minimal protobuf / TFRecord writer (the builder only reads) ---------------------------------------------------------


def _varint(n):
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _vi(field, n):
    return _varint(field << 3) + _varint(n)


def _ld(field, payload):
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _feature(values, packed=True):
    if values and isinstance(values[0], bytes):
        return _ld(1, b"".join(_ld(1, v) for v in values))
    if values and isinstance(values[0], float):
        return _ld(2, _ld(1, struct.pack("<%df" % len(values), *values)))
    body = _ld(1, b"".join(_varint(v) for v in values)) if packed else b"".join(_vi(1, v) for v in values)
    return _ld(3, body)


def _example(features, packed=True):
    entries = b"".join(_ld(1, _ld(1, k.encode()) + _ld(2, _feature(v, packed))) for k, v in features.items())
    return _ld(1, entries)


def _tfrecord(records):
    return b"".join(struct.pack("<Q", len(r)) + b"\0\0\0\0" + r + b"\0\0\0\0" for r in records)


def _rect(b):
    return b"".join(_vi(i + 1, v) for i, v in enumerate(b))


def _pb_node(uid, bounds, cls="android.widget.TextView", text="", desc="", clickable=False, visible=True, kids=()):
    out = _vi(1, uid) + _ld(2, _rect(bounds)) + _ld(3, cls.encode())
    if desc:
        out += _ld(4, desc.encode())
    if text:
        out += _ld(7, text.encode())
    out += _vi(14, int(clickable)) + _vi(16, 1) + _vi(23, int(visible))
    if kids:
        out += _ld(25, b"".join(_varint(k) for k in kids))
    return out


def _pb_window(nodes, bounds=(0, 0, 1080, 2400), layer=0, wtype=1):
    return _ld(1, _ld(1, _rect(bounds)) + _vi(4, layer) + _vi(6, wtype) + _ld(11, b"".join(_ld(1, n) for n in nodes)))


def test_tfrecord_and_example_parsing_on_a_synthetic_record(tmp_path):
    feats = {"episode_id": [42], "goal": ["Open the clock app".encode()], "actions": [b'{"action_type":"wait"}', b"{}"],
             "screenshot_widths": [1080, 1080, 1080], "neg": [-5, 1 << 40], "scores": [0.5, -2.0]}
    rec = _example(feats)
    ex = AC.parse_example(rec)
    assert ex["episode_id"] == [42] and ex["goal"] == [b"Open the clock app"] and ex["actions"] == feats["actions"]
    assert ex["screenshot_widths"] == [1080] * 3 and ex["neg"] == [-5, 1 << 40] and ex["scores"] == [0.5, -2.0]
    assert AC.parse_example(_example({"ids": [7, 300]}, packed=False))["ids"] == [7, 300]  # unpacked repeated int64
    assert AC.parse_example(_ld(1, _ld(1, _ld(1, b"empty") + _ld(2, b""))))["empty"] == []
    recs = [rec, _example({"episode_id": [43]})]
    plain, zipped = tmp_path / "plain.tfrecord", tmp_path / "zipped.tfrecord"
    plain.write_bytes(_tfrecord(recs))
    with gzip.open(zipped, "wb") as f:
        f.write(_tfrecord(recs))
    for p in (plain, zipped):
        got = list(AC.read_records(str(p)))
        assert got == recs and [AC.parse_example(r)["episode_id"] for r in got] == [[42], [43]]
    (tmp_path / "cut.tfrecord").write_bytes(_tfrecord(recs)[:-9])
    with pytest.raises(ValueError):
        list(AC.read_records(str(tmp_path / "cut.tfrecord")))
    ep = AC.episode_of(AC.parse_example(_example({"episode_id": [9], "goal": [b"g"]})))  # a record without actions
    assert ep["actions"] == [] and ep["episode_id"] == 9 and ep["goal"] == "g"


def _forest():
    """A settings screen: a toolbar with a back button and a title, three clickable rows (the label sits in a child
    text view), a switch inside the first row, an invisible node, a tiny node, and a keyboard window over the bottom."""
    app = [
        _pb_node(0, (0, 0, 1080, 2400), cls="android.widget.FrameLayout", kids=(1, 2, 3, 5, 7, 9, 10)),
        _pb_node(1, (0, 100, 150, 250), cls="android.widget.ImageButton", desc="Navigate up", clickable=True),
        _pb_node(2, (150, 100, 1080, 250), text="Settings"),
        _pb_node(3, (0, 300, 1080, 500), cls="android.widget.LinearLayout", clickable=True, kids=(4, 11)),
        _pb_node(4, (40, 340, 700, 460), text="Wi-Fi"),
        _pb_node(5, (0, 500, 1080, 700), cls="android.widget.LinearLayout", clickable=True, kids=(6,)),
        _pb_node(6, (40, 540, 700, 660), text="Bluetooth"),
        _pb_node(7, (0, 700, 1080, 900), cls="android.widget.LinearLayout", clickable=True, kids=(8,)),
        _pb_node(8, (40, 740, 700, 860), text="Display"),
        _pb_node(9, (0, 900, 1080, 1100), cls="android.widget.LinearLayout", text="Hidden", clickable=True, visible=False),
        _pb_node(10, (500, 1200, 510, 1210), cls="android.widget.ImageView", desc="dot", clickable=True),
        _pb_node(11, (850, 340, 1040, 460), cls="android.widget.Switch", desc="Wi-Fi switch", clickable=True),
        _pb_node(12, (0, 1900, 1080, 2100), cls="android.widget.LinearLayout", text="Under the keyboard", clickable=True),
    ]
    keys = [_pb_node(0, (0, 1800, 1080, 2400), cls="android.widget.FrameLayout", kids=(1, 2)),
            _pb_node(1, (0, 1850, 108, 2000), cls="android.inputmethod.Key", desc="q", clickable=True),
            _pb_node(2, (108, 1850, 216, 2000), cls="android.inputmethod.Key", desc="w", clickable=True)]
    return _pb_window(app, layer=0, wtype=1) + _pb_window(keys, bounds=(0, 1800, 1080, 2400), layer=1, wtype=2)


def test_forest_parsing_gold_node_and_labels():
    nodes = AC.parse_forest(_forest())
    assert len(nodes) == 16 and {n["wtype"] for n in nodes} == {1, 2}
    row = AC.target_node(nodes, 300, 400)  # a tap on the "Wi-Fi" text lands on the clickable row, not the text view
    assert row["cls"].endswith("LinearLayout") and row["bounds"] == (0, 300, 1080, 500) and AC.node_label(row) == "Wi-Fi"
    switch = AC.target_node(nodes, 900, 400)  # the smallest clickable node containing the tap: the switch in the row
    assert switch["cls"].endswith("Switch") and AC.node_label(switch) == "Wi-Fi switch"
    assert AC.target_node(nodes, 50, 1900)["desc"] == "q"  # the keyboard is in front of the app row under it
    assert AC.target_node(nodes, 600, 150)["text"] == "Settings"  # no clickable node there: the labelled one
    assert AC.target_node(nodes, 600, 150, clickable_only=True) is None
    assert AC.target_node(nodes, 600, 1000) is None or AC.target_node(nodes, 600, 1000)["text"] != "Hidden"
    boxes = {AC.node_label(n) for n in AC.clickable_nodes(nodes, 1080, 2400)}
    assert boxes == {"Navigate up", "Wi-Fi", "Bluetooth", "Display", "Wi-Fi switch", "q", "w"}  # no hidden, tiny, covered
    assert AC.render_action({"action_type": "click", "x": 300, "y": 600}, nodes, 1080, 2400) == 'click "Bluetooth" (top center)'
    assert AC.render_action({"action_type": "long_press", "x": 60, "y": 150}, nodes, 1080, 2400) == 'long press "Navigate up" (top left)'
    assert AC.render_action({"action_type": "click", "x": 600, "y": 2000}, [], 1080, 2400) == "click at the bottom center of the screen"
    assert AC.render_action({"action_type": "input_text", "text": "pizza  near me"}, [], 0, 0) == 'type "pizza near me"'
    assert AC.render_action({"action_type": "scroll", "direction": "down"}, [], 0, 0) == "scroll down"
    assert AC.render_action({"action_type": "open_app", "app_name": "Clock"}, [], 0, 0) == 'open the app "Clock"'
    assert AC.render_action({"action_type": "navigate_back"}, [], 0, 0) == "go back"
    assert AC.bucket(540, 1200, 1080, 2400) == "center" and AC.bucket(1079, 2399, 1080, 2400) == "bottom right"


def test_element_candidates_gold_box_and_uniform_numbers():
    nodes = AC.parse_forest(_forest())
    positions = {4: Counter(), 5: Counter(), 6: Counter()}
    for seed in range(600):
        got = AC.element_candidates(nodes, 300, 600, 1080, 2400, random.Random(seed))  # the tap on "Bluetooth"
        assert got is not None
        boxes, gold = got
        assert 4 <= len(boxes) <= 6 and len(set(boxes)) == len(boxes)
        assert boxes[gold] == (0, 500, 1080, 700)
        for i, b in enumerate(boxes):
            for j, c in enumerate(boxes):
                assert i == j or AC._overlap(b, c) == 0  # no nested or crossing boxes: the Wi-Fi row or its switch, not both
        positions[len(boxes)][gold] += 1
    for k, c in positions.items():  # the gold's number is uniform over 1..k
        assert set(c) == set(range(k)) and max(c.values()) < 2 * min(c.values()), (k, c)
    # a tap that hits no clickable node, or a screen with too few other elements, gives no row
    assert AC.element_candidates(nodes, 600, 150, 1080, 2400, random.Random(0)) is None
    few = [n for n in nodes if n["wtype"] == 1 and AC.node_label(n) in ("Wi-Fi", "Bluetooth", "Display")]
    assert AC.element_candidates(few, 300, 600, 1080, 2400, random.Random(0)) is None


def test_draw_boxes_numbers_are_outside_the_boxes_and_apart():
    from PIL import Image

    im = Image.new("RGB", (461, 1024), (30, 30, 30))
    boxes = [(0, 300, 1080, 500), (0, 500, 1080, 700), (0, 100, 150, 250), (850, 340, 1040, 460), (0, 1850, 108, 2000)]
    boxes = [boxes[0], boxes[1], boxes[2], boxes[4]] + [(900, 100, 1080, 250)]
    scale = 461 / 1080.0
    tags = AC.draw_boxes(im, boxes, scale)
    assert len(tags) == len(boxes)
    px = im.load()
    for i, (b, t) in enumerate(zip(boxes, tags)):
        assert 0 <= t[0] < t[2] <= im.width and 0 <= t[1] < t[3] <= im.height
        assert t[2] - t[0] >= 14 and t[3] - t[1] >= 18  # a readable digit: at least ~20 px tall on a 1024 px screen
        colour = AC.PALETTE[i]
        assert px[t[0] + 2, t[1] + 2] == colour  # the tag is filled with the box colour ...
        inner = [px[x, y] for x in range(t[0] + 2, t[2] - 2) for y in range(t[1] + 2, t[3] - 2)]
        assert (255, 255, 255) in inner  # ... and carries a white digit
        r = [int(round(v * scale)) for v in b]
        assert px[r[0] + 1, (r[1] + r[3]) // 2] == colour  # the outline
        for j, o in enumerate(tags):
            assert i == j or AC._overlap(t, o) == 0
        for j, o in enumerate(boxes):
            if i != j:
                assert AC._overlap(t, tuple(int(round(v * scale)) for v in o)) == 0  # a tag never covers another box
    assert len({AC.PALETTE[i] for i in range(6)}) == 6


def test_next_action_options_never_repeat_the_history_and_positions_are_uniform():
    type_p = {"click": 0.6, "scroll": 0.1, "input_text": 0.1, "open_app": 0.1, "navigate_back": 0.05, "wait": 0.05}
    pools = {"input_text": ['type "%s"' % w for w in ("pizza", "hotel", "shoes", "jazz")],
             "open_app": ['open the app "%s"' % a for a in ("Clock", "Maps", "Gmail", "Notes")]}
    history = ['open the app "Settings"', "scroll down", 'click "Network" (top center)']
    later = ['click "Save" (bottom right)', "go back", "scroll down"]
    screen = ['click "%s" (middle left)' % w for w in ("Wi-Fi", "Bluetooth", "Display", "Sound", "Network")]
    gold = 'click "Wi-Fi switch" (top right)'
    positions, sizes, seen_later, seen_screen = {4: Counter(), 5: Counter(), 6: Counter()}, Counter(), 0, 0
    for seed in range(900):
        opts, label = AC.next_action_options(gold, history, later, [s for s in screen if s != gold], random.Random(seed), type_p, pools)
        assert opts[label] == gold and 4 <= len(opts) <= 6 and len(set(opts)) == len(opts)
        assert not set(opts) & set(history)  # "scroll down" and the Network click were already taken
        seen_later += bool(set(opts) & set(later))
        seen_screen += bool(set(opts) & set(screen))
        positions[len(opts)][label] += 1
        sizes[len(opts)] += 1
    assert seen_later == 900 and seen_screen == 900 and set(sizes) == {4, 5, 6}
    for k, c in positions.items():  # the gold's position is uniform
        assert set(c) == set(range(k)) and max(c.values()) < 2 * min(c.values()), (k, c)
    # screen elements that share words with the goal are used first (keyword matching must not find the gold)
    ranked = AC.hard_first(screen, "Turn on Bluetooth and lower the display brightness", random.Random(1))
    assert set(ranked[:2]) == {'click "Bluetooth" (middle left)', 'click "Display" (middle left)'} and sorted(ranked) == sorted(screen)
    assert sorted(AC.hard_first(screen, "Book a flight", random.Random(1))) == sorted(screen)
    opts, _ = AC.next_action_options(gold, history, [], ranked, random.Random(3), {"scroll": 1.0}, pools, k=4)
    assert 'click "Bluetooth" (middle left)' in opts or 'click "Display" (middle left)' in opts
    # a gold that repeats a previous action is skipped, however many distractors there are
    assert AC.next_action_options("scroll down", history, later, screen, random.Random(0), type_p, pools) is None
    # which app to open: other apps are among the options
    for seed in range(50):
        opts, label = AC.next_action_options('open the app "Settings"', [], later, screen, random.Random(seed), type_p, pools)
        assert sum(o.startswith("open the app") for o in opts) >= 3
    # nothing to choose from: no row
    assert AC.next_action_options(gold, [], [], [], random.Random(0), {"click": 1.0}, {}) is None


def test_truncation_steps_lie_between_30_and_80_percent():
    assert AC.truncation_steps(1) == [] and AC.truncation_steps(2) == [1] and AC.truncation_steps(3) == [1, 2]
    assert AC.truncation_steps(10) == [3, 4, 5, 6, 7, 8]
    for n in range(2, 40):
        assert all(0 < t < n and 0.3 <= t / n <= 0.8 for t in AC.truncation_steps(n))


# ---- end to end on synthetic shards -----------------------------------------------------------------------------------------


def _png(seed, step):
    from PIL import Image, ImageDraw

    rng = random.Random("%s-%s" % (seed, step))
    im = Image.new("RGB", (270, 600), tuple(rng.randint(0, 255) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x, y = rng.randint(0, 200), rng.randint(0, 520)
        d.rectangle([x, y, x + rng.randint(20, 70), y + rng.randint(20, 80)], fill=tuple(rng.randint(0, 255) for _ in range(3)))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _screen_forest(eid):
    rows = [_pb_node(i + 1, (0, 60 + 80 * i, 270, 130 + 80 * i), cls="android.widget.LinearLayout", text="item %d-%d" % (eid, i),
                     clickable=True) for i in range(6)]
    return _pb_window([_pb_node(0, (0, 0, 270, 600), cls="android.widget.FrameLayout", kids=tuple(range(1, 7)))] + rows,
                      bounds=(0, 0, 270, 600))


def _episode(eid):
    actions = [{"action_type": "open_app", "app_name": "App %d" % eid}, {"action_type": "click", "x": 100, "y": 100},
               {"action_type": "input_text", "text": "query %d" % eid}, {"action_type": "click", "x": 100, "y": 260},
               {"action_type": "scroll", "direction": "down"}, {"action_type": "click", "x": 100, "y": 420}]
    n = len(actions)
    return _example({"episode_id": [eid], "goal": [("Do the thing %d" % eid).encode()],
                     "actions": [json.dumps(a).encode() for a in actions],
                     "step_instructions": [("step %d of %d" % (i, eid)).encode() for i in range(n)],
                     "screenshots": [_png(eid, i) for i in range(n + 1)],
                     "accessibility_trees": [_screen_forest(eid) for _ in range(n + 1)],
                     "screenshot_widths": [270] * (n + 1), "screenshot_heights": [600] * (n + 1)})


def _check_built(root, names=AC.NAMES):
    for name in names:
        base = os.path.join(root, name)
        rows = {s: [json.loads(l) for l in open(os.path.join(base, s + ".jsonl"))] for s in ("train", "val")}
        assert rows["train"] and rows["val"]
        eps = {s: {r["episode"] for r in rs} for s, rs in rows.items()}
        assert not eps["train"] & eps["val"]  # no episode in both splits
        for s, rs in rows.items():
            assert len({r["id"] for r in rs}) == len(rs)
            for r in rs:
                assert os.path.isfile(os.path.join(base, r["image"])), r["image"]
                q = r["question"]
                if q["type"] == "choice":
                    assert 4 <= len(q["criteria"]) <= 6 and len(set(q["criteria"])) == len(q["criteria"])
                    assert 0 <= r["label"] < len(q["criteria"])
                else:
                    assert r["label"] in (0, 1)
        used = {r["image"] for rs in rows.values() for r in rs}
        assert {"images/" + f for f in os.listdir(os.path.join(base, "images"))} == used  # no orphan images
        m = json.load(open(os.path.join(base, "manifest.json")))
        assert m["license"]["commercial"] is True and "CC BY 4.0" in m["license"]["annotations"]
        assert m["counts"] == {s: len(rs) for s, rs in rows.items()}
        yield name, rows


def test_build_end_to_end_on_synthetic_shards(tmp_path):
    from PIL import Image

    raw, root = tmp_path / "raw", tmp_path / "vqa"
    raw.mkdir()
    root.mkdir()
    eids = list(range(100, 124))
    for shard in range(2):
        with gzip.open(raw / (AC.SHARD % (shard, AC.N_SHARDS)), "wb") as f:
            f.write(_tfrecord([_episode(e) for e in eids if e % 2 == shard] + ([_example({"episode_id": [999]})] if shard else [])))
    (raw / "splits.json").write_text(json.dumps({"train": eids[:16] + [999], "validation": eids[16:19], "test": eids[19:]}))
    assert AC.load_splits(str(raw))[100] == "train" and AC.load_splits(str(raw))[123] == "val"
    AC.build(str(root), str(raw), workers=2, log=lambda m: None)
    built = dict(_check_built(str(root)))
    for name, rows in built.items():
        assert {r["episode"] for r in rows["train"]} <= set(eids[:16]) and {r["episode"] for r in rows["val"]} <= set(eids[16:])
    # next action: the gold is the human's action at that step, never one already in the shown history
    for r in built["ac_next_action"]["train"] + built["ac_next_action"]["val"]:
        s, gold = json.loads(r["state_text"]), r["question"]["criteria"][r["label"]]
        assert s["goal"] == "Do the thing %d" % r["episode"] and len(s["previous_actions"]) == r["step"]
        assert not set(r["question"]["criteria"]) & set(s["previous_actions"])
        expect = {0: 'open the app "App %d"' % r["episode"], 1: 'click "item %d-0" (top center)' % r["episode"],
                  2: 'type "query %d"' % r["episode"], 3: 'click "item %d-2" (center)' % r["episode"], 4: "scroll down",
                  5: 'click "item %d-4" (bottom center)' % r["episode"]}
        assert gold == expect[r["step"]]
        assert Image.open(os.path.join(str(root), "ac_next_action", r["image"])).size == (270, 600)
    # element: the gold box is the row under the tap; the instruction is the step's
    for r in built["ac_element"]["train"]:
        assert json.loads(r["state_text"])["instruction"] == "step %d of %d" % (r["step"], r["episode"]) and r["step"] in (1, 3, 5)
    # goal reached: each episode gives its final screen (yes) and one cut at 30-80% (no)
    for split, rows in built["ac_goal_reached"].items():
        by_ep = {}
        for r in rows:
            by_ep.setdefault(r["episode"], {})[r["label"]] = r["step"]
        assert all(set(v) == {0, 1} and v[1] == 6 and v[0] in (2, 3, 4) for v in by_ep.values())
        assert Counter(r["label"] for r in rows)[0] == Counter(r["label"] for r in rows)[1]
    m = json.load(open(root / "ac_goal_reached" / "manifest.json"))
    assert m["build_stats"]["episodes_without_actions"] == 1 and "by construction" in m["labels"]


@pytest.mark.skipif(not all(os.path.exists(os.path.join(DATA_ROOT, n, "_READY")) for n in AC.NAMES),
                    reason="the AndroidControl sets are not built on this machine")
def test_built_files_episodes_disjoint_and_images_exist():
    splits = AC.load_splits() if os.path.exists(os.path.join(AC.RAW, "splits.json")) else None
    for name, rows in _check_built(DATA_ROOT):
        assert 300 <= len(rows["val"]) <= AC.VAL_CAP and len(rows["train"]) <= AC.TRAIN_CAP
        if splits:  # the official split: train episodes in train, validation / test episodes in val
            assert all(splits[r["episode"]] == s for s, rs in rows.items() for r in rs)
        if name == "ac_goal_reached":
            for rs in rows.values():
                c = Counter(r["label"] for r in rs)
                assert c[0] == c[1]
        else:
            for rs in rows.values():
                by_k = {}
                for r in rs:
                    by_k.setdefault(len(r["question"]["criteria"]), Counter())[r["label"]] += 1
                for k, c in by_k.items():  # gold positions about uniform
                    assert max(c.values()) < 1.5 * sum(c.values()) / k + 15, (name, k, c)
