"""AndroidControl (Google Research): human demonstrations of tasks on Android apps, as typed decisions on a screenshot.

Source: ``gs://gresearch/android_control`` (20 GZIP TFRecord shards, ~50 GB, + ``splits.json``), described in
github.com/google-research/google-research/tree/master/android_control (Li et al., "On the Effects of Data Scale on
Computer Control Agents", arXiv 2406.03679). Licence as read on 2026-09-30: the bucket holds no licence file; the
google-research repository README states "All datasets in this repository are released under the CC BY 4.0
International license" (source files Apache-2.0, which is the notice at the end of android_control/README.md). The
screenshots show third-party app UIs. Every action is a human rater's; nothing here is model-generated.

Read without TensorFlow or protobuf: a TFRecord file is length-prefixed records (``read_records``), each record a
``tf.train.Example`` and each accessibility tree an ``AndroidAccessibilityForest`` (android_env), both decoded by a
small protobuf wire-format reader (``fields``).

Split: the official ``splits.json``. Train rows come from ``train`` episodes; val rows from ``validation`` + ``test``
episodes (never an episode in both). Val screenshots that also appear in train (the launcher's home screen, an app's
start screen) are dropped with the data validator's own matcher.

* ``ac_next_action``  (choice, 4-6) goal + the previous actions (text) + the current screenshot -> which action comes
                      next? Options are rendered actions (``click "Settings" (top right)``, ``type "pizza"``, ``scroll
                      down``, ...). Distractors: later actions of the same episode, clicks on OTHER elements of the
                      current screen (accessibility tree; the ones sharing words with the goal first), and other actions
                      drawn so that the action types of the distractors follow the types of the gold actions. No option is an action already in the history;
                      a gold that repeats a previous action is skipped.
* ``ac_element``      (choice, 4-6) the step's low-level instruction + the screenshot with numbered boxes drawn on
                      clickable elements -> which box? Gold = the smallest clickable node containing the human tap;
                      distractors = other clickable nodes that do not touch the gold box (same window and class and
                      nearby first). Steps whose instruction covers several actions are skipped.
* ``ac_goal_reached`` (noul, balanced) goal + one screenshot -> has the goal been reached? yes = the final screenshot of
                      the complete human episode; no = the screenshot of the SAME episode cut at 30-80% of its steps.
                      The "no" label is by construction (the episode was cut), not a human judgement.

    python -m laya.visionjev_androidcontrol download            # ~50 GB into <raw>/android_control
    python -m laya.visionjev_androidcontrol build --workers 8
    python -m laya.visionjev_androidcontrol audit                # question-only (no screenshot) baseline, next action
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import random
import re
import shutil
import struct
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

from .visionjev_data import DATA_ROOT, RAW_ROOT

RAW = os.path.join(RAW_ROOT, "android_control")
BUCKET = "https://storage.googleapis.com/gresearch/android_control/"
N_SHARDS = 20
SHARD = "android_control-%05d-of-%05d"
MAX_SIDE = 1024
TRAIN_CAP, VAL_CAP = 20000, 1000
NAMES = ("ac_next_action", "ac_element", "ac_goal_reached")
IMAGE_LICENSE = "CC BY 4.0 (AndroidControl, Google Research)"
LICENSE = {"annotations": "CC BY 4.0 (AndroidControl, Google Research; human demonstrations)",
           "images": "AndroidControl screenshots (CC BY 4.0 dataset; they show third-party app UIs)", "commercial": True,
           "verified": "2026-09-30: google-research README (\"All datasets in this repository are released under the CC BY "
                       "4.0 International license\"); android_control/README.md carries the Apache-2.0 notice for the code; "
                       "no licence file in the bucket. Attribution: Li et al. 2024, arXiv 2406.03679"}
Q_NEXT = "Which action should be taken next to reach the goal?"
Q_ELEMENT = "Which numbered box marks the element to tap to carry out the instruction?"
Q_GOAL = "Has the goal been reached on this screen?"

# ---- protobuf wire format and TFRecord ------------------------------------------------------------------------------------


def _varint(buf, i: int) -> Tuple[int, int]:
    r = s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7F) << s
        if b < 0x80:
            return r, i
        s += 7


def fields(buf) -> Iterator[Tuple[int, int, object]]:
    """(field number, wire type, value) of one protobuf message: varint -> int, length-delimited -> the bytes slice,
    fixed 64 / 32 -> their raw bytes."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        wire = key & 7
        if wire == 0:
            v, i = _varint(buf, i)
        elif wire == 2:
            size, i = _varint(buf, i)
            v, i = buf[i:i + size], i + size
            if len(v) != size:
                raise ValueError("truncated protobuf field")
        elif wire == 1:
            v, i = buf[i:i + 8], i + 8
        elif wire == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError("unsupported protobuf wire type %d" % wire)
        yield key >> 3, wire, v


def _packed(wire: int, v) -> List[int]:
    if wire == 0:
        return [v]
    out, j = [], 0
    while j < len(v):
        x, j = _varint(v, j)
        out.append(x)
    return out


def _signed(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def parse_example(buf) -> Dict[str, list]:
    """A serialized ``tf.train.Example`` -> {feature name: list of bytes | ints | floats}."""
    out = {}
    for f, _, features in fields(memoryview(buf)):
        if f != 1:
            continue
        for f2, _, entry in fields(features):  # map<string, Feature> entries
            if f2 != 1:
                continue
            key, feature = None, b""
            for f3, _, x in fields(entry):
                if f3 == 1:
                    key = bytes(x).decode("utf-8")
                elif f3 == 2:
                    feature = x
            values: list = []
            for kind, _, lst in fields(feature):
                for f5, wire, x in fields(lst):
                    if f5 != 1:
                        continue
                    if kind == 1:  # BytesList
                        values.append(bytes(x))
                    elif kind == 3:  # Int64List
                        values += [_signed(q) for q in _packed(wire, x)]
                    elif kind == 2:  # FloatList
                        values += list(struct.unpack("<%df" % (len(x) // 4), bytes(x)))
            out[key] = values
    return out


def read_records(path: str) -> Iterator[bytes]:
    """The records of a TFRecord file, GZIP-compressed or plain: uint64 length, uint32 crc, data, uint32 crc (the CRCs are
    skipped: a damaged file shows up as a short read or a record that does not parse)."""
    with open(path, "rb") as probe:
        gz = probe.read(2) == b"\x1f\x8b"
    with (gzip.open(path, "rb") if gz else open(path, "rb")) as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) < 12:
                raise ValueError("truncated TFRecord header in %s" % path)
            (size,) = struct.unpack("<Q", head[:8])
            data = f.read(size)
            if len(data) != size or len(f.read(4)) != 4:
                raise ValueError("truncated TFRecord in %s" % path)
            yield data


# ---- accessibility tree ---------------------------------------------------------------------------------------------------


def _rect(buf) -> Tuple[int, int, int, int]:
    d = {1: 0, 2: 0, 3: 0, 4: 0}
    for f, _, v in fields(buf):
        if f in d:
            d[f] = _signed(v)
    return d[1], d[2], d[3], d[4]


def _text(v) -> str:
    return bytes(v).decode("utf-8", "replace")


def _node(buf) -> dict:
    n = {"id": 0, "bounds": (0, 0, 0, 0), "cls": "", "text": "", "desc": "", "hint": "", "clickable": False,
         "editable": False, "enabled": False, "visible": False, "child_ids": []}
    for f, wire, v in fields(buf):
        if f == 1:
            n["id"] = v
        elif f == 2:
            n["bounds"] = _rect(v)
        elif f == 3:
            n["cls"] = _text(v)
        elif f == 4:
            n["desc"] = _text(v)
        elif f == 5:
            n["hint"] = _text(v)
        elif f == 7:
            n["text"] = _text(v)
        elif f == 14:
            n["clickable"] = bool(v)
        elif f == 15:
            n["editable"] = bool(v)
        elif f == 16:
            n["enabled"] = bool(v)
        elif f == 23:
            n["visible"] = bool(v)
        elif f == 25:
            n["child_ids"] += _packed(wire, v)
    return n


def parse_forest(buf) -> List[dict]:
    """A serialized ``AndroidAccessibilityForest`` -> its nodes (all windows), each {bounds (left, top, right, bottom)
    in screen pixels, cls, text, desc, hint, clickable, editable, enabled, visible, kids, win, wtype, layer, wbounds}.
    Window types: 1 application, 2 input method (keyboard), 3 system."""
    nodes = []
    for w, (f, _, win) in enumerate(fields(memoryview(buf))):
        if f != 1:
            continue
        wtype = layer = 0
        wbounds = (0, 0, 0, 0)
        mine = []
        for f2, _, v in fields(win):
            if f2 == 1:
                wbounds = _rect(v)
            elif f2 == 4:
                layer = _signed(v)
            elif f2 == 6:
                wtype = v
            elif f2 == 11:
                mine += [_node(x) for f3, _, x in fields(v) if f3 == 1]
        by_id = {n["id"]: n for n in mine}
        for n in mine:
            n.update(win=w, wtype=wtype, layer=layer, wbounds=wbounds,
                     kids=[by_id[c] for c in n.pop("child_ids") if c in by_id and c != n["id"]])
        nodes += mine
    return nodes


def _area(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inside(b, x, y) -> bool:
    return b[0] <= x < b[2] and b[1] <= y < b[3]


def node_label(node: dict, limit: int = 60) -> str:
    """What the element says: its text, else its content description, else the first text / description among its
    descendants (a clickable row is labelled by the text views inside it), else its hint. "" when it says nothing."""
    queue, seen = [node], 0
    while queue and seen < 40:
        n = queue.pop(0)
        seen += 1
        for key in ("text", "desc"):
            s = re.sub(r"\s+", " ", n.get(key) or "").strip()
            if s:
                return s[:limit]
        queue += n.get("kids") or []
    return re.sub(r"\s+", " ", node.get("hint") or "").strip()[:limit]


def target_node(nodes: List[dict], x: int, y: int, clickable_only: bool = False) -> Optional[dict]:
    """The element a tap at (x, y) lands on: in the front-most window with a hit, the smallest visible *clickable* node
    containing the point; without one (and unless ``clickable_only``) the smallest node there that has a label."""
    hits = [n for n in nodes if n["visible"] and _area(n["bounds"]) > 0 and _inside(n["bounds"], x, y)]
    for want in (lambda n: n["clickable"],) + (() if clickable_only else (lambda n: bool(node_label(n)),)):
        for layer in sorted({n["layer"] for n in hits}, reverse=True):
            cand = [n for n in hits if n["layer"] == layer and want(n)]
            if cand:
                return min(cand, key=lambda n: _area(n["bounds"]))
    return None


def bucket(x: float, y: float, w: int, h: int) -> str:
    """A 3 x 3 screen position: "top left" ... "center" ... "bottom right"."""
    col = ("left", "center", "right")[min(2, max(0, int(3 * x / max(1, w))))]
    row = ("top", "middle", "bottom")[min(2, max(0, int(3 * y / max(1, h))))]
    return "center" if (row, col) == ("middle", "center") else "%s %s" % (row, col)


def _describe(node: Optional[dict]) -> str:
    if node is None:
        return "the screen"
    label = node_label(node)
    return '"%s"' % label if label else "the %s" % ((node["cls"] or "element").rsplit(".", 1)[-1] or "element")


def _center(b) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def render_node_click(node: dict, w: int, h: int, verb: str = "click") -> str:
    cx, cy = _center(node["bounds"])
    return "%s %s (%s)" % (verb, _describe(node), bucket(cx, cy, w, h))


def render_action(action: dict, nodes: List[dict], w: int, h: int) -> str:
    """An action as one option string: its type + the target element's text / description and screen position (the
    tap's position alone when the tree has no element there) or the typed text."""
    t = action.get("action_type")
    if t in ("click", "long_press"):
        verb = "click" if t == "click" else "long press"
        node = target_node(nodes, action["x"], action["y"])
        if node is None:
            return "%s at the %s of the screen" % (verb, bucket(action["x"], action["y"], w, h))
        return render_node_click(node, w, h, verb)
    if t == "scroll":
        return "scroll %s" % action.get("direction")
    if t == "input_text":
        return 'type "%s"' % re.sub(r"\s+", " ", str(action.get("text") or "")).strip()[:80]
    if t == "open_app":
        return 'open the app "%s"' % str(action.get("app_name") or "").strip()[:60]
    return {"navigate_back": "go back", "navigate_home": "go to the home screen", "wait": "wait for the screen to load"}.get(t, str(t))


def clickable_nodes(nodes: List[dict], w: int, h: int, min_side: int = 24, max_share: float = 0.4) -> List[dict]:
    """Visible clickable elements one could draw a box on: on screen, not tiny, not most of the screen, and not under
    a window in front of theirs (the keyboard, a dialog)."""
    windows = {(n["win"], n["layer"], n["wbounds"]) for n in nodes}
    out = []
    for n in nodes:
        b = n["bounds"]
        if not (n["visible"] and n["clickable"]) or b[0] < 0 or b[1] < 0 or b[2] > w or b[3] > h:
            continue
        if b[2] - b[0] < min_side or b[3] - b[1] < min_side or _area(b) > max_share * w * h:
            continue
        cx, cy = _center(b)
        if any(layer > n["layer"] and win != n["win"] and _inside(wb, cx, cy) for win, layer, wb in windows):
            continue
        out.append(n)
    return out


def _overlap(a, b) -> int:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


# ---- option construction ----------------------------------------------------------------------------------------------------

GENERIC = {"navigate_back": "go back", "navigate_home": "go to the home screen", "wait": "wait for the screen to load"}
#: multipliers on the gold type distribution for the type-sampled distractor slots: the later-step and screen-element
#: slots already supply clicks, and "go back" / "wait" have a single rendering that is often already in the history, so
#: without the boost such an option was the gold too often (first build: P(gold | "go back" option) = 0.40 vs 0.20)
SAMPLE_BOOST = {"click": 0.5, "long_press": 2.0, "scroll": 2.0, "navigate_back": 3.0, "navigate_home": 3.0, "wait": 3.0}
DIRECTIONS = ("up", "down", "left", "right")


def next_action_options(gold: str, history: List[str], later: List[str], screen: List[str], rng: random.Random,
                        type_p: Dict[str, float], pools: Dict[str, List[str]], k: Optional[int] = None):
    """(options, label) or None. ``gold`` and every distractor are rendered actions; ``history`` the rendered previous
    actions (none of them may be an option: a gold that repeats one is skipped); ``later`` the later actions of the
    episode; ``screen`` clicks on other elements of the current screen, in the order they are to be used
    (``hard_first``); ``pools`` typed texts and app names from other episodes of the split. Up to 2 distractors are later actions (for an open-app gold: other apps), at least 1 is
    another element of the screen when there is one, and the rest are drawn by action type with the gold actions' type
    distribution ``type_p``, so that an option's type says little about being the gold. The gold's position is uniform."""
    shown = set(history)
    if gold in shown:
        return None
    k = k or rng.randint(4, 6)
    opts = [gold]

    def add(o) -> bool:
        if o and o not in shown and o not in opts:
            opts.append(o)
            return True
        return False

    if gold.startswith("open the app"):  # which app, not "the only open-app option"
        for o in rng.sample(pools.get("open_app") or [], min(12, len(pools.get("open_app") or []))):
            if len(opts) >= 3:
                break
            add(o)
    for o in rng.sample(later, len(later)):
        if len(opts) >= 3:
            break
        add(o)
    for o in screen:
        if add(o):
            break
    types = sorted(type_p)
    weights = [type_p[t] * SAMPLE_BOOST.get(t, 1.0) for t in types]
    tries = 0
    while len(opts) < k and tries < 60:
        tries += 1
        t = rng.choices(types, weights)[0]
        if t in ("click", "long_press"):
            cand = screen[tries % len(screen)] if screen else None
            if cand and t == "long_press":
                cand = "long press" + cand[len("click"):]
        elif t == "scroll":
            cand = "scroll %s" % rng.choice(DIRECTIONS)
        elif t == "input_text":
            cand = rng.choice(pools["input_text"]) if pools.get("input_text") else None
        elif t == "open_app":
            cand = rng.choice(pools["open_app"]) if pools.get("open_app") else None
        else:
            cand = GENERIC.get(t)
        add(cand)
    if len(opts) < 4:
        return None
    order = rng.sample(range(len(opts)), len(opts))
    return [opts[i] for i in order], order.index(0)


def hard_first(options: List[str], goal: str, rng: random.Random, n_hard: int = 2) -> List[str]:
    """``options`` in random order, except that the ``n_hard`` sharing the most words with the goal come first (when
    they share any): matching option words against the goal alone must not find the gold."""
    words = lambda t: set(re.findall(r"[a-z0-9]{3,}", t.lower()))  # noqa: E731
    gw = words(goal)
    opts = rng.sample(options, len(options))
    ranked = sorted(opts, key=lambda o: -len(words(o.rsplit(" (", 1)[0]) & gw))
    hard = [o for o in ranked[:n_hard] if words(o.rsplit(" (", 1)[0]) & gw]
    return hard + [o for o in opts if o not in hard]


def element_candidates(nodes: List[dict], x: int, y: int, w: int, h: int, rng: random.Random, k: Optional[int] = None):
    """(boxes, gold index) or None: the gold box (the smallest clickable node containing the tap at (x, y)) and k - 1
    other clickable nodes, in a random order (the order gives the numbers 1..k drawn on the screenshot). Distractors do
    not touch the gold box or each other (no nested boxes); same window and class and the nearest ones first."""
    cands = clickable_nodes(nodes, w, h)
    gold = target_node(nodes, x, y, clickable_only=True)
    if gold is None or not any(gold is c for c in cands):
        return None
    k = k or rng.randint(4, 6)
    gx, gy = _center(gold["bounds"])
    others = [c for c in cands if c is not gold and not _inside(c["bounds"], x, y) and _overlap(c["bounds"], gold["bounds"]) == 0]

    def rank(c):
        cx, cy = _center(c["bounds"])
        return (c["win"] != gold["win"], c["cls"] != gold["cls"], (cx - gx) ** 2 + (cy - gy) ** 2)

    others.sort(key=rank)
    near = others[: 3 * (k - 1)]
    chosen = []
    for c in rng.sample(near, len(near)) + others[3 * (k - 1):]:
        if len(chosen) >= k - 1:
            break
        if all(_overlap(c["bounds"], o["bounds"]) == 0 for o in chosen):  # no nested or crossing boxes
            chosen.append(c)
    if len(chosen) < 3:
        return None
    boxes = [gold] + chosen
    order = rng.sample(range(len(boxes)), len(boxes))
    return [boxes[i]["bounds"] for i in order], order.index(0)


# ---- images -----------------------------------------------------------------------------------------------------------------

PALETTE = [(220, 20, 60), (0, 90, 230), (0, 140, 70), (235, 120, 0), (140, 30, 180), (0, 150, 160)]
_FONTS: dict = {}


def _font(size: int):
    if size not in _FONTS:
        from PIL import ImageFont

        try:
            _FONTS[size] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except OSError:
            _FONTS[size] = ImageFont.load_default(size=size)
    return _FONTS[size]


def load_screenshot(png: bytes, max_side: int = MAX_SIDE):
    """(RGB image with its long side <= max_side, scale applied)."""
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    scale = min(1.0, max_side / float(max(im.size)))
    if scale < 1.0:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    return im, scale


def draw_boxes(im, boxes: List[Tuple[int, int, int, int]], scale: float = 1.0, font_size: int = 20):
    """Draw box i (screen pixels, scaled by ``scale``) with the number i + 1: a coloured outline and a tag (white digit
    on the box colour, white edge) at a corner OUTSIDE the box where it covers neither another box nor another tag;
    when every outside position is taken, inside the box's top-left corner. Returns the tag rectangles."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(im)
    font = _font(font_size)
    W, H = im.size
    rects = [tuple(int(round(v * scale)) for v in b) for b in boxes]
    for i, r in enumerate(rects):
        colour = PALETTE[i % len(PALETTE)]
        d.rectangle([r[0] - 1, r[1] - 1, r[2], r[3]], outline=(255, 255, 255), width=1)
        d.rectangle([r[0], r[1], r[2] - 1, r[3] - 1], outline=colour, width=3)
    tags = []
    for i, r in enumerate(rects):
        text = str(i + 1)
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0] + 8, tb[3] - tb[1] + 8
        spots = [(r[0], r[1] - th), (r[2] - tw, r[1] - th), (r[0], r[3]), (r[2] - tw, r[3]), (r[0] - tw, r[1]),
                 (r[2], r[1]), (r[0] - tw, r[3] - th), (r[2], r[3] - th)]
        best = None
        for x0, y0 in spots:
            t = (x0, y0, x0 + tw, y0 + th)
            if x0 < 0 or y0 < 0 or t[2] > W or t[3] > H:
                continue
            if any(_overlap(t, o) for j, o in enumerate(rects) if j != i) or any(_overlap(t, o) for o in tags):
                continue
            best = t
            break
        if best is None:  # inside the top-left corner, kept on the image
            x0, y0 = min(max(0, r[0]), W - tw), min(max(0, r[1]), H - th)
            best = (x0, y0, x0 + tw, y0 + th)
        tags.append(best)
        colour = PALETTE[i % len(PALETTE)]
        d.rectangle([best[0], best[1], best[2] - 1, best[3] - 1], fill=colour, outline=(255, 255, 255), width=1)
        d.text((best[0] + 4 - tb[0], best[1] + 4 - tb[1]), text, fill=(255, 255, 255), font=font)
    return tags


def _dhash_of_file(path: str):
    dv = _validator()
    from PIL import Image

    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with Image.open(path) as im:
        return sha, dv.dhash(im)


def _validator():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    import data_validator

    return data_validator


# ---- episodes -> candidate rows ------------------------------------------------------------------------------------------------


def episode_of(ex: Dict[str, list]) -> dict:
    """The text part of an episode record (an episode with a single screenshot has no ``actions`` feature at all)."""
    return {"episode_id": int(ex["episode_id"][0]), "goal": (ex.get("goal") or [b""])[0].decode("utf-8"),
            "actions": [json.loads(a) for a in ex.get("actions", [])],
            "instructions": [s.decode("utf-8").strip() for s in ex.get("step_instructions", [])],
            "widths": list(ex.get("screenshot_widths", [])), "heights": list(ex.get("screenshot_heights", []))}


def truncation_steps(n_actions: int) -> List[int]:
    """The screenshots of an episode with ``n_actions`` actions that lie at 30-80% of it (screenshot t is the screen
    after t actions; the last one, ``n_actions``, is the completed episode)."""
    return [t for t in range(1, n_actions) if 0.3 <= t / float(n_actions) <= 0.8]


def _scan_shard(path: str) -> List[dict]:
    out = []
    for rec in read_records(path):
        ex = parse_example(rec)
        ep = episode_of({k: v for k, v in ex.items() if k not in ("screenshots", "accessibility_trees")})
        out.append(ep)
    return out


def _save(im, path: str):
    im.save(path, "JPEG", quality=88)
    return _dhash_of_file(path)


def _process_shard(job) -> dict:
    """All candidate rows of one shard for the three sets; images are written under ``tmp[name]/images``."""
    path, split_of, per_episode, type_p, pools, tmp, seed = job
    out = {n: [] for n in NAMES}
    stats = Counter()
    for rec in read_records(path):
        ex = parse_example(rec)
        ep = episode_of(ex)
        eid = ep["episode_id"]
        split = split_of.get(eid)
        if split is None:
            stats["episodes_not_in_splits"] += 1
            continue
        rng = random.Random("ac-%d-%d" % (seed, eid))
        n_act = len(ep["actions"])
        if not n_act:
            stats["episodes_without_actions"] += 1
            continue
        if len(ex["screenshots"]) != n_act + 1 or len(ex["accessibility_trees"]) != n_act + 1 or not ep["goal"].strip() \
                or len(ep["instructions"]) != n_act or len(ep["widths"]) != n_act + 1:
            stats["episodes_malformed"] += 1
            continue
        stats["episodes_" + split] += 1
        forests: dict = {}

        def nodes_at(i):
            if i not in forests:
                forests[i] = parse_forest(ex["accessibility_trees"][i])
            return forests[i]

        def screen_image(i):
            return load_screenshot(ex["screenshots"][i])

        size = lambda i: (ep["widths"][i], ep["heights"][i])  # noqa: E731
        rendered = [render_action(a, nodes_at(i) if a["action_type"] in ("click", "long_press") else [], *size(i))
                    for i, a in enumerate(ep["actions"])]
        base = {"episode": eid, "image_license": IMAGE_LICENSE}

        # -- next action (step 0, almost always "open the app named in the goal", is used for at most ~30% of episodes)
        steps = rng.sample(range(n_act), n_act)
        if n_act > 1 and rng.random() < 0.7:
            steps.remove(0)
            steps.append(0)
        made = 0
        for t in steps:
            if made >= per_episode[split]:
                break
            w, h = size(t)
            gold_node = target_node(nodes_at(t), ep["actions"][t]["x"], ep["actions"][t]["y"]) \
                if ep["actions"][t]["action_type"] in ("click", "long_press") else None
            screen = [render_node_click(n, w, h) for n in clickable_nodes(nodes_at(t), w, h)
                      if n is not gold_node and n["wtype"] != 2 and node_label(n)]
            later = list(dict.fromkeys(rendered[t + 1:]))
            typed = [r for r in later if r.startswith("type ")]
            got = next_action_options(rendered[t], rendered[:t], later, hard_first(list(dict.fromkeys(screen)), ep["goal"], rng),
                                      rng, type_p,
                                      dict(pools[split], input_text=typed or pools[split]["input_text"]))
            if got is None:
                stats["next_skip_" + ("gold_in_history" if rendered[t] in rendered[:t] else "few_options")] += 1
                continue
            opts, label = got
            im, _ = screen_image(t)
            name = "images/ep%05d_s%02d.jpg" % (eid, t)
            sig = _save(im, os.path.join(tmp["ac_next_action"], name))
            state = {"goal": ep["goal"], "previous_actions": rendered[:t][-15:]}
            out["ac_next_action"].append(dict(base, _split=split, _sig=sig, image=name, step=t,
                                              state_text=json.dumps(state, ensure_ascii=False),
                                              question={"type": "choice", "instructions": Q_NEXT, "criteria": opts},
                                              label=label, kind=ep["actions"][t]["action_type"]))
            made += 1

        # -- element choice
        made = 0
        for t in rng.sample(range(n_act), n_act):
            if made >= per_episode[split]:
                break
            a = ep["actions"][t]
            ins = ep["instructions"][t]
            if a["action_type"] not in ("click", "long_press") or not ins:
                continue
            if (t and ep["instructions"][t - 1] == ins) or (t + 1 < n_act and ep["instructions"][t + 1] == ins):
                stats["element_skip_instruction_spans_actions"] += 1
                continue
            w, h = size(t)
            got = element_candidates(nodes_at(t), a["x"], a["y"], w, h, rng)
            if got is None:
                stats["element_skip_no_gold_or_few_boxes"] += 1
                continue
            boxes, label = got
            im, scale = screen_image(t)
            draw_boxes(im, boxes, scale)
            name = "images/ep%05d_s%02d_boxes.jpg" % (eid, t)
            sig = _save(im, os.path.join(tmp["ac_element"], name))
            out["ac_element"].append(dict(base, _split=split, _sig=sig, image=name, step=t,
                                          state_text=json.dumps({"instruction": ins}, ensure_ascii=False),
                                          question={"type": "choice", "instructions": Q_ELEMENT,
                                                    "criteria": ["box %d" % (i + 1) for i in range(len(boxes))]},
                                          label=label))
            made += 1

        # -- goal reached: the final screen (yes) and the same episode cut at 30-80% (no)
        cuts = truncation_steps(n_act)
        if cuts:
            final, _ = screen_image(n_act)
            dv = _validator()
            fh = dv.dhash(final)
            for t in rng.sample(cuts, len(cuts)):
                im, _ = screen_image(t)
                th = dv.dhash(im)
                if fh is not None and th is not None and bin(fh ^ th).count("1") <= 6:
                    stats["goal_skip_cut_screen_looks_like_final"] += 1
                    continue  # the cut screen is (nearly) the final one: not a "no"
                state = json.dumps({"goal": ep["goal"]}, ensure_ascii=False)
                for step, image, label in ((n_act, final, 1), (t, im, 0)):
                    name = "images/ep%05d_s%02d.jpg" % (eid, step)
                    sig = _save(image, os.path.join(tmp["ac_goal_reached"], name))
                    out["ac_goal_reached"].append(dict(base, _split=split, _sig=sig, image=name, step=step, state_text=state,
                                                       question={"type": "noul", "instructions": Q_GOAL, "criteria": None},
                                                       label=label))
                break
            else:
                stats["goal_skip_no_distinct_cut"] += 1
        else:
            stats["goal_skip_too_short"] += 1
    return {"rows": out, "stats": stats}


# ---- build ----------------------------------------------------------------------------------------------------------------


def load_splits(raw: str = RAW) -> Dict[int, str]:
    """episode id -> "train" | "val" from the official splits.json (val = validation + test episodes)."""
    with open(os.path.join(raw, "splits.json")) as f:
        s = json.load(f)
    train, held = set(s["train"]), set(s["validation"]) | set(s["test"])
    if train & held:
        raise ValueError("splits.json: %d episodes are both train and validation/test" % len(train & held))
    out = {int(e): "train" for e in train}
    out.update({int(e): "val" for e in held})
    return out


def drop_leaky_val(rows: Dict[str, List[dict]], image_path, log=print, paired=("ac_goal_reached",)) -> Dict[str, int]:
    """Remove val rows (of any of the sets in ``rows``) whose image is also a train image of any of them: same bytes, or
    dHash within the validator's distance confirmed by its thumbnail correlation. ``rows``: name -> candidate rows with
    ``_split`` and ``_sig`` = (sha256, dhash); ``image_path(name, row)`` -> file. In a ``paired`` set an episode loses
    all its rows when one goes (its yes and no rows stay together)."""
    dv = _validator()
    from PIL import Image

    train_sha, train_dh = set(), defaultdict(list)
    for name, rs in rows.items():
        for r in rs:
            if r["_split"] == "train":
                train_sha.add(r["_sig"][0])
                if r["_sig"][1] is not None:
                    train_dh[r["_sig"][1]].append(image_path(name, r))
    thumbs: dict = {}

    def thumb(p):
        if p not in thumbs:
            with Image.open(p) as im:
                thumbs[p] = dv.thumbnail(im)
        return thumbs[p]

    val_dh = defaultdict(list)
    for name, rs in rows.items():
        for r in rs:
            if r["_split"] == "val" and r["_sig"][1] is not None:
                val_dh[r["_sig"][1]].append((name, r))
    near = defaultdict(list)
    for hv, ht in dv._near_pairs({k: 0 for k in val_dh}, {k: 0 for k in train_dh}, dv.DHASH_MAX_DISTANCE):
        near[hv].append(ht)
    leaky = set()
    for hv, hts in near.items():
        for name, r in val_dh[hv]:
            p = image_path(name, r)
            if any(dv.correlation(thumb(p), thumb(q)) >= dv.MIN_THUMB_CORRELATION for ht in hts for q in train_dh[ht]):
                leaky.add(p)
    dropped = Counter()
    for name in rows:
        bad = lambda r: r["_split"] == "val" and (r["_sig"][0] in train_sha or image_path(name, r) in leaky)  # noqa: E731
        gone = {r["episode"] for r in rows[name] if bad(r)} if name in paired else set()
        keep = [r for r in rows[name] if not bad(r) and r["episode"] not in gone]
        dropped[name] = len(rows[name]) - len(keep)
        rows[name] = keep
    log("val rows dropped for showing a train screenshot: %s" % dict(dropped))
    return dict(dropped)


def _cap(rows: List[dict], cap: int, rng: random.Random, paired: bool) -> List[dict]:
    """At most ``cap`` rows, episodes spread evenly (one row per episode first); ``paired`` keeps an episode's yes and
    no rows together."""
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["episode"]].append(r)
    eps = sorted(by_ep)
    rng.shuffle(eps)
    out = []
    if paired:
        for e in eps:
            if len(out) + len(by_ep[e]) <= cap:
                out += by_ep[e]
        return out
    depth = 0
    while len(out) < cap and any(len(by_ep[e]) > depth for e in eps):
        out += [by_ep[e][depth] for e in eps if len(by_ep[e]) > depth][: cap - len(out)]
        depth += 1
    return out


def build(root: str = DATA_ROOT, raw: str = RAW, shards: Optional[int] = None, workers: int = 8, seed: int = 0,
          train_cap: int = TRAIN_CAP, val_cap: int = VAL_CAP, log=print) -> Dict[str, dict]:
    from multiprocessing import Pool

    paths = [os.path.join(raw, SHARD % (i, N_SHARDS)) for i in range(N_SHARDS)]
    paths = [p for p in paths if os.path.exists(p)][: shards or N_SHARDS]
    if not paths:
        raise FileNotFoundError("no AndroidControl shards under %s (run the download command)" % raw)
    split_of = load_splits(raw)
    t0 = time.time()
    with Pool(min(workers, len(paths))) as pool:
        scanned = [ep for part in pool.map(_scan_shard, paths) for ep in part]
    log("scanned %d episodes in %d shards (%.0f s)" % (len(scanned), len(paths), time.time() - t0))
    # the types of the gold actions and, per split, the typed texts and app names other episodes use
    types = Counter()
    for ep in scanned:
        if split_of.get(ep["episode_id"]) == "train":
            seen = set()
            for a in ep["actions"]:
                key = json.dumps(a, sort_keys=True)
                types[a["action_type"]] += key not in seen  # a repeated action (scroll, wait) is never a gold
                seen.add(key)
    type_p = {t: c / float(sum(types.values())) for t, c in types.items()}
    pools = {s: {"input_text": [], "open_app": []} for s in ("train", "val")}
    for ep in scanned:
        s = split_of.get(ep["episode_id"])
        if s:
            for a in ep["actions"]:
                if a["action_type"] in ("input_text", "open_app"):
                    pools[s][a["action_type"]].append(render_action(a, [], 0, 0))
    pools = {s: {k: sorted(set(v)) for k, v in p.items()} for s, p in pools.items()}
    tmp = {n: os.path.join(root, ".tmp-" + n) for n in NAMES}
    for d in tmp.values():
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(os.path.join(d, "images"))
    jobs = [(p, split_of, {"train": 2, "val": 1}, type_p, pools, tmp, seed) for p in paths]
    rows = {n: [] for n in NAMES}
    stats = Counter()
    with Pool(min(workers, len(paths))) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_shard, jobs)):
            for n in NAMES:
                rows[n] += res["rows"][n]
            stats.update(res["stats"])
            log("shard %d/%d done (%.0f s): %s" % (i + 1, len(paths), time.time() - t0, {n: len(rows[n]) for n in NAMES}))
    for n in NAMES:
        rows[n].sort(key=lambda r: (r["episode"], r["step"], r["label"]))
    candidates = {n: dict(Counter(r["_split"] for r in rows[n])) for n in NAMES}
    dropped = drop_leaky_val(rows, lambda n, r: os.path.join(tmp[n], r["image"]), log)
    manifests = {}
    for n in NAMES:
        rng = random.Random("%s-%d" % (n, seed))
        out = {s: _cap([r for r in rows[n] if r["_split"] == s], cap, rng, paired=n == "ac_goal_reached")
               for s, cap in (("train", train_cap), ("val", val_cap))}
        used = set()
        labels = {}
        for s, rs in out.items():
            rng.shuffle(rs)
            with open(os.path.join(tmp[n], s + ".jsonl"), "w") as f:
                for i, r in enumerate(rs):
                    rec = {"id": "%s-%s-%d" % (n, s, i)}
                    rec.update({k: v for k, v in r.items() if not k.startswith("_")})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    used.add(r["image"])
            labels[s] = dict(Counter("%s:%d" % (r["question"]["type"], r["label"]) for r in rs))
        for fn in os.listdir(os.path.join(tmp[n], "images")):  # candidates cut by the caps or the leak filter
            if "images/" + fn not in used:
                os.remove(os.path.join(tmp[n], "images", fn))
        m = {"dataset": n, "source": "gs://gresearch/android_control (%d of %d shards)" % (len(paths), N_SHARDS),
             "revision": "bucket objects as of 2026-09-30", "license": LICENSE, "seed": seed,
             "split": "official splits.json: train -> train; validation + test -> val (episode-disjoint)",
             "labels": {"ac_next_action": "the human rater's next action", "ac_element": "the clickable node containing the human tap",
                        "ac_goal_reached": "yes = final screenshot of the complete human episode; no = the same episode cut "
                                           "at 30-80% of its steps (by construction, not a human judgement)"}[n],
             "counts": {s: len(rs) for s, rs in out.items()}, "label_counts": labels, "candidates": candidates[n],
             "episodes": {s: len({r["episode"] for r in rs}) for s, rs in out.items()},
             "val_rows_dropped_for_train_screenshot": dropped.get(n, 0), "build_stats": dict(stats),
             "image_max_side": MAX_SIDE, "validator": {"noul_min_minority": 0.2}}
        with open(os.path.join(tmp[n], "manifest.json"), "w") as f:
            json.dump(m, f, indent=2)
        final = os.path.join(root, n)
        if os.path.exists(final):
            shutil.rmtree(final)
        os.rename(tmp[n], final)
        with open(os.path.join(final, "_READY"), "w") as f:
            json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
        manifests[n] = m
        log("%s: %s %s" % (n, m["counts"], labels))
    log("build stats: %s" % dict(stats))
    return manifests


# ---- audit: what a model that never sees the screenshot can get ---------------------------------------------------------------


def _option_type(option: str) -> str:
    for prefix, t in (("click", "click"), ("long press", "long_press"), ("scroll", "scroll"), ("type ", "input_text"),
                      ("open the app", "open_app"), ("go back", "navigate_back"), ("go to the home", "navigate_home"),
                      ("wait", "wait")):
        if option.startswith(prefix):
            return t
    return "other"


def _blind_features(state: dict, option: str, dim: int) -> List[int]:
    import zlib

    words = lambda t: re.findall(r"[a-z0-9]+", t.lower())  # noqa: E731
    goal, prev = set(words(state.get("goal", ""))), state.get("previous_actions") or []
    t = _option_type(option)
    label = re.search(r'"(.*)"', option)
    lw = words(label.group(1)) if label else []
    hit = sum(w in goal for w in lw)
    where = re.search(r"\(([a-z ]+)\)$", option)
    feats = ["bias", "type=" + t, "type=%s|hist=%d" % (t, min(len(prev), 3)), "type=%s|prev=%s" % (t, _option_type(prev[-1]) if prev else "none"),
             "where=" + (where.group(1) if where else "-"), "type=%s|ingoal=%d" % (t, min(hit, 3)),
             "allingoal=%d" % int(bool(lw) and hit == len(lw)), "phrase=%d" % int(bool(label) and label.group(1).lower() in state.get("goal", "").lower()),
             "prevoverlap=%d" % min(2, len(set(lw) & set(words(prev[-1]))) if prev else 0)]
    feats += ["tok=" + w for w in set(lw)]
    return [zlib.crc32(f.encode()) % dim for f in feats]


def blind_baseline(name: str = "ac_next_action", root: str = DATA_ROOT, epochs: int = 6, dim: int = 1 << 15, seed: int = 0) -> dict:
    """The question-only baseline for a choice set: a linear scorer over TEXT features of each option (its action type,
    alone and crossed with the history length and the previous action's type; its screen position; how many of its
    label words are in the goal; its words), trained on train and scored on val. It never sees the screenshot, so what it
    gets above chance is what the goal, the history and the options give away."""
    import math

    rows = {s: [json.loads(l) for l in open(os.path.join(root, name, s + ".jsonl"))] for s in ("train", "val")}
    feats = {s: [[_blind_features(json.loads(r["state_text"]), o, dim) for o in r["question"]["criteria"]] for r in rs]
             for s, rs in rows.items()}
    w = [0.0] * dim
    rng = random.Random(seed)
    order = list(range(len(rows["train"])))

    def probs(options):
        z = [sum(w[i] for i in f) for f in options]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        t = sum(e)
        return [v / t for v in e]

    for epoch in range(epochs):
        rng.shuffle(order)
        lr = 0.2 / (1 + epoch)
        for k in order:
            p = probs(feats["train"][k])
            for j, f in enumerate(feats["train"][k]):
                g = p[j] - (1.0 if j == rows["train"][k]["label"] else 0.0)
                for i in f:
                    w[i] -= lr * g
    out = {}
    for s in ("train", "val"):
        hit = sum(max(range(len(f)), key=lambda j: sum(w[i] for i in f[j])) == r["label"] for f, r in zip(feats[s], rows[s]))
        out[s] = round(hit / float(len(rows[s])), 3)
    out["chance"] = round(sum(1.0 / len(r["question"]["criteria"]) for r in rows["val"]) / len(rows["val"]), 3)
    g, d = Counter(), Counter()
    for r in rows["train"]:
        for j, o in enumerate(r["question"]["criteria"]):
            (g if j == r["label"] else d)[_option_type(o)] += 1
    out["p_gold_given_option_type_train"] = {t: round(g[t] / float(g[t] + d[t]), 3) for t in sorted(set(g) | set(d))}
    first = [r for r in rows["val"] if not json.loads(r["state_text"]).get("previous_actions")]
    out["val_rows_with_empty_history"] = len(first)
    return out


def download(raw: str = RAW, shards: int = N_SHARDS, log=print):
    """Fetch splits.json and the first ``shards`` TFRecord shards over HTTPS (resumes a partial file)."""
    import subprocess

    os.makedirs(raw, exist_ok=True)
    for name in ["splits.json", "test_subsplits.json"] + [SHARD % (i, N_SHARDS) for i in range(shards)]:
        subprocess.run(["curl", "-s", "-C", "-", "--retry", "5", "-o", os.path.join(raw, name), BUCKET + name], check=True)
        log("fetched %s" % name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "build", "audit"])
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--shards", type=int, default=None, help="use only the first N shards")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.command == "download":
        download(a.raw, a.shards or N_SHARDS)
    elif a.command == "audit":
        print(json.dumps({"ac_next_action question-only (no screenshot)": blind_baseline("ac_next_action", a.root)}, indent=1))
    else:
        build(a.root, a.raw, a.shards, a.workers, log=lambda m: print(m, flush=True))
