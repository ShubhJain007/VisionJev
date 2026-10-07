"""Human-labelled screen / UI sources as typed decisions on a screenshot (V6.3 section 5.1).

Sources (licences re-read on the primary source on 2026-09-30, pinned in ``SOURCES`` and copied into every manifest):

* UIBert RefExp      github.com/google-research-datasets/uibert (CC BY 4.0): crowd-written referring expressions
                     for one of the detected elements of a Rico screen (candidates + target index in the TFRecords).
* Widget Captioning  github.com/google-research-datasets/widget-caption (CC BY 4.0): 2-4 human captions per widget
                     of a Rico screen (``datasetId == rc`` only; the ``nr`` screens were never released).
* screen_annotation  github.com/google-research-datasets/screen_annotation (CC BY 4.0): element type + box + text
                     of Rico screens, produced automatically, then verified and corrected by human raters.
* OmniACT            huggingface.co/datasets/Writer/omniact (MIT): tasks and PyAutoGUI scripts written by student
                     annotators for desktop / web screens whose elements were boxed and named by hand.
* Rico screenshots   interactionmining.org/rico ``unique_uis.tar.gz`` (no CC licence: the site's copyright notice
                     says the screenshots "may contain copyrighted work" and binds commercial users to its terms;
                     same caveat as ``rico_screenqa``, PROBLEMS D9).

Datasets (``<root>/<name>/{train,val}.jsonl``, images max side 1024, JPEG):

* ``ui_refexp``          (choice, 4-6) numbered boxes on candidate elements + a referring expression -> which box.
                         Distractors are the targets of OTHER expressions about the same screen (so the picture
                         alone cannot tell which box is meant); where a screen has fewer than three, given candidates
                         that look like targets fill the set to four (same view class first; ``kind`` says which).
* ``ui_widget_caption``  (choice, 4) one widget outlined in red + four human captions -> which describes it.
                         Distractors: captions of other widgets of the same screen, else the same app.
* ``ui_widget_find``     (choice, 4-5) a human caption + numbered boxes on captioned widgets of the screen -> which.
                         The caption shown is the widget's most specific one; widgets whose captions only say what
                         kind of widget it is ("toggle setting on/off") are skipped (``specific_caption``).
* ``ui_element_type``    (choice, 4) one element outlined in red -> its type. The option SET is drawn first (a class
                         and classes confusable with it), then the gold uniformly from the set, then an element of
                         that class: the class prior cannot pick the answer.
* ``ui_element_count``   (choice, 4) how many elements of type X are on the screen (0-12). Per type, the option set is
                         drawn first (near values), the gold uniformly from it, then a screen with that count: neither
                         the rank nor the value prior picks the answer. Types: checkboxes, radio buttons, switches,
                         list items (the ones annotated completely; see ``COUNT_NAMES``).
* ``omniact_next_action`` (choice, 4) screenshot + task -> which PyAutoGUI script. Options come in symmetric designs
                         (four targets; two targets x two keys; two tasks x two targets) so the gold is never the
                         odd one out or the "centre" of the option set.

Splits: each source's train -> train and dev / valid -> val; every source's TEST split is left untouched. Rico screens
repeat across sources and as near-duplicates, so every Rico screen gets ONE side from the screens it matches under the
data validator's matcher (with a margin: dHash <= 9 bits, thumbnail correlation >= 0.97): a screen that is, or matches,
a test screen of any source is unused; a train screen that is, or matches, a dev screen of any source or a
``rico_screenqa`` val screen is unused; a dev screen that matches a ``rico_screenqa`` train screen is unused
(``assign_sides``). OmniACT's own splits share every screen, so whole apps are moved to val (``omniact_val_apps``):
train rows are source-train tasks of train apps, val rows source-val tasks of val apps.

    python -m laya.visionjev_uihuman fetch                  # ~7 GB into <raw>/{rico,uibert_refexp,...}
    python -m laya.visionjev_uihuman build all --workers 6         # builds, then cross-dedupes against each other
    python -m laya.visionjev_uihuman cross-dedupe ui_refexp        # (+ rico_screenqa) with the validator's matcher
    python -m laya.visionjev_uihuman text-only ui_element_count
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import struct
import time
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .visionjev_data import DATA_ROOT, RAW_ROOT

MAX_SIDE = 1024
TRAIN_CAP, VAL_CAP = 15000, 1000
VERIFIED = "2026-09-30"
RICO_IMAGES = ("Rico screenshots (interactionmining.org/rico, unique_uis.tar.gz). No CC licence on the screenshots: "
               "the Rico copyright notice says they may contain copyrighted work and binds commercial users to its "
               "terms. VERIFY before release (PROBLEMS D9)")
SOURCES = {
    "uibert": {"url": "https://github.com/google-research-datasets/uibert",
               "raw": "https://raw.githubusercontent.com/google-research-datasets/uibert",
               "revision": "eceafbde7567959899bfb7f8d9e6fd71a52aaa12", "dir": "uibert_refexp",
               "files": ["ref_exp/train.tfrecord", "ref_exp/dev.tfrecord", "ref_exp/test.tfrecord", "README.md"],
               "license": "CC BY 4.0", "verified": "README 'License' section at the pinned commit, " + VERIFIED,
               "labels": "referring expressions collected from human raters (UIBert, arXiv 2107.13731)",
               "attribution": "Bai et al., UIBert: Learning Generic Multimodal Representations for UI Understanding "
                              "(2021), RefExp dataset, CC BY 4.0"},
    "widget_caption": {"url": "https://github.com/google-research-datasets/widget-caption",
                       "raw": "https://raw.githubusercontent.com/google-research-datasets/widget-caption",
                       "revision": "c65047cfa210968a0bc836f1f202709d7628ce80", "dir": "widget_caption",
                       "files": ["widget_captions.csv", "split/train.txt", "split/dev.txt", "split/test.txt", "README.md"],
                       "license": "CC BY 4.0", "verified": "README 'License' line at the pinned commit, " + VERIFIED,
                       "labels": "human captions from crowd workers (Widget Captioning, EMNLP 2020)",
                       "attribution": "Li et al., Widget Captioning: Generating Natural Language Description for "
                                      "Mobile User Interface Elements (EMNLP 2020), CC BY 4.0"},
    "screen_annotation": {"url": "https://github.com/google-research-datasets/screen_annotation",
                          "raw": "https://raw.githubusercontent.com/google-research-datasets/screen_annotation",
                          "revision": "053a4a1c4737969def99fd36084d7adf3715798f", "dir": "screen_annotation",
                          "files": ["train.csv", "valid.csv", "test.csv", "README.md"],
                          "license": "CC BY 4.0", "verified": "README 'License' section at the pinned commit, " + VERIFIED,
                          "labels": "produced with automated techniques, then verified and corrected by human raters "
                                    "(README; ScreenAI, arXiv 2402.04615)",
                          "attribution": "Baechler et al., ScreenAI: A Vision-Language Model for UI and Infographics "
                                         "Understanding (2024), Screen Annotation dataset, CC BY 4.0"},
    "omniact": {"url": "https://huggingface.co/datasets/Writer/omniact", "revision": "5e4bcb26745afc8d58ba9519ea9c9fb832386d26",
                "dir": "omniact", "files": ["train.json", "val.json", "test.json", "data.zip", "README.md"],
                "license": "MIT", "verified": "Hub card 'license: mit' at the pinned revision, " + VERIFIED,
                "labels": "tasks and PyAutoGUI scripts written by student annotators, element boxes drawn and named by "
                          "hand (OmniACT, arXiv 2402.17553 section 3); the ocr / detact_* files (automatic) are not used",
                "attribution": "Kapoor et al., OmniACT: A Dataset and Benchmark for Enabling Multimodal Generalist "
                               "Autonomous Agents for Desktop and Web (2024), MIT"},
    "rico": {"url": "https://www.interactionmining.org/rico",
             "base": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/",
             "revision": "unique_uis.tar.gz md5 bdcf99d4dc7f36a3396e57e306c75696 (last-modified 2019-03-08)", "dir": "rico",
             "files": ["unique_uis.tar.gz", "ui_details.csv"], "license": RICO_IMAGES,
             "attribution": "Deka et al., Rico: A Mobile App Dataset for Building Data-Driven Design Applications (UIST 2017)"},
}


# ---- reading TFRecords without TensorFlow ---------------------------------------------------------------------------

def _varint(buf, i: int) -> Tuple[int, int]:
    out = shift = 0
    while True:
        c = buf[i]
        i += 1
        out |= (c & 0x7F) << shift
        shift += 7
        if not c & 0x80:
            return out, i


def _fields(buf) -> Iterator[Tuple[int, int, object]]:
    """(field number, wire type, value) of one protobuf message; length-delimited values stay bytes."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        wire = key & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 2:
            n, i = _varint(buf, i)
            val, i = buf[i:i + n], i + n
        elif wire == 5:
            val, i = buf[i:i + 4], i + 4
        elif wire == 1:
            val, i = buf[i:i + 8], i + 8
        else:
            raise ValueError("unsupported wire type %d" % wire)
        yield key >> 3, wire, val


def parse_example(buf) -> Dict[str, list]:
    """A serialized ``tf.train.Example`` -> {feature name: list of bytes / float / int}."""
    out = {}
    for f, _, feats in _fields(buf):
        if f != 1:
            continue
        for _, _, entry in _fields(feats):
            key, val = None, b""
            for f3, _, v in _fields(entry):
                if f3 == 1:
                    key = v.decode()
                elif f3 == 2:
                    val = v
            vals = []
            for kind, _, lst in _fields(val):  # 1 bytes_list, 2 float_list, 3 int64_list
                for _, wire, v in _fields(lst):
                    if kind == 1:
                        vals.append(bytes(v))
                    elif kind == 2:
                        vals += struct.unpack("<%df" % (len(v) // 4), v)
                    elif wire == 2:
                        j = 0
                        while j < len(v):
                            x, j = _varint(v, j)
                            vals.append(x)
                    else:
                        vals.append(v)
            out[key] = vals
    return out


def read_tfrecord(path: str) -> Iterator[Dict[str, list]]:
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            data = f.read(struct.unpack("<Q", head[:8])[0])
            f.read(4)
            yield parse_example(data)


# ---- geometry (boxes are (x0, y0, x1, y1), normalised to [0, 1] unless said otherwise) ------------------------------

Box = Tuple[float, float, float, float]


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: Box, b: Box) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def touches(a: Box, b: Box, share: float = 0.2) -> bool:
    """The boxes overlap by more than ``share`` of the smaller one (nested boxes always do)."""
    m = min(_area(a), _area(b))
    return m <= 0 or _inter(a, b) / m > share


def box_ok(b: Box, min_w: float = 0.02, min_h: float = 0.012, max_area: float = 0.6) -> bool:
    """Inside the screen, large enough to see at 576 x 1024 (>= ~12 px) and not most of the screen."""
    return (-0.01 <= b[0] < b[2] <= 1.01 and -0.01 <= b[1] < b[3] <= 1.01 and b[2] - b[0] >= min_w
            and b[3] - b[1] >= min_h and _area(b) <= max_area)


def _center(b: Box) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


# ---- drawing --------------------------------------------------------------------------------------------------------

#: one colour per number: saturated and dark enough for white digits; each box also gets a white outer line
PALETTE = [(226, 28, 36), (20, 90, 255), (0, 135, 62), (190, 0, 190), (222, 98, 0), (104, 56, 222)]
HIGHLIGHT = (226, 28, 36)
_FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf")


def _font(size: int):
    from PIL import ImageFont

    for p in _FONTS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rects_touch(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def tag_position(box, size, tag, taken, pad: int = 4) -> Tuple[Tuple[int, int, int, int], bool]:
    """Where the number tag of ``box`` goes (pixel rects): the first spot OUTSIDE the box that lies inside the image
    and touches no other box or tag (``taken``); otherwise the top-left corner inside the box. -> (rect, outside)."""
    x0, y0, x1, y1 = box
    tw, th = tag
    W, H = size
    spots = [(x0 - pad, y0 - pad - th), (x1 + pad - tw, y0 - pad - th),          # above: left, right
             (x0 - pad - tw, y0 - pad), (x1 + pad, y0 - pad),                   # beside the top: left, right
             (x0 - pad, y1 + pad), (x1 + pad - tw, y1 + pad),                   # below: left, right
             (x0 - pad - tw, y1 + pad - th), (x1 + pad, y1 + pad - th)]         # beside the bottom: left, right
    for sx, sy in spots:
        r = (int(sx), int(sy), int(sx) + tw, int(sy) + th)
        if r[0] >= 0 and r[1] >= 0 and r[2] <= W and r[3] <= H and not any(_rects_touch(r, t) for t in taken):
            return r, True
    sx, sy = min(max(0, int(x0)), max(0, W - tw)), min(max(0, int(y0)), max(0, H - th))
    return (sx, sy, sx + tw, sy + th), False


def draw_boxes(im, boxes: Sequence[Box], numbered: bool = True, width: Optional[int] = None):
    """Outline ``boxes`` (normalised) on a copy of ``im``. ``numbered``: box i gets colour i and the tag ``i + 1``
    (white digits on the box colour, white rim), outside the box when a free spot exists. Not numbered: one red
    outline per box. Every outline is a coloured line with a white line around it, drawn just OUTSIDE the element so
    nothing of it is covered. -> (image, [tag rect or None], [tag is outside])."""
    from PIL import ImageDraw

    im = im.convert("RGB").copy()
    W, H = im.size
    scale = max(W, H) / 1024.0
    lw = width or max(2, round(3 * scale))
    d = ImageDraw.Draw(im)
    px = []
    for b in boxes:
        x0, y0 = max(0, round(b[0] * W) - lw - 1), max(0, round(b[1] * H) - lw - 1)
        x1, y1 = min(W - 1, round(b[2] * W) + lw), min(H - 1, round(b[3] * H) + lw)
        px.append((x0, y0, x1, y1))
    for i, (x0, y0, x1, y1) in enumerate(px):
        col = PALETTE[i % len(PALETTE)] if numbered else HIGHLIGHT
        d.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), outline=(255, 255, 255), width=1)
        d.rectangle((x0, y0, x1, y1), outline=col, width=lw)
        if lw >= 3 and x1 - x0 > 2 * lw + 2 and y1 - y0 > 2 * lw + 2:
            # a dark hairline inside the colour (white outside): the edge shows on a background of any colour
            d.rectangle((x0 + lw, y0 + lw, x1 - lw, y1 - lw), outline=(0, 0, 0), width=1)
    tags, outside = [None] * len(px), [False] * len(px)
    if numbered:
        font = _font(max(13, round(22 * scale)))
        placed = []
        for i, b in enumerate(px):
            text = str(i + 1)
            l, t, r, btm = d.textbbox((0, 0), text, font=font)
            tw, th = (r - l) + max(8, round(9 * scale)), (btm - t) + max(8, round(9 * scale))
            taken = [o for j, o in enumerate(px) if j != i] + placed
            rect, out = tag_position(b, (W, H), (tw, th), taken)
            placed.append(rect)
            tags[i], outside[i] = rect, out
        for i, rect in enumerate(tags):
            col = PALETTE[i % len(PALETTE)]
            d.rectangle((rect[0] - 1, rect[1] - 1, rect[2], rect[3]), fill=col, outline=(255, 255, 255), width=2)
            text = str(i + 1)
            l, t, r, btm = d.textbbox((0, 0), text, font=font)
            d.text((rect[0] + (rect[2] - rect[0] - (r - l)) / 2 - l, rect[1] + (rect[3] - rect[1] - (btm - t)) / 2 - t),
                   text, fill=(255, 255, 255), font=font)
    return im, tags, outside


def _render(job) -> Tuple[str, int]:
    """(src, dst, boxes, numbered) -> writes dst (JPEG, max side 1024); returns (dst, number of tags inside a box)."""
    from PIL import Image

    src, dst, boxes, numbered = job
    with Image.open(src) as im:
        im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    inside = 0
    if boxes:
        im, _, outside = draw_boxes(im, boxes, numbered)
        inside = sum(1 for o in outside if not o) if numbered else 0
    im.save(dst, "JPEG", quality=88)
    return dst, inside


# ---- candidate / distractor selection (pure functions, tested on hand-made records) -------------------------------

def vh_types(boxes: Sequence[Box], vh_boxes: Sequence[Box], vh_classes: Sequence[str], min_iou: float = 0.5) -> List[str]:
    """The view-hierarchy class of each candidate box (best-IoU leaf, '' when none matches)."""
    out = []
    for b in boxes:
        best, cls = min_iou, ""
        for v, c in zip(vh_boxes, vh_classes):
            s = iou(b, v)
            if s >= best:
                best, cls = s, c
        out.append(cls)
    return out


def _feature(box: Box, cls: str) -> Tuple[str, int, str]:
    """What a candidate looks like before anyone reads the expression: view class, log-area bin, screen zone."""
    y = _center(box)[1]
    return (cls, min(9, max(0, int(math.log10(max(_area(box), 1e-6)) * 2 + 10))),
            "status" if y < 0.04 else "nav" if y > 0.945 else "body")


def target_propensity(rows: Sequence[dict]) -> Dict[Tuple, float]:
    """{candidate feature: how often a candidate with it is the human's target} over RefExp rows. The detected
    candidates are mostly things nobody refers to (status-bar icons: 0.1% are targets; boxes matching no view: 0.4%)
    while an image button is the target 38% of the time, so random distractors would let "the box that looks like a
    control" answer without the expression (measured: 0.74 against a chance of 0.21)."""
    seen, hit = Counter(), Counter()
    for r in rows:
        types = vh_types(r["boxes"], r["vh_boxes"], r["vh_classes"])
        for i, b in enumerate(r["boxes"]):
            f = _feature(b, types[i])
            seen[f] += 1
            hit[f] += i == r["target"]
    return {f: (hit[f] + 0.5) / (n + 5.0) for f, n in seen.items()}


def fill_tiers(boxes: Sequence[Box], types: Sequence[str], propensity: Dict[Tuple, float],
               fulls: Sequence[float] = (0.35, 0.2)) -> List[List[float]]:
    """Keep-probabilities for filler distractors, strictest tier first. In a thinned tier a candidate is kept with
    probability propensity / ``full`` (1 from ``full`` upwards), i.e. as often as boxes like it are targets; the last
    tier keeps every candidate that matches a view of the hierarchy and lies outside the status / navigation bar."""
    prop = [propensity.get(_feature(b, t), 0.0) for b, t in zip(boxes, types)]
    tiers = [[min(1.0, p / full) for p in prop] for full in fulls]
    tiers.append([1.0 if (t and 0.04 <= _center(b)[1] <= 0.945) else 0.0 for b, t in zip(boxes, types)])
    return tiers


def refexp_candidates(boxes: Sequence[Box], target: int, types: Sequence[str], rng: random.Random,
                      other_targets: Sequence[Box] = (), tiers: Optional[Sequence[Sequence[float]]] = None,
                      k: Optional[int] = None) -> Optional[Tuple[List[Box], int, int, int]]:
    """The boxes to draw for one referring expression -> (boxes in number order, label, distractors that are targets
    of other expressions, filler tier or -1), or None when four separate boxes cannot be found.

    Distractors are, first, the targets of OTHER expressions about the same screen (``other_targets``): elements a
    human did refer to, a random subset, so nothing in the picture says which drawn box this expression means. Only
    when fewer than three of those exist is the set filled up to FOUR boxes from the row's own candidates: the target's
    view class first, thinned by ``tiers`` (``fill_tiers``; None = every drawable candidate). Fillers are never the
    nearest candidates: that would put the target in the middle of the drawn boxes. No two drawn boxes touch. The
    numbers are a random permutation, so the gold position is uniform."""
    if not 0 <= target < len(boxes) or not box_ok(boxes[target]):
        return None
    gold = tuple(boxes[target])
    chosen = [gold]
    pool = [b for b in dict.fromkeys(tuple(b) for b in other_targets) if box_ok(b)]
    rng.shuffle(pool)
    k = k or rng.randint(4, 6)
    for b in pool:
        if len(chosen) == k:
            break
        if not any(touches(b, c) for c in chosen):
            chosen.append(b)
    n_other, tier = len(chosen) - 1, -1
    if len(chosen) < 4:
        for tier, keep in enumerate(tiers if tiers is not None else [[1.0] * len(boxes)]):
            cand = [i for i in range(len(boxes)) if i != target and box_ok(boxes[i]) and rng.random() < keep[i]]
            rng.shuffle(cand)
            cand.sort(key=lambda i: not (types[i] and types[i] == types[target]))  # stable: same class first
            trial = chosen[:]
            for i in cand:
                if len(trial) == 4:
                    break
                if not any(touches(boxes[i], c) for c in trial):
                    trial.append(tuple(boxes[i]))
            if len(trial) == 4:
                chosen = trial
                break
        else:
            return None
    rng.shuffle(chosen)
    return chosen, chosen.index(gold), n_other, tier


_STOP = {"the", "a", "an", "to", "of", "on", "in", "for", "and", "or", "this", "that", "is", "it", "button", "icon",
         "option", "click", "select", "tap", "press", "go", "open", "view", "show", "see"}


def _cap_tokens(c: str) -> frozenset:
    toks = re.findall(r"[a-z0-9]+", c.lower())
    core = [t for t in toks if t not in _STOP]
    return frozenset(core or toks)


#: words that describe what a widget IS rather than which one it is; a caption made only of them ("toggle setting
#: on/off", "select option") fits every widget of its kind on the screen and is not used as the gold / description
GENERIC = {"the", "a", "an", "to", "of", "on", "in", "for", "and", "or", "this", "that", "is", "it", "with", "here",
           "button", "icon", "option", "options", "click", "select", "tap", "press", "choose", "toggle", "switch", "off",
           "setting", "turn", "enable", "disable", "check", "box", "checkbox", "radio", "field", "input", "element",
           "widget", "item", "control", "one"}


def specific_caption(captions: Sequence[str]) -> Optional[str]:
    """The human caption that says most about WHICH widget (most words outside ``GENERIC``, shortest on a tie); None
    when every caption is generic ("toggle on/off", "select this item")."""
    scored = sorted(captions, key=lambda c: (-len([t for t in re.findall(r"[a-z0-9]+", c.lower()) if t not in GENERIC]), len(c)))
    best = scored[0]
    return best if any(t not in GENERIC for t in re.findall(r"[a-z0-9]+", best.lower())) else None


def captions_conflict(a: Sequence[str], b: Sequence[str]) -> bool:
    """Could a caption of one widget also describe the other? True when any pair of their human captions is the same
    string, has the same content words, one's content words contain the other's, or they share >= half of them."""
    for x in a:
        tx = _cap_tokens(x)
        for y in b:
            ty = _cap_tokens(y)
            if x.strip().lower() == y.strip().lower() or tx <= ty or ty <= tx:
                return True
            if len(tx & ty) / len(tx | ty) >= 0.5:
                return True
    return False


def caption_options(target: dict, same_screen: Sequence[dict], same_app: Sequence[dict], rng: random.Random,
                    k: int = 4) -> Optional[Tuple[List[str], int, int]]:
    """Four captions for the highlighted widget -> (options, label, number of distractors from the same screen).
    Widgets are ``{"captions": [...], ...}``. The gold is the target's most specific human caption
    (``specific_caption``; None when all are generic); each distractor is the most specific
    human caption of ANOTHER widget (same screen first, then the same app; chosen by the same rule as the gold, so
    the gold is not the longest option) that conflicts neither with the target nor with a distractor already chosen.
    None when fewer than k - 1 such widgets exist."""
    gold = specific_caption(target["captions"])
    if gold is None:
        return None
    picked, n_same = [], 0
    for pool, is_same in ((same_screen, True), (same_app, False)):
        pool = [w for w in pool if w is not target]
        rng.shuffle(pool)
        for w in pool:
            if len(picked) == k - 1:
                break
            if specific_caption(w["captions"]) is None:
                continue
            if captions_conflict(target["captions"], w["captions"]) or any(captions_conflict(w["captions"], p["captions"]) for p in picked):
                continue
            picked.append(w)
            n_same += is_same
    if len(picked) < k - 1:
        return None
    opts = [gold] + [specific_caption(w["captions"]) for w in picked]  # the same rule as the gold: no length tell
    if len({o.strip().lower() for o in opts}) < k:
        return None
    order = list(range(k))
    rng.shuffle(order)
    return [opts[i] for i in order], order.index(0), n_same


def find_candidates(target: dict, same_screen: Sequence[dict], rng: random.Random,
                    k: Optional[int] = None) -> Optional[Tuple[List[dict], int]]:
    """Numbered widgets for "which element matches this caption" -> (widgets in number order, label) or None.
    Distractors are other CAPTIONED widgets of the screen (so their function is known to differ): no caption conflict
    with the target or each other, boxes not touching. A uniformly random subset, randomly numbered."""
    pool = [w for w in same_screen if w is not target]
    rng.shuffle(pool)
    picked = []
    k = k or rng.randint(4, 5)
    for w in pool:
        if len(picked) == k - 1:
            break
        if captions_conflict(target["captions"], w["captions"]) or touches(target["box"], w["box"]):
            continue
        if any(captions_conflict(w["captions"], p["captions"]) or touches(w["box"], p["box"]) for p in picked):
            continue
        picked.append(w)
    if len(picked) < 3:
        return None
    chosen = [target] + picked
    rng.shuffle(chosen)
    return chosen, chosen.index(target)


# ---- screen_annotation ---------------------------------------------------------------------------------------------

SA_TYPES = ("TEXT", "PICTOGRAM", "LIST_ITEM", "BUTTON", "IMAGE", "NAVIGATION_BAR", "LABEL", "TOOLBAR", "TEXT_INPUT",
            "CHECKBOX", "RADIO_BUTTON", "SWITCH", "PAGER_INDICATOR", "MAP", "KEYBOARD")
_T = "(?:%s)" % "|".join(sorted(SA_TYPES, key=len, reverse=True))
_SA_HEAD = re.compile(r"(%s)(?: (.*?))?? (\d{1,3}) (\d{1,3}) (\d{1,3}) (\d{1,3})(?= \(%s |\)*(?:, %s |$))" % (_T, _T, _T))


def parse_screen_annotation(s: str) -> Optional[List[dict]]:
    """``TYPE [text] x0 x1 y0 y1 [(children)], ...`` (coordinates 0-999) -> flat list of
    ``{"type", "text", "box": (x0, y0, x1, y1) in [0, 1], "depth", "parent"}``; None unless the whole string parses."""
    out, stack, i = [], [], 0
    while i < len(s):
        m = _SA_HEAD.match(s, i)
        if not m:
            return None
        x0, x1, y0, y1 = (int(m.group(g)) / 999.0 for g in (3, 4, 5, 6))
        out.append({"type": m.group(1), "text": m.group(2) or "", "box": (x0, y0, x1, y1), "depth": len(stack),
                    "parent": stack[-1] if stack else None})
        i = m.end()
        if s.startswith(" (", i):
            stack.append(len(out) - 1)
            i += 2
            continue
        while s.startswith(")", i):
            if not stack:
                return None
            stack.pop()
            i += 1
        if s.startswith(", ", i):
            i += 2
        elif i < len(s):
            return None
    return None if stack else out


#: classes asked about in ``ui_element_type`` (LABEL is left out: a LABEL is a TEXT that belongs to a control, which
#: the box alone does not show; MAP / KEYBOARD are too rare) and the name shown as an option
TYPE_NAMES = {"TEXT": "text", "PICTOGRAM": "icon", "LIST_ITEM": "list item", "BUTTON": "button", "IMAGE": "image",
              "NAVIGATION_BAR": "navigation bar", "TOOLBAR": "toolbar", "TEXT_INPUT": "text input field",
              "CHECKBOX": "checkbox", "RADIO_BUTTON": "radio button", "SWITCH": "switch", "PAGER_INDICATOR": "page indicator"}
CONFUSABLE = {"TEXT": ("BUTTON", "TEXT_INPUT", "LIST_ITEM", "TOOLBAR"),
              "BUTTON": ("TEXT", "PICTOGRAM", "TEXT_INPUT", "LIST_ITEM", "SWITCH"),
              "PICTOGRAM": ("IMAGE", "BUTTON", "CHECKBOX", "RADIO_BUTTON", "PAGER_INDICATOR"),
              "IMAGE": ("PICTOGRAM", "LIST_ITEM", "BUTTON"),
              "LIST_ITEM": ("TOOLBAR", "NAVIGATION_BAR", "TEXT", "BUTTON", "TEXT_INPUT"),
              "NAVIGATION_BAR": ("TOOLBAR", "LIST_ITEM", "PAGER_INDICATOR"),
              "TOOLBAR": ("NAVIGATION_BAR", "LIST_ITEM", "TEXT"),
              "TEXT_INPUT": ("TEXT", "BUTTON", "LIST_ITEM"),
              "CHECKBOX": ("RADIO_BUTTON", "SWITCH", "PICTOGRAM", "BUTTON"),
              "RADIO_BUTTON": ("CHECKBOX", "SWITCH", "PICTOGRAM", "PAGER_INDICATOR"),
              "SWITCH": ("CHECKBOX", "RADIO_BUTTON", "BUTTON", "PICTOGRAM"),
              "PAGER_INDICATOR": ("PICTOGRAM", "RADIO_BUTTON", "NAVIGATION_BAR")}
#: types counted in ``ui_element_count`` and how the question names them. Only types whose annotation is COMPLETE on
#: the screens looked at (2026-09-30, every element of the type outlined on ~20 screens): BUTTON, PICTOGRAM, IMAGE and
#: TEXT_INPUT are left out (some price chips are buttons and others not, navigation-bar icons are counted on some
#: screens only, some of a list's thumbnails are missing, hint bars are marked as inputs); TEXT blocks have no
#: countable unit.
COUNT_NAMES = {"CHECKBOX": "checkboxes", "RADIO_BUTTON": "radio buttons", "SWITCH": "toggle switches", "LIST_ITEM": "list items"}
MAX_COUNT = 12


def type_targets(elements: Sequence[dict]) -> List[int]:
    """Elements whose type can be asked: a named class, a drawable box, and no element of ANOTHER type on nearly the
    same box (a PICTOGRAM wrapping an IMAGE, a BUTTON wrapping its TEXT) that would make two answers right."""
    out = []
    for i, e in enumerate(elements):
        if e["type"] not in TYPE_NAMES or not box_ok(e["box"], max_area=0.5):
            continue
        if any(j != i and o["type"] != e["type"] and iou(e["box"], o["box"]) > 0.6 for j, o in enumerate(elements)):
            continue
        out.append(i)
    return out


def type_option_set(universe: Sequence[str], rng: random.Random, k: int = 4) -> List[str]:
    """An option set for ``ui_element_type``: a seed class plus k - 1 others, classes confusable with the seed 4x as
    likely. The caller then draws the gold UNIFORMLY from the set, so P(gold | options) = 1 / k whatever the class
    frequencies are."""
    seed = rng.choice(list(universe))
    rest = [c for c in universe if c != seed]
    weights = [4.0 if (c in CONFUSABLE.get(seed, ()) or seed in CONFUSABLE.get(c, ())) else 1.0 for c in rest]
    chosen = [seed]
    while len(chosen) < min(k, len(universe)):
        c = rng.choices(rest, weights)[0]
        j = rest.index(c)
        rest.pop(j)
        weights.pop(j)
        chosen.append(c)
    rng.shuffle(chosen)
    return chosen


def count_option_set(universe: Sequence[int], rng: random.Random, k: int = 4, span: int = 6) -> List[int]:
    """k distinct counts from a window of ``span`` neighbouring values of the (sorted) universe, ascending. The gold is
    then drawn uniformly from the set by the caller: rank-uniform, and no value is likelier to be right than another."""
    u = sorted(universe)
    span = min(span, len(u))
    start = rng.randrange(len(u) - span + 1)
    return sorted(rng.sample(u[start:start + span], min(k, span)))


def draw_uniform_gold(pools: Dict, rng: random.Random, option_set, n: int, min_universe: int = 4):
    """The exact "set first, gold second" sampler shared by ``ui_element_type`` and ``ui_element_count``.

    ``pools``: {value: [instances]} (consumed). Repeats up to ``n`` times: draw an option set from the values that
    still have instances (``option_set(universe, rng)``), draw the gold uniformly from the set, pop an instance of it.
    A value whose pool runs dry leaves the universe, so it is never offered as an option it could not be. Every row
    therefore has P(gold = o | options) = 1 / len(options) exactly. Yields (options, gold value, instance)."""
    universe = sorted(v for v, p in pools.items() if p)
    for _ in range(n):
        if len(universe) < min_universe:
            return
        opts = option_set(universe, rng)
        gold = rng.choice(opts)
        inst = pools[gold].pop()
        if not pools[gold]:
            universe.remove(gold)
        yield opts, gold, inst


# ---- OmniACT scripts ---------------------------------------------------------------------------------------------

_LINE = re.compile(r"^[Pp]yautogui\.(\w+)\((.*)\)$")
_POINT = re.compile(r"^(\s*)(-?\d+(?:\.\d+)?)(\s*,\s*)(-?\d+(?:\.\d+)?)")
POINT_FUNCS = {"click", "moveto", "rightclick", "doubleclick", "dragto", "tripleclick"}
PRESS_KEYS = ("enter", "tab", "space", "down", "up", "left", "right", "backspace", "esc", "delete")
HOTKEY_LAST = ("a", "c", "v", "x", "s", "z", "f", "t", "w", "r", "n", "p")


def parse_script(text: str) -> Optional[List[dict]]:
    """A task file's script -> [{"func", "args", "point": (x, y) or None}], None if a line is not a pyautogui call."""
    out = []
    for raw in text.strip().split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            return None
        func, args = m.group(1), m.group(2)
        p = _POINT.match(args) if func.lower() in POINT_FUNCS else None
        out.append({"func": func, "args": args, "point": (float(p.group(2)), float(p.group(4))) if p else None})
    return out or None


def _fmt(v: float, like: str) -> str:
    """``v`` written the way the script wrote the number it replaces: one decimal, or a whole number (a centre at .5
    is rounded: still inside the element, and the option does not stand out by its decimals)."""
    return "%.1f" % v if "." in like else "%d" % round(v)


def render_script(lines: Sequence[dict]) -> str:
    return "\n".join("pyautogui.%s(%s)" % (l["func"], l["args"]) for l in lines)


def swap_target(lines: Sequence[dict], old: Tuple[float, float], new: Tuple[float, float]) -> List[dict]:
    """The script with every use of point ``old`` replaced by ``new`` (same number format, same separators)."""
    out = []
    for l in lines:
        if l["point"] == old:
            m = _POINT.match(l["args"])
            args = "%s%s%s%s%s" % (m.group(1), _fmt(new[0], m.group(2)), m.group(3), _fmt(new[1], m.group(4)), l["args"][m.end():])
            out.append({"func": l["func"], "args": args, "point": new})
        else:
            out.append(dict(l))
    return out


def change_key(lines: Sequence[dict], rng: random.Random) -> Optional[List[dict]]:
    """The script with ONE key changed: another key for a ``press``, another last key for a ``hotkey``, the opposite
    direction for a ``scroll``. None when the script has no such line."""
    idx = [i for i, l in enumerate(lines) if l["func"].lower() in ("press", "hotkey", "scroll", "hscroll")]
    rng.shuffle(idx)
    for i in idx:
        l, func = lines[i], lines[i]["func"].lower()
        args = None
        if func in ("scroll", "hscroll"):
            m = re.match(r"^\s*([\"']?)([+-]?)(\d+)([\"']?)\s*$", l["args"])
            if m:
                args = "%s%s%s%s" % (m.group(1), "" if m.group(2) == "-" else "-", m.group(3), m.group(4))
        else:
            keys = list(re.finditer(r"([\"'])([^\"']+)\1", l["args"]))
            if keys:
                last = keys[-1]
                old = last.group(2)
                space = HOTKEY_LAST if (func == "hotkey" and len(keys) > 1) else PRESS_KEYS
                alts = [k for k in space if k != old.lower()]
                new = rng.choice(alts)
                new = new.capitalize() if old[:1].isupper() else new
                args = l["args"][:last.start(2)] + new + l["args"][last.end(2):]
        if args is not None and args != l["args"]:
            out = [dict(x) for x in lines]
            out[i]["args"] = args
            return out
    return None


def box_centers(boxes: dict) -> List[dict]:
    """OmniACT box file -> [{"key", "label", "center": (x, y), "rect": (l, t, r, b)}] (pixels of the screenshot)."""
    out = []
    for key, b in boxes.items():
        try:
            (l, t), (r, btm) = b["top_left"], b["bottom_right"]
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"key": key, "label": str(b.get("label", "")).strip().lower(), "center": ((l + r) / 2, (t + btm) / 2),
                    "rect": (l, t, r, btm)})
    return out


def _box_at(point, centers):
    for c in centers:
        if abs(c["center"][0] - point[0]) < 0.01 and abs(c["center"][1] - point[1]) < 0.01:
            return c
    return None


def other_targets(lines: Sequence[dict], centers: Sequence[dict], screen: Tuple[int, int], popularity: Dict,
                  rng: random.Random, n: int, min_gap: float = 0.035) -> List[dict]:
    """``n`` elements of the screen to point a wrong script at: not an element the script already uses, not one with
    the same hand-written label as a used one, its box not overlapping a used one, and its centre at least
    ``min_gap`` of the screen diagonal away from every used point and from each other (so the options differ visibly
    at 1024 px). Elements that other tasks of the screen point at are preferred (``popularity``: {box key: number
    of scripts using it}), so a distractor's coordinates are as "task-like" as the gold's."""
    used = [c for c in (_box_at(l["point"], centers) for l in lines if l["point"]) if c]
    pts = [l["point"] for l in lines if l["point"]]
    diag = math.hypot(*screen)
    pool = []
    for c in centers:
        if any(c["label"] and c["label"] == u["label"] for u in used):
            continue
        if any(_rects_touch(c["rect"], u["rect"]) for u in used):
            continue
        if any(math.hypot(c["center"][0] - p[0], c["center"][1] - p[1]) < min_gap * diag for p in pts):
            continue
        pool.append(c)
    out = []
    while pool and len(out) < n:
        c = rng.choices(pool, [0.3 + popularity.get(x["key"], 0) for x in pool])[0]
        out.append(c)
        pool = [x for x in pool if x is not c and not (x["label"] and x["label"] == c["label"])
                and math.hypot(x["center"][0] - c["center"][0], x["center"][1] - c["center"][1]) >= min_gap * diag]
    return out


def script_options(lines: Sequence[dict], centers: Sequence[dict], screen: Tuple[int, int], others: Sequence[Sequence[dict]],
                   popularity: Dict, rng: random.Random, design: Optional[str] = None) -> Optional[Tuple[List[str], int, str]]:
    """Four scripts, one correct -> (options, label, design) or None. The designs are symmetric, so nothing about the
    option SET (which script is the odd one out, which one the others were derived from) points at the gold:

      targets  the gold and three copies that point ONE of its clicks at three other elements of the screen;
      keys     two targets x two keys: gold, other target, other key, both;
      tasks    two tasks x two targets: gold, gold at another element, the script of ANOTHER task of this screen,
               that script at another element.
    ``others``: parsed gold scripts of other tasks on the same screen (same split)."""
    gold = render_script(lines)
    slots = sorted({l["point"] for l in lines if l["point"] and _box_at(l["point"], centers)})
    if not slots:
        return None
    slot = rng.choice(slots)
    designs = ["targets"]
    keyed = change_key(lines, rng)
    if keyed is not None:
        designs.append("keys")
    rivals = [o for o in others if render_script(o) != gold and any(l["point"] and _box_at(l["point"], centers) for l in o)]
    if rivals:
        designs.append("tasks")
    if design is None:
        design = rng.choices(designs, [{"targets": 0.7, "keys": 0.15, "tasks": 0.15}[d] for d in designs])[0]
    elif design not in designs:
        return None
    if design == "targets":
        alts = other_targets(lines, centers, screen, popularity, rng, 3)
        opts = [gold] + [render_script(swap_target(lines, slot, a["center"])) for a in alts]
    elif design == "keys":
        alts = other_targets(lines, centers, screen, popularity, rng, 1)
        if not alts:
            return None
        opts = [gold, render_script(swap_target(lines, slot, alts[0]["center"])), render_script(keyed),
                render_script(swap_target(keyed, slot, alts[0]["center"]))]
    else:
        rival = rng.choice(rivals)
        r_slot = rng.choice(sorted({l["point"] for l in rival if l["point"] and _box_at(l["point"], centers)}))
        a = other_targets(lines, centers, screen, popularity, rng, 1)
        b = other_targets(rival, centers, screen, popularity, rng, 1)
        if not a or not b:
            return None
        opts = [gold, render_script(swap_target(lines, slot, a[0]["center"])), render_script(rival),
                render_script(swap_target(rival, r_slot, b[0]["center"]))]
    if len(opts) < 4 or len(set(opts)) < 4:
        return None
    order = list(range(4))
    rng.shuffle(order)
    return [opts[i] for i in order], order.index(0), design


# ---- raw data: download, Rico extraction ----------------------------------------------------------------------------

def _dir(key: str, raw: str = RAW_ROOT) -> str:
    return os.path.join(raw, SOURCES[key]["dir"])


def _download(url: str, dst: str) -> None:
    import urllib.request

    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, open(dst + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.rename(dst + ".part", dst)


def fetch(raw: str = RAW_ROOT, log=print) -> None:
    """Download every source at its pinned revision (skips files already there): ~7 GB, 6.5 GB of it Rico."""
    for key in ("uibert", "widget_caption", "screen_annotation"):
        s = SOURCES[key]
        for rel in s["files"]:
            _download("%s/%s/%s" % (s["raw"], s["revision"], rel), os.path.join(_dir(key, raw), rel))
        log("%s: %s @ %s" % (key, s["url"], s["revision"][:12]))
    for rel in SOURCES["rico"]["files"]:
        _download(SOURCES["rico"]["base"] + rel, os.path.join(_dir("rico", raw), rel))
    log("rico: %s" % SOURCES["rico"]["revision"])
    from huggingface_hub import hf_hub_download

    o = SOURCES["omniact"]
    for rel in o["files"]:
        if not os.path.exists(os.path.join(_dir("omniact", raw), rel)):
            hf_hub_download("Writer/omniact", rel, repo_type="dataset", revision=o["revision"], local_dir=_dir("omniact", raw))
    if not os.path.isdir(os.path.join(_dir("omniact", raw), "data", "tasks")):
        import zipfile

        with zipfile.ZipFile(os.path.join(_dir("omniact", raw), "data.zip")) as z:
            z.extractall(_dir("omniact", raw))
    log("omniact: %s @ %s" % (o["url"], o["revision"][:12]))


def _read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().split()


def _widget_rows(raw: str = RAW_ROOT) -> List[dict]:
    with open(os.path.join(_dir("widget_caption", raw), "widget_captions.csv"), encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["datasetId"] == "rc"]


def source_splits(raw: str = RAW_ROOT) -> Dict[str, Dict[str, set]]:
    """{source: {"train" | "val" | "test": set of Rico screen ids}} from each source's own split files."""
    out = {"uibert": {}, "widget_caption": {}, "screen_annotation": {}}
    for split, part in (("train", "train"), ("val", "dev"), ("test", "test")):
        out["uibert"][split] = {e["image/id"][0].decode() for e in
                                read_tfrecord(os.path.join(_dir("uibert", raw), "ref_exp", part + ".tfrecord"))}
        out["widget_caption"][split] = set(_read_lines(os.path.join(_dir("widget_caption", raw), "split", part + ".txt")))
    rc = {r["screenId"] for r in _widget_rows(raw)}
    out["widget_caption"] = {s: v & rc for s, v in out["widget_caption"].items()}
    csv.field_size_limit(10 ** 9)
    for split, part in (("train", "train"), ("val", "valid"), ("test", "test")):
        with open(os.path.join(_dir("screen_annotation", raw), part + ".csv"), encoding="utf-8", newline="") as f:
            out["screen_annotation"][split] = {r["screen_id"] for r in csv.DictReader(f)}
    return out


def screenqa_screens(root: str = DATA_ROOT) -> Dict[str, set]:
    """Rico screen ids already used by ``rico_screenqa`` (empty when that set is not built)."""
    out = {"train": set(), "val": set()}
    for split in out:
        p = os.path.join(root, "rico_screenqa", split + ".jsonl")
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    m = re.search(r"screen-(\d+)\.jpg", json.loads(line).get("image") or "")
                    if m:
                        out[split].add(m.group(1))
    return out


def _save_screen(job) -> str:
    from PIL import Image

    data, dst = job
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    im.save(dst + ".part", "JPEG", quality=93)
    os.rename(dst + ".part", dst)
    return dst


def _vh_nodes(vh: dict, node_ids: Sequence[str]) -> dict:
    """Bounds of the captioned nodes of one Rico view hierarchy (``nodeId`` = child indices from the root, as in the
    Widget Captioning README)."""
    root = (vh.get("activity") or {}).get("root") or {}
    out = {"root": root.get("bounds"), "nodes": {}}
    for nid in node_ids:
        node = root
        try:
            for i in nid.split(".")[1:]:
                node = node["children"][int(i)]
            out["nodes"][nid] = {"bounds": node["bounds"], "class": node.get("class", ""),
                                 "visible": bool(node.get("visible-to-user", True))}
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


RICO_FRAME = (1440, 2560)  # view-hierarchy bounds are pixels of this frame, whatever the screenshot's resolution


def screen_path(sid: str, raw: str = RAW_ROOT) -> str:
    return os.path.join(_dir("rico", raw), "screens", "%s.jpg" % sid)


def extract_rico(raw: str = RAW_ROOT, root: str = DATA_ROOT, workers: int = 6, log=print) -> None:
    """One streaming pass over ``unique_uis.tar.gz``: the screenshots of every screen a source (or ``rico_screenqa``)
    names -> ``<raw>/rico/screens/<id>.jpg`` (max side 1024), and the bounds of the captioned nodes of the Widget
    Captioning screens -> ``<raw>/rico/widget_nodes.json``. Nothing else is unpacked."""
    import tarfile
    from multiprocessing import Pool

    need = set().union(*(v for s in source_splits(raw).values() for v in s.values()), *screenqa_screens(root).values())
    wc = defaultdict(list)
    for r in _widget_rows(raw):
        wc[r["screenId"]].append(r["nodeId"])
    os.makedirs(os.path.join(_dir("rico", raw), "screens"), exist_ok=True)
    nodes_path = os.path.join(_dir("rico", raw), "widget_nodes.json")
    todo = {s for s in need if not os.path.exists(screen_path(s, raw))}
    if not todo and os.path.exists(nodes_path):
        log("rico: %d screens already extracted" % len(need))
        return
    nodes, pending, done = {}, [], 0
    with tarfile.open(os.path.join(_dir("rico", raw), "unique_uis.tar.gz"), "r|gz") as tar, Pool(workers) as pool:
        for m in tar:
            sid, ext = os.path.splitext(os.path.basename(m.name))
            if ext == ".jpg" and sid in todo:
                pending.append(pool.apply_async(_save_screen, ((tar.extractfile(m).read(), screen_path(sid, raw)),)))
                if len(pending) >= 8 * workers:  # bounded: at most this many screenshots in memory
                    pending.pop(0).get()
                done += 1
            elif ext == ".json" and sid in wc:
                nodes[sid] = _vh_nodes(json.load(tar.extractfile(m)), wc[sid])
        for p in pending:
            p.get()
    with open(nodes_path, "w") as f:
        json.dump(nodes, f)
    log("rico: extracted %d screenshots (%d needed), node bounds for %d screens" % (done, len(need), len(nodes)))


# ---- near-duplicate screens and the side (train / val / unused) of every Rico screen ---------------------------------

def _dv():
    from . import visionjev_dedupe

    return visionjev_dedupe.dv


def _signature(path: str):
    from PIL import Image

    dv = _dv()
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with Image.open(path) as im:
        return sha, dv.dhash(im), dv.thumbnail(im)


def near_pairs(shas: Sequence[str], dhashes: Sequence[Optional[int]], thumbs, max_dist: int, min_corr: float) -> List[Tuple[int, int]]:
    """Index pairs (i < j) of images that match: equal bytes, or dHashes within ``max_dist`` bits AND 64 px thumbnails
    correlating >= ``min_corr`` (the data validator's test; flat images have no dHash and only match by bytes)."""
    import numpy as np

    pairs, first = set(), {}
    for i, s in enumerate(shas):
        if s in first:
            pairs.add((first[s], i))
        first.setdefault(s, i)
    idx = np.array([i for i, h in enumerate(dhashes) if h is not None], dtype=np.int64)
    if len(idx) > 1:
        h = np.array([dhashes[i] for i in idx], dtype=np.uint64)
        t = np.asarray(thumbs, dtype=np.uint8)[idx]
        for a in range(0, len(idx), 256):
            d = np.bitwise_count(h[a:a + 256, None] ^ h[None, :])
            ii, jj = np.nonzero(d <= max_dist)
            keep = jj > ii + a
            ii, jj = ii[keep] + a, jj[keep]
            for b in range(0, len(ii), 4096):
                x, y = t[ii[b:b + 4096]].astype(np.float32), t[jj[b:b + 4096]].astype(np.float32)
                x -= x.mean(1, keepdims=True)
                y -= y.mean(1, keepdims=True)
                den = np.sqrt((x * x).sum(1) * (y * y).sum(1))
                corr = np.divide((x * y).sum(1), den, out=np.zeros_like(den), where=den > 0)
                for p in np.nonzero(corr >= min_corr)[0]:
                    pairs.add((int(idx[ii[b + p]]), int(idx[jj[b + p]])))
    return sorted(pairs)


def assign_sides(ids: Sequence[str], pairs: Sequence[Tuple[int, int]], splits: Dict[str, Dict[str, set]],
                 screenqa: Dict[str, set]) -> Dict[str, str]:
    """The side of every screen, from the screens it matches DIRECTLY (``pairs``; chains are not followed: sparse
    white screens chain into one group of thousands). With near(s) = s and its matches:

      "test"   near(s) holds a screen of some source's TEST split                 -> unused (tests stay clean)
      "val"    s is a source DEV screen (or a ``rico_screenqa`` val screen) and near(s) holds no ``rico_screenqa``
               train screen                                                        -> val rows only
      "none"   near(s) holds a dev / ``rico_screenqa`` val screen otherwise        -> unused
      "train"  otherwise                                                           -> train rows only.
    A source-train row is used when its screen's side is "train", a source-dev row when it is "val"; so no train
    image matches a val image of any of these sets or of ``rico_screenqa``, and no val image a train image."""
    near = {s: {s} for s in ids}
    for a, b in pairs:
        near[ids[a]].add(ids[b])
        near[ids[b]].add(ids[a])
    test = set().union(*(s["test"] for s in splits.values()))
    val = set().union(*(s["val"] for s in splits.values())) | screenqa["val"]

    def side(s):
        if near[s] & test:
            return "test"
        if near[s] & val:
            return "val" if (s in val and not near[s] & screenqa["train"]) else "none"
        return "train"

    return {s: side(s) for s in ids}


RAW_MAX_DIST, RAW_MIN_CORR = 9, 0.97  # a margin over the validator's 8 bits / 0.98: drawn boxes move both a little


def rico_sides(raw: str = RAW_ROOT, root: str = DATA_ROOT, workers: int = 6, log=print) -> Dict[str, str]:
    """{screen id: side} for every extracted screen (cached in ``<raw>/rico/sides.json`` with what it was built from)."""
    from multiprocessing import Pool

    splits, sq = source_splits(raw), screenqa_screens(root)
    ids = sorted(set().union(*(v for s in splits.values() for v in s.values()), *sq.values()), key=int)
    ids = [s for s in ids if os.path.exists(screen_path(s, raw))]
    key = hashlib.md5(json.dumps([ids, sorted(sq["train"]), sorted(sq["val"]), RAW_MAX_DIST, RAW_MIN_CORR]).encode()).hexdigest()
    path = os.path.join(_dir("rico", raw), "sides.json")
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        if cached.get("key") == key:
            return cached["side"]
    with Pool(workers) as pool:
        sig = pool.map(_signature, [screen_path(s, raw) for s in ids], chunksize=256)
    import numpy as np

    thumbs = np.frombuffer(b"".join(s[2] for s in sig), dtype=np.uint8).reshape(len(sig), -1)
    pairs = near_pairs([s[0] for s in sig], [s[1] for s in sig], thumbs, RAW_MAX_DIST, RAW_MIN_CORR)
    side = assign_sides(ids, pairs, splits, sq)
    stats = {"screens": len(ids), "matching_pairs": len(pairs), "sides": dict(Counter(side.values()))}
    for name, s in splits.items():
        stats[name] = {"train_kept": sum(side.get(x) == "train" for x in s["train"]), "train": len(s["train"]),
                       "val_kept": sum(side.get(x) == "val" for x in s["val"]), "val": len(s["val"])}
    with open(path, "w") as f:
        json.dump({"key": key, "match": "sha256, or dHash <= %d bits and 64 px thumbnail correlation >= %.2f (direct "
                   "matches, no chains)" % (RAW_MAX_DIST, RAW_MIN_CORR), "stats": stats, "side": side}, f)
    log("rico sides: %s" % json.dumps(stats))
    return side


def sides_stats(raw: str = RAW_ROOT) -> dict:
    with open(os.path.join(_dir("rico", raw), "sides.json")) as f:
        c = json.load(f)
    return {"match": c["match"], **c["stats"]}


# ---- loading the annotations -----------------------------------------------------------------------------------------

def _clamp(b) -> Box:
    return tuple(min(1.0, max(0.0, float(v))) for v in b)


def load_refexp(raw: str = RAW_ROOT) -> Dict[str, List[dict]]:
    out = {}
    for split, part in (("train", "train"), ("val", "dev")):
        rows = []
        for e in read_tfrecord(os.path.join(_dir("uibert", raw), "ref_exp", part + ".tfrecord")):
            def col(prefix):
                return list(zip(*(e.get("%s/%s" % (prefix, k), []) for k in ("xmin", "ymin", "xmax", "ymax"))))

            rows.append({"screen": e["image/id"][0].decode(), "target": int(e["image/ref_exp/label"][0]),
                         "text": e["image/ref_exp/text"][0].decode("utf-8", "replace").strip(),
                         "boxes": [_clamp(b) for b in col("image/object/bbox")],
                         "vh_boxes": [_clamp(b) for b in col("image/view_hierarchy/bbox")],
                         "vh_classes": [c.decode("utf-8", "replace") for c in e.get("image/view_hierarchy/class/name", [])]})
        out[split] = rows
    return out


def load_widgets(raw: str = RAW_ROOT) -> Dict[str, Dict[str, List[dict]]]:
    """{split: {screen: [{"screen", "node", "captions", "box" (None if not drawable), "app"}]}} (rc screens only)."""
    with open(os.path.join(_dir("rico", raw), "widget_nodes.json")) as f:
        nodes = json.load(f)
    with open(os.path.join(_dir("rico", raw), "ui_details.csv"), encoding="utf-8", newline="") as f:
        apps = {r["UI Number"]: r["App Package Name"] for r in csv.DictReader(f)}
    split_of, sizes = {}, {}
    for split, part in (("train", "train"), ("val", "dev"), ("test", "test")):
        for sid in _read_lines(os.path.join(_dir("widget_caption", raw), "split", part + ".txt")):
            split_of[sid] = split

    def portrait(sid):
        from PIL import Image

        if sid not in sizes:
            try:
                with Image.open(screen_path(sid, raw)) as im:
                    sizes[sid] = im.size[0] < im.size[1]
            except OSError:
                sizes[sid] = False
        return sizes[sid]

    out = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    for r in _widget_rows(raw):
        sid = r["screenId"]
        caps = list(dict.fromkeys(c.strip() for c in r["captions"].split("|") if c.strip()))
        if sid not in split_of or not caps:
            continue
        box = None
        node = ((nodes.get(sid) or {}).get("nodes") or {}).get(r["nodeId"])
        if node and node["visible"] and portrait(sid):
            b = node["bounds"]  # absolute pixels of the 1440 x 2560 device frame (NOT relative to the root view,
            #                     whose own bounds often stop above the navigation bar)
            full = (b[0] / RICO_FRAME[0], b[1] / RICO_FRAME[1], b[2] / RICO_FRAME[0], b[3] / RICO_FRAME[1])
            c = _clamp(full)
            if _area(full) > 0 and _area(c) / _area(full) >= 0.7 and box_ok(c):
                box = c
        out[split_of[sid]][sid].append({"screen": sid, "node": r["nodeId"], "captions": caps, "box": box, "app": apps.get(sid, "")})
    return out


def load_screen_annotation(raw: str = RAW_ROOT) -> Dict[str, Dict[str, List[dict]]]:
    csv.field_size_limit(10 ** 9)
    out = {}
    for split, part in (("train", "train"), ("val", "valid")):
        out[split] = {}
        with open(os.path.join(_dir("screen_annotation", raw), part + ".csv"), encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                els = parse_screen_annotation(r["screen_annotation"])
                if els:
                    out[split][r["screen_id"]] = els
    return out


def _omniact_file(base: str, rel: str) -> Optional[str]:
    """The Hub index names ``screen_1.png`` / ``boxes/<app>/screen_1.json``; the web part of data.zip holds
    ``screen1.png`` / ``screen1_boxes.json``."""
    p = os.path.join(base, rel)
    if os.path.exists(p):
        return p
    d, fn = os.path.split(p)
    stem, ext = os.path.splitext(fn)
    for alt in (stem.replace("_", "") + ext, stem.replace("_", "") + "_boxes" + ext):
        if os.path.exists(os.path.join(d, alt)):
            return os.path.join(d, alt)
    return None


def load_omniact(raw: str = RAW_ROOT) -> Dict[str, List[dict]]:
    """{split: [{"task", "lines", "image", "boxes", "app", "screen"}]} for train / val (test is never read). Tasks
    whose script does not parse as pyautogui calls, or whose screenshot / box file is missing, are skipped."""
    base = _dir("omniact", raw)
    out = {}
    for split in ("train", "val"):
        rows = []
        with open(os.path.join(base, split + ".json")) as f:
            index = json.load(f)
        for _, v in sorted(index.items(), key=lambda kv: int(kv[0])):
            img, box = _omniact_file(base, v["image"]), _omniact_file(base, v["box"])
            if not img or not box or not os.path.exists(os.path.join(base, v["task"])):
                continue
            with open(os.path.join(base, v["task"]), encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
            head, _, rest = text.partition("\n")
            task = re.sub(r"^\s*task\s*:\s*", "", head, flags=re.I).strip()
            lines = parse_script(re.sub(r"^\s*output script\s*:?[ \t]*", "", rest.strip(), flags=re.I))
            if not task or not lines:
                continue
            parts = v["image"].split("/")
            rows.append({"task": task, "lines": lines, "image": img, "boxes": box, "app": "%s/%s" % (parts[2], parts[3]),
                         "screen": "%s-%s-%s" % (parts[2], parts[3], os.path.splitext(parts[4])[0].replace("_", ""))})
        out[split] = rows
    return out


# ---- writing a dataset -----------------------------------------------------------------------------------------------

RICO_ROW_LICENSE = "Rico screenshot (third-party app UI; see manifest license.images)"
OMNIACT_ROW_LICENSE = "MIT (OmniACT; screenshot of third-party software)"


def _numbers(k: int) -> List[str]:
    return [str(i + 1) for i in range(k)]


def box_shortcuts(rows: Sequence[Tuple[Sequence[Box], int]]) -> dict:
    """Can the geometry of the drawn boxes alone find the gold? (largest box, smallest box, the box nearest the
    centroid of all drawn boxes, the top-most box) vs chance."""
    if not rows:
        return {}
    hit = Counter()
    for boxes, label in rows:
        cx = sum(_center(b)[0] for b in boxes) / len(boxes)
        cy = sum(_center(b)[1] for b in boxes) / len(boxes)
        idx = range(len(boxes))
        hit["largest"] += max(idx, key=lambda i: _area(boxes[i])) == label
        hit["smallest"] += min(idx, key=lambda i: _area(boxes[i])) == label
        hit["nearest_centroid"] += min(idx, key=lambda i: math.hypot(_center(boxes[i])[0] - cx, _center(boxes[i])[1] - cy)) == label
        hit["topmost"] += min(idx, key=lambda i: boxes[i][1]) == label
    out = {k: round(v / len(rows), 3) for k, v in hit.items()}
    out["chance"] = round(sum(1 / len(b) for b, _ in rows) / len(rows), 3)
    return out


class _Out:
    """Collects the rows and render jobs of one dataset, then writes it the way the other builders do (tmp dir,
    manifest, rename, ``_READY``, validator-matcher dedupe of val against train)."""

    def __init__(self, root: str, name: str):
        self.root, self.name = root, name
        self.tmp = os.path.join(root, ".tmp-" + name)
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.makedirs(os.path.join(self.tmp, "images"))
        self.rows = {"train": [], "val": []}
        self.jobs, self.per_key, self.shared = [], Counter(), {}
        self.geometry = {"train": [], "val": []}
        self.stats = Counter()

    def image(self, src: str, key: str, boxes: Optional[Sequence[Box]], numbered: bool) -> str:
        """Register one image to render. With boxes: a new file per call; without: one file per ``key``."""
        if not boxes:
            if key not in self.shared:
                self.shared[key] = "images/%s.jpg" % key
                self.jobs.append((src, os.path.join(self.tmp, self.shared[key]), None, False))
            return self.shared[key]
        rel = "images/%s-%d.jpg" % (key, self.per_key[key])
        self.per_key[key] += 1
        self.jobs.append((src, os.path.join(self.tmp, rel), list(boxes), numbered))
        return rel

    def add(self, split: str, rec: dict, boxes: Optional[Sequence[Box]] = None) -> None:
        self.rows[split].append(rec)
        if boxes is not None:
            self.geometry[split].append((list(boxes), rec["label"]))

    def close(self, manifest: dict, workers: int = 6, log=print) -> dict:
        from multiprocessing import Pool

        from .visionjev_dedupe import dedupe

        with Pool(workers) as pool:
            inside = sum(n for _, n in pool.imap_unordered(_render, self.jobs, chunksize=64))
        counts, labels = {}, {}
        for split, rows in self.rows.items():
            counts[split] = len(rows)
            labels[split] = dict(sorted(Counter("%d:%d" % (len(r["question"]["criteria"]), r["label"]) for r in rows).items()))
            with open(os.path.join(self.tmp, split + ".jsonl"), "w", encoding="utf-8") as f:
                for i, r in enumerate(rows):
                    f.write(json.dumps(dict({"id": "%s-%s-%d" % (self.name, split, i)}, **r), ensure_ascii=False) + "\n")
        manifest = dict(manifest, counts=counts, label_counts=labels, images=len(self.jobs),
                        build=dict(self.stats, number_tags_inside_a_box=inside,
                                   number_tags=sum(len(j[2]) for j in self.jobs if j[2] and j[3])),
                        box_shortcuts={s: box_shortcuts(g) for s, g in self.geometry.items() if g})
        with open(os.path.join(self.tmp, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        final = os.path.join(self.root, self.name)
        if os.path.exists(final):
            shutil.rmtree(final)
        os.rename(self.tmp, final)
        with open(os.path.join(final, "_READY"), "w") as f:
            json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
        log("%s: %s %s" % (self.name, counts, dict(self.stats)))
        dedupe(self.name, self.root, log=log)  # val rows whose (drawn) image still matches a train image are dropped
        return manifest


def _manifest(name: str, source: str, use: str, note: str, rico: bool, raw: str = RAW_ROOT) -> dict:
    s = SOURCES[source]
    m = {"dataset": name, "source": "%s (%s)" % (s["url"], note), "revision": s["revision"],
         "license": {"annotations": "%s (%s)" % (s["license"], s["url"].split("//")[1]),
                     "images": RICO_IMAGES if rico else "OmniACT screenshots (MIT repository; screens of third-party "
                                                        "desktop apps and websites)",
                     "commercial": True, "verified": s["verified"]},
         "attribution": [s["attribution"]] + ([SOURCES["rico"]["attribution"]] if rico else []),
         "labels": s["labels"], "use": use,
         "splits": "source train -> train, source dev / valid -> val, source test untouched"}
    if rico:
        m["image_source"] = {"url": SOURCES["rico"]["url"], "revision": SOURCES["rico"]["revision"]}
        m["rico_sides"] = sides_stats(raw)
    return m


def _passes(targets: Sequence[dict], limits: Sequence[int]):
    """Targets in the given order, one per screen first, then a second per screen, ... (screen variety before depth)."""
    used, done = Counter(), set()
    for limit in limits:
        for i, t in enumerate(targets):
            if i not in done and used[t["screen"]] < limit:
                done.add(i)
                used[t["screen"]] += 1
                yield t


INS_REFEXP = ("Which numbered box marks the element this description refers to?",
              "The description refers to one of the numbered elements on the screen. Which one?")
INS_CAPTION = ("Which caption describes the element outlined in red?",
               "What does the element outlined in red do?")
INS_FIND = ("Which numbered element matches this description?",
            "Which of the numbered boxes marks the element described?")
INS_TYPE = ("What type of UI element is outlined in red?",
            "The red outline marks one UI element. What kind of element is it?")
INS_COUNT = ("How many %s are on this screen?", "Count the %s on this screen. How many are there?")
INS_SCRIPT = ("Which PyAutoGUI script carries out the task on this screen?",
              "Which script performs the requested task on the screen shown?")


def build_refexp(root=DATA_ROOT, raw=RAW_ROOT, workers=6, log=print, seed=0, caps=(TRAIN_CAP, VAL_CAP)) -> dict:
    name = "ui_refexp"
    side, data, out = rico_sides(raw, root, workers, log), load_refexp(raw), _Out(root, name)
    propensity = target_propensity(data["train"])  # from the source train split only
    hits = defaultdict(lambda: [0, 0])
    for split, cap in zip(("train", "val"), caps):
        rng = random.Random("%s-%s-%d" % (name, split, seed))
        rows = [r for r in data[split] if side.get(r["screen"]) == split]
        out.stats[split + "_source_rows_on_this_side"] = len(rows)
        targets = defaultdict(list)  # screen -> (row, its target box)
        for n, r in enumerate(rows):
            if 0 <= r["target"] < len(r["boxes"]):
                targets[r["screen"]].append((n, r["boxes"][r["target"]]))
        order = list(range(len(rows)))
        rng.shuffle(order)
        for n in order:
            r = rows[n]
            if len(out.rows[split]) >= cap:
                break
            if len(r["text"]) < 3 or not 0 <= r["target"] < len(r["boxes"]):
                out.stats[split + "_skip_bad_record"] += 1
                continue
            types = vh_types(r["boxes"], r["vh_boxes"], r["vh_classes"])
            gold = r["boxes"][r["target"]]
            others = [b for m, b in targets[r["screen"]] if m != n and not touches(b, gold)]
            pick = refexp_candidates(r["boxes"], r["target"], types, rng, others, fill_tiers(r["boxes"], types, propensity))
            if pick is None:
                out.stats[split + "_skip_under_4_separate_boxes"] += 1
                continue
            boxes, label, n_other, tier = pick
            kind = "other_targets" if tier < 0 else "fill_%d_tier_%d" % (len(boxes) - 1 - n_other, tier)
            out.stats["%s_%s" % (split, kind)] += 1
            # the strongest picture-only guess: the drawn box whose (class, size, zone) is most often a target
            prop = [propensity.get(_feature(b, vh_types([b], r["vh_boxes"], r["vh_classes"])[0]), 0.0) for b in boxes]
            h = hits["%s_%s" % (split, "other_targets" if tier < 0 else "with_fillers")]
            h[0] += max(range(len(boxes)), key=lambda i: prop[i]) == label
            h[1] += 1
            out.add(split, {"image": out.image(screen_path(r["screen"], raw), r["screen"], boxes, True),
                            "state_text": json.dumps({"description": r["text"]}, ensure_ascii=False),
                            "question": {"type": "choice", "instructions": rng.choice(INS_REFEXP), "criteria": _numbers(len(boxes))},
                            "label": label, "image_license": RICO_ROW_LICENSE, "kind": kind,
                            "attribution": {"source": "UIBert RefExp", "rico_screen": r["screen"]}}, boxes)
    m = _manifest(name, "uibert", "UI grounding: which numbered element a referring expression means",
                  "ref_exp TFRecords: candidates + target index", True, raw)
    m["most_target_like_box_is_gold"] = {k: {"rate": round(h / max(1, n), 3), "n": n} for k, (h, n) in sorted(hits.items())}
    return out.close(m, workers, log)


def _widget_screens(raw, root, workers, log):
    side, data = rico_sides(raw, root, workers, log), load_widgets(raw)
    return {split: {sid: ws for sid, ws in sorted(data[split].items(), key=lambda kv: int(kv[0])) if side.get(sid) == split}
            for split in ("train", "val")}


def build_widget_caption(root=DATA_ROOT, raw=RAW_ROOT, workers=6, log=print, seed=0, caps=(TRAIN_CAP, VAL_CAP)) -> dict:
    name = "ui_widget_caption"
    screens, out = _widget_screens(raw, root, workers, log), _Out(root, name)
    for split, cap in zip(("train", "val"), caps):
        rng = random.Random("%s-%s-%d" % (name, split, seed))
        by_app = defaultdict(list)
        for ws in screens[split].values():
            for w in ws:
                if w["app"]:
                    by_app[w["app"]].append(w)
        targets = [w for ws in screens[split].values() for w in ws if w["box"]]
        out.stats[split + "_drawable_widgets"] = len(targets)
        rng.shuffle(targets)
        for w in _passes(targets, (1, 2, 3, 4)):
            if len(out.rows[split]) >= cap:
                break
            res = caption_options(w, screens[split][w["screen"]], [x for x in by_app[w["app"]] if x["screen"] != w["screen"]], rng)
            if res is None:
                out.stats[split + "_skip_under_3_distractor_widgets"] += 1
                continue
            opts, label, n_same = res
            out.stats["%s_distractors_from_same_screen_%d" % (split, n_same)] += 1
            out.add(split, {"image": out.image(screen_path(w["screen"], raw), w["screen"], [w["box"]], False),
                            "state_text": None,
                            "question": {"type": "choice", "instructions": rng.choice(INS_CAPTION), "criteria": opts},
                            "label": label, "image_license": RICO_ROW_LICENSE, "kind": "same_screen_%d" % n_same,
                            "attribution": {"source": "Widget Captioning", "rico_screen": w["screen"], "node": w["node"]}})
    return out.close(_manifest(name, "widget_caption", "UI understanding: which human caption describes the outlined widget",
                               "widget_captions.csv, datasetId rc; node bounds from the Rico view hierarchy", True, raw),
                     workers, log)


def build_widget_find(root=DATA_ROOT, raw=RAW_ROOT, workers=6, log=print, seed=0, caps=(TRAIN_CAP, VAL_CAP)) -> dict:
    name = "ui_widget_find"
    screens, out = _widget_screens(raw, root, workers, log), _Out(root, name)
    for split, cap in zip(("train", "val"), caps):
        rng = random.Random("%s-%s-%d" % (name, split, seed))
        drawable = {sid: [w for w in ws if w["box"]] for sid, ws in screens[split].items()}
        targets = [w for ws in drawable.values() if len(ws) >= 4 for w in ws]
        out.stats[split + "_widgets_on_screens_with_4_drawable"] = len(targets)
        rng.shuffle(targets)
        for w in _passes(targets, (1, 2, 3, 4, 5, 6, 8)):
            if len(out.rows[split]) >= cap:
                break
            res = find_candidates(w, drawable[w["screen"]], rng)
            if res is None:
                out.stats[split + "_skip_under_3_distinct_widgets"] += 1
                continue
            chosen, label = res
            description = specific_caption(w["captions"])
            if description is None:
                out.stats[split + "_skip_generic_caption"] += 1
                continue
            boxes = [c["box"] for c in chosen]
            out.add(split, {"image": out.image(screen_path(w["screen"], raw), w["screen"], boxes, True),
                            "state_text": json.dumps({"description": description}, ensure_ascii=False),
                            "question": {"type": "choice", "instructions": rng.choice(INS_FIND), "criteria": _numbers(len(chosen))},
                            "label": label, "image_license": RICO_ROW_LICENSE,
                            "attribution": {"source": "Widget Captioning", "rico_screen": w["screen"], "node": w["node"]}}, boxes)
    return out.close(_manifest(name, "widget_caption", "UI grounding: which numbered widget a human caption describes",
                               "widget_captions.csv, datasetId rc; node bounds from the Rico view hierarchy", True, raw),
                     workers, log)


def build_element_type(root=DATA_ROOT, raw=RAW_ROOT, workers=6, log=print, seed=0, caps=(TRAIN_CAP, VAL_CAP)) -> dict:
    name = "ui_element_type"
    side, data, out = rico_sides(raw, root, workers, log), load_screen_annotation(raw), _Out(root, name)
    for split, cap in zip(("train", "val"), caps):
        rng = random.Random("%s-%s-%d" % (name, split, seed))
        sids = sorted((s for s in data[split] if side.get(s) == split), key=int)
        out.stats[split + "_screens"] = len(sids)
        rng.shuffle(sids)
        pools = defaultdict(list)
        for sid in sids:  # at most two elements per screen, of the two classes whose pools are smallest so far
            by_cls = defaultdict(list)
            for i in type_targets(data[split][sid]):
                by_cls[data[split][sid][i]["type"]].append(i)
            for cls in sorted(by_cls, key=lambda c: (len(pools[c]), c))[:2]:
                pools[cls].append((sid, rng.choice(by_cls[cls])))
        for p in pools.values():
            rng.shuffle(p)
        for opts, gold, (sid, i) in draw_uniform_gold(pools, rng, type_option_set, cap):
            el = data[split][sid][i]
            out.stats["%s_gold_%s" % (split, gold)] += 1
            out.add(split, {"image": out.image(screen_path(sid, raw), sid, [el["box"]], False), "state_text": None,
                            "question": {"type": "choice", "instructions": rng.choice(INS_TYPE),
                                         "criteria": [TYPE_NAMES[c] for c in opts]},
                            "label": opts.index(gold), "image_license": RICO_ROW_LICENSE,
                            "attribution": {"source": "screen_annotation", "rico_screen": sid}})
    return out.close(_manifest(name, "screen_annotation", "UI understanding: the type of the outlined element",
                               "element type + box; LABEL / MAP / KEYBOARD not asked", True, raw), workers, log)


def build_element_count(root=DATA_ROOT, raw=RAW_ROOT, workers=6, log=print, seed=0, caps=(TRAIN_CAP, VAL_CAP)) -> dict:
    name = "ui_element_count"
    side, data, out = rico_sides(raw, root, workers, log), load_screen_annotation(raw), _Out(root, name)
    for split, cap in zip(("train", "val"), caps):
        rng = random.Random("%s-%s-%d" % (name, split, seed))
        sids = sorted((s for s in data[split] if side.get(s) == split), key=int)
        rng.shuffle(sids)
        gens = {}
        for kind in COUNT_NAMES:
            pools = defaultdict(list)
            for sid in sids:
                n = sum(e["type"] == kind for e in data[split][sid])
                if n <= MAX_COUNT:
                    pools[n].append(sid)
            gens[kind] = draw_uniform_gold(pools, rng, count_option_set, 10 ** 9)
        active, limit = list(gens), max(cap // 6, 300)  # list items could fill the set alone: at most this many per type
        while active and len(out.rows[split]) < cap:  # round-robin: every type gets the same share until it runs dry
            for kind in list(active):
                got = next(gens[kind], None) if out.stats["%s_%s" % (split, kind)] < limit else None
                if got is None:
                    active.remove(kind)
                    continue
                opts, gold, sid = got
                out.stats["%s_%s" % (split, kind)] += 1
                out.add(split, {"image": out.image(screen_path(sid, raw), "screen-%s" % sid, None, False),
                                "state_text": None,
                                "question": {"type": "choice", "instructions": rng.choice(INS_COUNT) % COUNT_NAMES[kind],
                                             "criteria": [str(o) for o in opts]},
                                "label": opts.index(gold), "image_license": RICO_ROW_LICENSE, "kind": kind,
                                "attribution": {"source": "screen_annotation", "rico_screen": sid}})
                if len(out.rows[split]) >= cap:
                    break
        rng.shuffle(out.rows[split])
    return out.close(_manifest(name, "screen_annotation", "UI understanding: how many elements of a type are on the screen",
                               "element counts per type, 0-%d" % MAX_COUNT, True, raw), workers, log)


OMNIACT_VAL_ROWS = 300


def omniact_val_apps(data: Dict[str, List[dict]], target: int = OMNIACT_VAL_ROWS) -> List[str]:
    """The apps held out for val. Every OmniACT screen is in all three source splits, so a val set needs whole apps
    taken out of train. To lose as little train data as possible per val row, apps are ranked by
    (source-val tasks) / (source-train tasks + 1) (the source assigned scripts to splits at random, so this ranking
    does not look at content) and taken until ``target`` val rows are reached."""
    tr, va = Counter(t["app"] for t in data["train"]), Counter(t["app"] for t in data["val"])
    out, n = [], 0
    for app in sorted(set(tr) | set(va), key=lambda a: (-va[a] / (tr[a] + 1), a)):
        if n >= target:
            break
        out.append(app)
        n += va[app]
    return out


def build_omniact(root=DATA_ROOT, raw=RAW_ROOT, workers=6, log=print, seed=0, caps=(TRAIN_CAP, VAL_CAP)) -> dict:
    from PIL import Image

    name = "omniact_next_action"
    data, out = load_omniact(raw), _Out(root, name)
    val_apps = set(omniact_val_apps(data))
    for split, cap in zip(("train", "val"), caps):
        rng = random.Random("%s-%s-%d" % (name, split, seed))
        by_screen = defaultdict(list)
        for t in data[split]:
            if (t["app"] in val_apps) == (split == "val"):
                by_screen[t["screen"]].append(t)
        out.stats[split + "_apps"] = len({t["app"] for ts in by_screen.values() for t in ts})
        out.stats[split + "_screens"] = len(by_screen)
        for screen, tasks in sorted(by_screen.items()):
            with open(tasks[0]["boxes"]) as f:
                centers = box_centers(json.load(f))
            with Image.open(tasks[0]["image"]) as im:
                size = im.size
            centers = [c for c in centers if 0 <= c["center"][0] <= size[0] and 0 <= c["center"][1] <= size[1]]
            scripts = {}
            for t in tasks:
                scripts.setdefault(render_script(t["lines"]), t["lines"])
            popularity = Counter()
            for lines in scripts.values():
                for c in {(_box_at(l["point"], centers) or {}).get("key") for l in lines if l["point"]} - {None}:
                    popularity[c] += 1
            for t in tasks:
                gold = render_script(t["lines"])
                res = script_options(t["lines"], centers, size, [l for s, l in scripts.items() if s != gold], popularity, rng)
                if res is None:
                    out.stats[split + "_skip_no_boxed_target_or_too_few_elements"] += 1
                    continue
                opts, label, design = res
                out.stats["%s_design_%s" % (split, design)] += 1
                out.add(split, {"image": (t["image"], screen),
                                "state_text": json.dumps({"task": t["task"], "screen_size": "%dx%d" % size}, ensure_ascii=False),
                                "question": {"type": "choice", "instructions": rng.choice(INS_SCRIPT), "criteria": opts},
                                "label": label, "image_license": OMNIACT_ROW_LICENSE, "kind": design,
                                "attribution": {"source": "OmniACT", "app": t["app"], "screen": screen}})
        rng.shuffle(out.rows[split])
        del out.rows[split][cap:]
        for r in out.rows[split]:  # one stored image per screen, only for screens that kept a row
            r["image"] = out.image(r["image"][0], r["image"][1], None, False)
    m = _manifest(name, "omniact", "GUI agent: which script carries out the task on this desktop / web screen",
                  "train.json / val.json tasks; test.json untouched", False, raw)
    m["splits"] = ("OmniACT's own splits share every screen, so whole apps are held out for val (ranked by source-val / "
                   "source-train task ratio until %d val rows): train = source-train tasks of the other apps, val = "
                   "source-val tasks of the val apps; source test untouched" % OMNIACT_VAL_ROWS)
    m["val_apps"] = sorted(val_apps)
    return out.close(m, workers, log)


def _split_rows(root: str, name: str) -> Dict[str, List[dict]]:
    out = {}
    for split in ("train", "val"):
        with open(os.path.join(root, name, split + ".jsonl"), encoding="utf-8") as f:
            out[split] = [json.loads(l) for l in f]
    return out


def cross_dedupe(name: str, others: Sequence[str], root: str = DATA_ROOT, workers: int = 6, log=print) -> dict:
    """Drop rows of ``name`` whose image matches, under the data validator's own test (dHash <= 8 bits and thumbnail
    correlation >= 0.98, or equal bytes), an image on the OTHER side of any of ``others``: train rows against their
    val images, val rows against their train images. The screen sides already keep raw screens apart; this catches
    the few pairs that a drawn box pushes over the threshold. Unreferenced images are deleted; the manifest records
    what was dropped."""
    from multiprocessing import Pool

    base = os.path.join(root, name)
    rows = _split_rows(root, name)
    others = [o for o in others if o != name and os.path.exists(os.path.join(root, o, "val.jsonl"))]
    dropped = {}
    for split, opp in (("train", "val"), ("val", "train")):
        mine = sorted({os.path.normpath(os.path.join(base, r["image"])) for r in rows[split]})
        theirs = sorted({os.path.normpath(os.path.join(root, o, r["image"])) for o in others for r in _split_rows(root, o)[opp]})
        if not mine or not theirs:
            dropped[split] = 0
            continue
        paths = mine + theirs
        with Pool(workers) as pool:
            sig = pool.map(_signature, paths, chunksize=128)
        import numpy as np

        thumbs = np.frombuffer(b"".join(x[2] for x in sig), dtype=np.uint8).reshape(len(sig), -1)
        pairs = near_pairs([x[0] for x in sig], [x[1] for x in sig], thumbs, _dv().DHASH_MAX_DISTANCE, _dv().MIN_THUMB_CORRELATION)
        n = len(mine)
        bad = {paths[i] for i, j in pairs if i < n <= j} | {paths[j] for i, j in pairs if j < n <= i}
        before = len(rows[split])
        rows[split] = [r for r in rows[split] if os.path.normpath(os.path.join(base, r["image"])) not in bad]
        dropped[split] = before - len(rows[split])
    for split, rs in rows.items():
        with open(os.path.join(base, split + ".jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(dict(r, id="%s-%s-%d" % (name, split, i)), ensure_ascii=False) + "\n" for i, r in enumerate(rs))
    used = {r["image"] for rs in rows.values() for r in rs}
    removed = 0
    for fn in os.listdir(os.path.join(base, "images")):
        if "images/" + fn not in used:
            os.remove(os.path.join(base, "images", fn))
            removed += 1
    mpath = os.path.join(base, "manifest.json")
    with open(mpath) as f:
        m = json.load(f)
    m.setdefault("cross_dedupe", []).append({"against": others, "dropped": dropped, "images_removed": removed,
                                             "match": "sha256 or dHash<=8 bits + 64px thumbnail corr>=0.98 (train vs their val, val vs their train)"})
    m["counts"] = {s: len(rs) for s, rs in rows.items()}
    with open(mpath, "w") as f:
        json.dump(m, f, indent=2)
    log("%s cross-dedupe against %s: dropped %s, %d images removed" % (name, others, dropped, removed))
    return m


BUILDERS = {"ui_refexp": build_refexp, "ui_widget_caption": build_widget_caption, "ui_widget_find": build_widget_find,
            "ui_element_type": build_element_type, "ui_element_count": build_element_count,
            "omniact_next_action": build_omniact}


# ---- a learned question-only baseline --------------------------------------------------------------------------------

def _words(s) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


def _text_features(rec: dict, dim: int):
    """Hashed features of each option as a text-only model could see it: its words and bigrams, its words crossed with
    the words of the state / question, its length rank, its numeric rank and value, its position, and how similar it
    is to the other options (the odd one out / the "centre" of the set)."""
    import zlib

    import numpy as np

    q = rec["question"]
    crit = q["criteria"]
    opts = ["%s: %s" % (k, v) if v else str(k) for k, v in crit.items()] if isinstance(crit, dict) else [str(o) for o in crit]
    ctx = list(dict.fromkeys(_words(rec.get("state_text") or "") + _words(q["instructions"])))[:48]
    toks = [_words(o)[:40] for o in opts]
    sets = [set(t) for t in toks]
    by_len = sorted(range(len(opts)), key=lambda i: len(opts[i]))
    sim = [sum(len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for j in range(len(opts)) if j != i) for i in range(len(opts))]
    by_sim = sorted(range(len(opts)), key=lambda i: sim[i])
    try:
        vals = [float(o) for o in opts]
        by_val = sorted(range(len(opts)), key=lambda i: vals[i])
    except ValueError:
        by_val = None
    out = []
    for i, o in enumerate(opts):
        f = ["pos%d" % i, "len%d" % by_len.index(i), "sim%d" % by_sim.index(i), "ov%d" % min(6, len(sets[i] & set(ctx)))]
        if by_val is not None:
            f += ["rank%d/%d" % (by_val.index(i), len(opts)), "val" + o] + ["vx:%s|%s" % (c, o) for c in ctx]
        f += ["o:" + t for t in sets[i]] + ["b:%s_%s" % (a, b) for a, b in zip(toks[i], toks[i][1:])]
        f += ["x:%s|%s" % (c, t) for c in ctx for t in sorted(sets[i])][:400]
        out.append(np.unique(np.array([zlib.crc32(x.encode()) % dim for x in f], dtype=np.int64)))
    return out


def text_only_baseline(name: str, root: str = DATA_ROOT, epochs: int = 4, dim: int = 1 << 18, seed: int = 0,
                       max_train: int = 15000) -> dict:
    """Train a linear scorer on text alone (state, question, options; never the image) on train, report its accuracy
    on val next to chance. A set built without text give-aways stays at chance."""
    import numpy as np

    def load(split):
        with open(os.path.join(root, name, split + ".jsonl")) as f:
            rows = [json.loads(l) for l in f]
        return [(_text_features(r, dim), r["label"]) for r in rows if r["question"]["type"] == "choice"]

    train, val = load("train")[:max_train], load("val")
    w = np.zeros(dim, dtype=np.float32)
    rng = random.Random(seed)

    def scores(feats):
        s = np.array([w[ix].sum() for ix in feats], dtype=np.float64)
        p = np.exp(s - s.max())
        return p / p.sum()

    for epoch in range(epochs):
        rng.shuffle(train)
        lr = 0.05 / (1 + epoch)
        for feats, label in train:
            p = scores(feats)
            for i, ix in enumerate(feats):
                w[ix] += lr * ((i == label) - p[i])
    acc = lambda rows: round(sum(int(np.argmax(scores(f))) == lab for f, lab in rows) / max(1, len(rows)), 3)  # noqa: E731
    return {"n_train": len(train), "n_val": len(val), "chance": round(sum(1 / len(f) for f, _ in val) / max(1, len(val)), 3),
            "text_only_val": acc(val), "text_only_train": acc(train)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "build", "cross-dedupe", "text-only"])
    ap.add_argument("names", nargs="*", default=["all"])
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--raw", default=RAW_ROOT)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    names = list(BUILDERS) if a.names in ([], ["all"]) else a.names
    if a.command == "fetch":
        fetch(a.raw)
        extract_rico(a.raw, a.root, a.workers)
    elif a.command in ("build", "cross-dedupe"):
        if a.command == "build":
            for n in names:
                BUILDERS[n](a.root, a.raw, a.workers)
        for n in names:  # then against every other set of the family and rico_screenqa (their sides were set jointly)
            cross_dedupe(n, [o for o in list(BUILDERS) + ["rico_screenqa"] if o != n], a.root, a.workers)
    else:
        for n in names:
            print(n, json.dumps(text_only_baseline(n, a.root)), flush=True)


if __name__ == "__main__":
    main()
