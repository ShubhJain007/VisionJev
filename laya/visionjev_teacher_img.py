"""Teacher hard questions with the state rendered as page images (``teacher_hard_img``).

The teacher sets (``teacher_hard``, ``teacher_hard2``: visionjev_teacher.py) are text only. Here every state short enough
to stay legible is drawn as one or two page images (free fonts, wrapped text, JSON states as indented key / value lines),
and its questions, labels, soft targets and rationales are kept unchanged: the same hard reasoning, but the state has to
be read from pixels. States that would need more than ``MAX_PAGES`` pages stay text only.

    python -m laya.visionjev_teacher_img            # -> data/vqa/teacher_hard_img
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter
from typing import List

from .visionjev_data import DATA_ROOT
from .visionjev_synth import _font

NAME = "teacher_hard_img"
SOURCES = ("teacher_hard", "teacher_hard2")
PAGE_W, PAGE_H, MARGIN = 704, 832, 18  # 704 x 832 = 585,728 px: one page fits the backbone's 589,824-pixel budget unscaled
MAX_PAGES = 2


def state_lines(state_text: str) -> List[str]:
    """The state as display lines: JSON objects become indented ``key: value`` lines, text keeps its own line breaks."""
    try:
        obj = json.loads(state_text)
    except ValueError:
        obj = None
    if not isinstance(obj, (dict, list)):
        return state_text.replace("\t", "    ").split("\n")
    out = []

    def walk(v, depth, key=None):
        pad = "    " * depth
        head = pad + ("%s: " % key if key is not None else "")
        if isinstance(v, dict):
            if key is not None:
                out.append(head.rstrip())
            for k, x in v.items():
                walk(x, depth + (key is not None), k)
        elif isinstance(v, list):
            if all(not isinstance(x, (dict, list)) for x in v) and len(json.dumps(v)) < 90:
                out.append(head + ", ".join(str(x) for x in v))
                return
            if key is not None:
                out.append(head.rstrip())
            for x in v:
                if isinstance(x, (dict, list)):
                    out.append(pad + "    -")
                    walk(x, depth + 2)
                else:
                    out.append(pad + "    - " + str(x))
        else:
            out.append(head + ("" if v is None else str(v)))

    walk(obj, 0)
    flat = []
    for line in out:  # a string value with its own line breaks: one display line each, at the key's indentation
        first, *rest = line.replace("\t", "    ").split("\n")
        pad = first[:len(first) - len(first.lstrip(" "))] + "    "
        flat.append(first)
        flat.extend(pad + x.strip() for x in rest if x.strip())
    return flat


def wrap(lines: List[str], font, width: int) -> List[str]:
    """Greedy word wrap to ``width`` pixels; a wrapped line keeps its indentation."""
    out = []
    for line in lines:
        if font.getlength(line) <= width:
            out.append(line)
            continue
        indent = line[:len(line) - len(line.lstrip(" "))]
        cur = ""
        for word in line.lstrip(" ").split(" "):
            trial = cur + " " + word if cur else indent + word
            if cur and font.getlength(trial) > width:
                out.append(cur)
                cur = indent + word
            else:
                cur = trial
            while font.getlength(cur) > width:  # one unbreakable token wider than the page
                cut = max(len(indent) + 1, int(len(cur) * width / font.getlength(cur)) - 1)
                out.append(cur[:cut])
                cur = indent + cur[cut:]
        out.append(cur)
    return out


def render_pages(state_text: str, rng: random.Random):
    """A list of PIL page images, or None if the state needs more than MAX_PAGES pages at the smallest font."""
    from PIL import Image, ImageDraw

    family = rng.randrange(4)
    lines = state_lines(state_text)
    for size in (15, 14, 13):
        font = _font(size, family=family)
        step = size + 4
        per_page = (PAGE_H - 2 * MARGIN) // step
        wrapped = wrap(lines, font, PAGE_W - 2 * MARGIN)
        if len(wrapped) <= per_page * MAX_PAGES:
            break
    else:
        return None
    bg = rng.choice([(255, 255, 255), (252, 252, 248), (248, 249, 251)])
    pages = []
    for p in range(0, len(wrapped), per_page):
        chunk = wrapped[p:p + per_page]
        img = Image.new("RGB", (PAGE_W, max(160, 2 * MARGIN + step * len(chunk))), bg)
        d = ImageDraw.Draw(img)
        for i, line in enumerate(chunk):
            d.text((MARGIN, MARGIN + i * step), line, font=font, fill=(20, 20, 20))
        pages.append(img)
    return pages


def build(root: str = DATA_ROOT, sources=SOURCES, log=print):
    tmp, final = os.path.join(root, ".tmp-" + NAME), os.path.join(root, NAME)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    stats, counts, rendered = Counter(), {}, {}
    for split in ("train", "val"):
        n = 0
        with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
            for src in sources:
                p = os.path.join(root, src, split + ".jsonl")
                if not os.path.exists(p):
                    continue
                for line in open(p):
                    r = json.loads(line)
                    key = "%s-%s" % (src, r["provenance"]["doc_id"])
                    if key not in rendered:
                        rng = random.Random(int(hashlib.md5(key.encode()).hexdigest(), 16))
                        pages = render_pages(r["state_text"], rng)
                        rendered[key] = []
                        for i, img in enumerate(pages or []):
                            rel = "images/%s-p%d.jpg" % (key, i + 1)
                            img.save(os.path.join(tmp, rel), quality=92)
                            rendered[key].append(rel)
                        stats["states_%d_pages" % len(rendered[key])] += 1
                    paths = rendered[key]
                    if not paths:
                        stats["rows_left_text_only"] += 1
                        continue
                    rec = {k: v for k, v in r.items() if k not in ("id", "state_text")}
                    rec.update({"id": "%s-%s-%d" % (NAME, split, n), "state_text": None})
                    rec["provenance"] = dict(r["provenance"], rendered_from=src)
                    if len(paths) == 1:
                        rec["image"] = paths[0]
                    else:
                        rec["images"] = paths
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    stats["rows_%s" % src] += 1
        counts[split] = n
    json.dump({"dataset": NAME, "source": "teacher_hard + teacher_hard2 (visionjev_teacher.py), states rendered as page images",
               "license": {"annotations": "outputs of Apache-2.0 open-weight models (see provenance)",
                           "images": "rendered in-house with DejaVu / Liberation / Lato free fonts", "commercial": True,
                           "verified": "Qwen3.6-27B / Qwen3.6-35B-A3B / gpt-oss-20b HF cards, Apache-2.0"},
               "labels": "writer's intended answer, confirmed by every answerer (same rows as the text sets)",
               "eval_only": False, "counts": counts, "stats": dict(stats),
               "page": {"width": PAGE_W, "max_height": PAGE_H, "max_pages": MAX_PAGES}},
              open(os.path.join(tmp, "manifest.json"), "w"), indent=2)
    open(os.path.join(tmp, "_READY"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    log("%s: %s %s" % (NAME, counts, dict(stats)))
    return counts, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    build(a.root)
