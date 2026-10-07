"""Never-seen use cases (eval only): procedurally generated decision tasks with exact labels, from task families that
appear in NO training set, to measure how the model handles a use case nobody trained it on.

``novel_text`` (text, noul / choice), 8 families:
* meeting_overlap  two meetings (day + start / end) -> do they overlap?
* password_policy  a policy (length, digit, uppercase, symbol) and a password -> does it comply?
* json_valid       a short JSON-like snippet -> is it valid JSON?
* list_sorted      a list of numbers -> is it sorted ascending?
* weekday          a date -> which weekday? (choice)
* log_errors       service log lines -> which service logged the most errors? (choice)
* stock_enough     inventory + an order -> can the order be fulfilled from stock?
* unit_convert     "is 2.5 km more than 2300 m?" style comparisons across units

``novel_vision`` (image, choice / noul), 5 families, drawn with PIL:
* clock            an analogue clock -> what time does it show? (choice, 4 options, 5-minute steps)
* shape_count      coloured shapes -> how many <colour> <shape>s? (choice 0-5)
* progress_bar     a progress bar -> roughly what percentage? (choice of 4 ranges, value away from bin edges)
* dice             one die face -> which number? (choice 1-6)
* swatch           a colour swatch -> which colour name? (choice among 4 well-separated colours)

    python -m laya.visionjev_novel --n 150
"""
import argparse
import datetime
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict

from .visionjev_data import DATA_ROOT

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _noul(q):
    return {"type": "noul", "instructions": q, "criteria": None}


def _choice(q, opts, gold, rng):
    opts = list(opts)
    rng.shuffle(opts)
    return {"type": "choice", "instructions": q, "criteria": opts}, opts.index(gold)


# ---- text -----------------------------------------------------------------------------------------------------------


def meeting_overlap(rng):
    day = rng.choice(DAYS[:5])
    s1 = rng.randint(8, 16) * 60 + rng.choice([0, 15, 30, 45])
    e1 = s1 + rng.choice([30, 45, 60, 90])
    if rng.random() < 0.5:  # overlap
        s2 = rng.randint(s1 - 45, e1 - 15)
    else:
        s2 = rng.choice([e1 + rng.choice([0, 15, 30]), s1 - rng.choice([60, 90, 120])])
    e2 = s2 + rng.choice([30, 45, 60])
    if s2 < s1 and e2 > s1 and rng.random() < 0.5:
        pass
    fmt = lambda m: "%02d:%02d" % (m // 60, m % 60)  # noqa: E731
    lab = int(s1 < e2 and s2 < e1)
    state = {"meeting_a": {"day": day, "start": fmt(s1), "end": fmt(e1)},
             "meeting_b": {"day": day, "start": fmt(s2), "end": fmt(e2)}}
    return state, _noul("Do these two meetings overlap in time?"), lab


def password_policy(rng):
    n = rng.randint(8, 14)
    policy = {"min_length": n, "needs_digit": rng.random() < 0.7, "needs_uppercase": rng.random() < 0.7,
              "needs_symbol": rng.random() < 0.5}
    ok = rng.random() < 0.5
    chars = "abcdefghijkmnpqrstuvwxyz"
    pw = [rng.choice(chars) for _ in range(n + rng.randint(0, 3))]
    if policy["needs_digit"]:
        pw[rng.randrange(len(pw))] = rng.choice("23456789")
    if policy["needs_uppercase"]:
        pw[rng.randrange(len(pw))] = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    if policy["needs_symbol"]:
        pw[rng.randrange(len(pw))] = rng.choice("!@#$%&*?")
    if not ok:  # break exactly one rule
        rules = ["min_length"] + [k for k in ("needs_digit", "needs_uppercase", "needs_symbol") if policy[k]]
        r = rng.choice(rules)
        if r == "min_length":
            pw = pw[:n - rng.randint(1, 3)]
        elif r == "needs_digit":
            pw = [c if not c.isdigit() else rng.choice(chars) for c in pw]
        elif r == "needs_uppercase":
            pw = [c.lower() for c in pw]
        else:
            pw = [c if c not in "!@#$%&*?" else rng.choice(chars) for c in pw]
    pw = "".join(pw)
    lab = int(len(pw) >= n and (not policy["needs_digit"] or any(c.isdigit() for c in pw))
              and (not policy["needs_uppercase"] or any(c.isupper() for c in pw))
              and (not policy["needs_symbol"] or any(c in "!@#$%&*?" for c in pw)))
    return {"password_policy": policy, "password": pw}, _noul("Does this password satisfy the policy?"), lab


def json_valid(rng):
    obj = {rng.choice(["id", "name", "count", "active", "tags"]): rng.choice([1, "x", True, [1, 2]]) for _ in range(3)}
    s = json.dumps(obj)
    lab = 1
    if rng.random() < 0.5:
        lab = 0
        s = rng.choice([s[:-1], s.replace('"', "'", 2), s.replace(",", "", 1) if "," in s else s[:-1],
                        s.replace(":", "", 1), s + ","])
        try:
            json.loads(s)
            lab = 1
        except ValueError:
            lab = 0
    return {"snippet": s}, _noul("Is this snippet valid JSON?"), lab


def list_sorted(rng):
    xs = sorted(rng.sample(range(1, 200), rng.randint(5, 9)))
    if rng.random() < 0.5:
        i = rng.randrange(len(xs) - 1)
        xs[i], xs[i + 1] = xs[i + 1], xs[i]
    return {"numbers": xs}, _noul("Is this list sorted in ascending order?"), int(xs == sorted(xs))


def weekday(rng):
    d = datetime.date(2020, 1, 1) + datetime.timedelta(days=rng.randrange(2500))
    gold = DAYS[d.weekday()]
    opts = [gold] + rng.sample([x for x in DAYS if x != gold], 3)
    q, lab = _choice("Which day of the week is this date?", opts, gold, rng)
    return {"date": d.isoformat()}, q, lab


def log_errors(rng):
    svcs = rng.sample(["auth", "billing", "search", "checkout", "email", "gateway", "inventory"], 4)
    counts = {s: rng.randint(0, 4) for s in svcs}
    top = rng.choice(svcs)
    counts[top] = max(counts.values()) + rng.randint(1, 3)
    lines = []
    for s in svcs:
        lines += ["%s ERROR request failed" % s] * counts[s] + ["%s INFO ok" % s] * rng.randint(1, 4)
    rng.shuffle(lines)
    ts = ["2026-03-%02d 10:%02d:%02d %s" % (rng.randint(1, 28), i % 60, rng.randint(0, 59), l) for i, l in enumerate(lines)]
    q, lab = _choice("Which service logged the most errors?", svcs, top, rng)
    return {"log": "\n".join(ts)}, q, lab


def stock_enough(rng):
    items = rng.sample(["widget", "bolt", "panel", "cable", "sensor", "valve"], 3)
    stock = {i: rng.randint(0, 40) for i in items}
    order = {i: rng.randint(1, 30) for i in items}
    if rng.random() < 0.5:
        for i in items:
            order[i] = min(order[i], stock[i]) if stock[i] > 0 else 0
        order = {k: v for k, v in order.items() if v > 0} or {items[0]: 0}
    lab = int(all(order[i] <= stock.get(i, 0) for i in order))
    return {"stock": stock, "order": order}, _noul("Can this order be fulfilled from current stock?"), lab


UNITS = {"m": 1, "km": 1000, "cm": 0.01, "g": 1, "kg": 1000, "mg": 0.001, "s": 1, "min": 60, "h": 3600}
GROUPS = [("m", "km", "cm"), ("g", "kg", "mg"), ("s", "min", "h")]


def unit_convert(rng):
    g = rng.choice(GROUPS)
    u1, u2 = rng.sample(g, 2)
    a = round(rng.uniform(0.5, 50), 1)
    base = a * UNITS[u1]
    factor = rng.choice([0.6, 0.8, 1.25, 1.6])
    b = round(base * factor / UNITS[u2], 3)
    lab = int(a * UNITS[u1] > b * UNITS[u2])
    return {"quantity_a": "%g %s" % (a, u1), "quantity_b": "%g %s" % (b, u2)}, _noul("Is quantity A larger than quantity B?"), lab


TEXT_FAMILIES = [meeting_overlap, password_policy, json_valid, list_sorted, weekday, log_errors, stock_enough, unit_convert]


# ---- vision ---------------------------------------------------------------------------------------------------------


def _canvas(rng, size=320):
    from PIL import Image

    bg = rng.choice([(255, 255, 255), (245, 240, 230), (230, 236, 245)])
    return Image.new("RGB", (size, size), bg)


def clock(rng):
    from PIL import ImageDraw

    img = _canvas(rng)
    d = ImageDraw.Draw(img)
    c, r = 160, 130
    d.ellipse((c - r, c - r, c + r, c + r), outline=(20, 20, 20), width=4)
    for k in range(12):
        a = math.radians(k * 30)
        d.line((c + 0.85 * r * math.sin(a), c - 0.85 * r * math.cos(a), c + r * math.sin(a), c - r * math.cos(a)),
               fill=(20, 20, 20), width=4 if k % 3 == 0 else 2)
    h, m = rng.randint(1, 12), rng.choice(range(0, 60, 5))
    am = math.radians(m * 6)
    ah = math.radians((h % 12) * 30 + m * 0.5)
    d.line((c, c, c + 0.8 * r * math.sin(am), c - 0.8 * r * math.cos(am)), fill=(20, 20, 20), width=4)
    d.line((c, c, c + 0.5 * r * math.sin(ah), c - 0.5 * r * math.cos(ah)), fill=(20, 20, 20), width=8)
    gold = "%d:%02d" % (h, m)
    others = set()
    while len(others) < 3:
        hh, mm = h, m
        if rng.random() < 0.5:
            hh = (h + rng.choice([-1, 1, 3, 6]) - 1) % 12 + 1
        else:
            mm = (m + rng.choice([15, 25, 30, 40])) % 60
        t = "%d:%02d" % (hh, mm)
        if t != gold:
            others.add(t)
    q, lab = _choice("What time does the clock show?", [gold] + sorted(others), gold, rng)
    return img, q, lab


COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 70), "yellow": (240, 200, 30),
          "purple": (140, 60, 180), "orange": (245, 130, 30)}


def shape_count(rng):
    from PIL import ImageDraw

    img = _canvas(rng)
    d = ImageDraw.Draw(img)
    target_c, target_s = rng.choice(list(COLORS)), rng.choice(["circle", "square"])
    n = rng.randint(0, 5)
    items = [(target_c, target_s)] * n
    for _ in range(rng.randint(2, 6)):
        c, s = rng.choice(list(COLORS)), rng.choice(["circle", "square"])
        if (c, s) != (target_c, target_s):
            items.append((c, s))
    rng.shuffle(items)
    cells = rng.sample(range(25), len(items))
    for (c, s), cell in zip(items, cells):
        x, y = 16 + (cell % 5) * 60, 16 + (cell // 5) * 60
        box = (x, y, x + 44, y + 44)
        (d.ellipse if s == "circle" else d.rectangle)(box, fill=COLORS[c])
    q, lab = _choice("How many %s %ss are in the image?" % (target_c, target_s), [str(i) for i in range(6)], str(n), rng)
    return img, q, lab


def progress_bar(rng):
    from PIL import ImageDraw

    img = _canvas(rng)
    d = ImageDraw.Draw(img)
    bins = ["0-25%", "25-50%", "50-75%", "75-100%"]
    b = rng.randrange(4)
    v = rng.uniform(b * 25 + 4, b * 25 + 21) / 100
    d.rectangle((30, 140, 290, 180), outline=(40, 40, 40), width=3)
    d.rectangle((33, 143, 33 + int(254 * v), 177), fill=rng.choice(list(COLORS.values())))
    q, lab = _choice("Roughly how full is the progress bar?", bins, bins[b], rng)
    return img, q, lab


PIPS = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)], 4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
        5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)], 6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)]}


def dice(rng):
    from PIL import ImageDraw

    img = _canvas(rng)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((80, 80, 240, 240), radius=22, fill=(250, 250, 250), outline=(30, 30, 30), width=4)
    n = rng.randint(1, 6)
    for px, py in PIPS[n]:
        cx, cy = 160 + px * 45, 160 + py * 45
        d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=(20, 20, 20))
    q, lab = _choice("What number does the die show?", [str(i) for i in range(1, 7)], str(n), rng)
    return img, q, lab


def swatch(rng):
    from PIL import ImageDraw

    img = _canvas(rng)
    name = rng.choice(list(COLORS))
    ImageDraw.Draw(img).rectangle((60, 60, 260, 260), fill=COLORS[name])
    opts = [name] + rng.sample([c for c in COLORS if c != name], 3)
    q, lab = _choice("What colour is the square?", opts, name, rng)
    return img, q, lab


VISION_FAMILIES = [clock, shape_count, progress_bar, dice, swatch]


def _write(root, name, rows, extra_manifest):
    tmp, final = os.path.join(root, ".tmp-" + name), os.path.join(root, name)
    counts, fams = Counter(), Counter()
    with open(os.path.join(tmp, "val.jsonl"), "w") as f:
        for i, rec in enumerate(rows):
            fams[rec["id"].rsplit("-", 1)[-1]] += 1
            f.write(json.dumps(rec) + "\n")
    m = {"dataset": name, "source": "generated (VisionJev visionjev_novel): task families absent from all training data",
         "revision": "v1", "license": {"annotations": "generated in-house, exact labels", "images": "generated", "commercial": True,
                                       "verified": "generated in-house"},
         "eval_only": True, "counts": {"val": len(rows)}, "families": dict(fams),
         "label_counts": {"val": dict(Counter(str(r["label"]) for r in rows))}, **extra_manifest}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    print("%s: %d rows %s" % (name, len(rows), dict(fams)), flush=True)


def build(root=DATA_ROOT, n=150, seed=0):
    rng = random.Random(seed)
    for name in ("novel_text", "novel_vision"):
        shutil.rmtree(os.path.join(root, ".tmp-" + name), ignore_errors=True)
    os.makedirs(os.path.join(root, ".tmp-novel_text"))
    rows = []
    for fam in TEXT_FAMILIES:
        for k in range(n):
            state, q, lab = fam(rng)
            rows.append({"id": "novel_text-val-%d-%s" % (len(rows), fam.__name__), "state_text": json.dumps(state),
                         "question": q, "label": lab})
    _write(root, "novel_text", rows, {"validator": {"noul_min_minority": 0.05}})
    os.makedirs(os.path.join(root, ".tmp-novel_vision", "images"))
    rows = []
    for fam in VISION_FAMILIES:
        for k in range(n):
            img, q, lab = fam(rng)
            p = "images/%s-%04d.png" % (fam.__name__, k)
            img.save(os.path.join(root, ".tmp-novel_vision", p))
            rows.append({"id": "novel_vision-val-%d-%s" % (len(rows), fam.__name__), "image": p, "state_text": None,
                         "question": q, "label": lab, "image_license": "generated (VisionJev)"})
    _write(root, "novel_vision", rows, {"validator": {"skip_content_leakage": True}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--n", type=int, default=150, help="questions per family")
    a = ap.parse_args()
    build(a.root, a.n)
