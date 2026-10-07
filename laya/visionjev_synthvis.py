"""Generated visual decisions with exact labels: charts / dashboards, and photo conditions.

* ``chart_qa``        (choice / noul / score) matplotlib charts rendered from random data in many styles (bar,
                      horizontal bar, line, pie, two-panel dashboards; light / dark themes, fonts, sizes, gridlines):
                      which category is highest / lowest, is A greater than B, did the series end higher than it
                      started, roughly how large is X (5 bins). Every question is asked only with a clear margin
                      (>= 8% of the range) so the answer is readable, and the label comes from the data.
* ``chart_score``     the "how large is X" score questions only (5 levels), from other charts (seed 1).
* ``photo_condition`` (choice, two photos) the same Open Images photo with and without a condition (fog, darkness,
                      motion blur, rain streaks, glare / over-exposure, low resolution): "which photo is foggier?".
                      Pairs make the label exact even when the original photo already shows some of it. Photos keep
                      their ``oi_*`` split (md5 % 100 < 5 -> val), so no photo crosses train / val.

    python -m laya.visionjev_synthvis --charts 12000 --conditions 6000
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict

from .visionjev_data import DATA_ROOT

CATS = [["North", "South", "East", "West", "Central"], ["Q1", "Q2", "Q3", "Q4"],
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
        ["Search", "Email", "Social", "Direct", "Referral"], ["Web", "iOS", "Android", "Desktop"],
        ["Apples", "Pears", "Plums", "Cherries", "Grapes", "Kiwis"], ["Team A", "Team B", "Team C", "Team D"],
        ["Tokyo", "Paris", "Lagos", "Lima", "Oslo", "Delhi"], ["Bronze", "Silver", "Gold", "Platinum"]]
METRICS = ["Revenue ($k)", "Signups", "Tickets", "Latency (ms)", "Orders", "Errors", "Users (k)", "Sales", "Visits",
           "Returns"]
LEVELS = ["very low", "low", "medium", "high", "very high"]


def _style(rng):
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    style = rng.choice(["default", "ggplot", "seaborn-v0_8-whitegrid", "dark_background", "bmh", "fivethirtyeight",
                        "seaborn-v0_8-pastel", "grayscale"])
    plt.style.use(style)
    plt.rcParams["font.family"] = rng.choice(["DejaVu Sans", "DejaVu Serif", "Liberation Sans", "Liberation Serif"])
    plt.rcParams["font.size"] = rng.choice([8, 9, 10, 11, 12])
    return style


def _chart(rng, path):
    """Draw one chart; return (facts, kind). facts: {cats, vals, metric, kind, title}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(rng)
    cats = list(rng.choice(CATS))
    n = rng.randint(3, len(cats))
    cats = cats[:n] if cats[0] in ("Q1", "Mon", "Jan") else rng.sample(cats, n)
    hi = rng.choice([10, 50, 100, 500, 1000, 5000])
    vals = [round(rng.uniform(0.05, 1.0) * hi, 1 if hi <= 50 else 0) for _ in cats]
    metric = rng.choice(METRICS)
    kind = rng.choice(["bar", "barh", "line", "pie", "bar", "line"])
    if kind == "pie":
        vals = [abs(v) + hi * 0.05 for v in vals]
    w, h = rng.choice([(6, 4), (5, 4), (7, 3.5), (4.5, 4.5), (8, 4.5)])
    fig, ax = plt.subplots(figsize=(w, h), dpi=rng.choice([80, 100, 120]))
    if kind == "bar":
        ax.bar(cats, vals, color=rng.choice([None, "tab:blue", "tab:green", "tab:orange", "tab:purple"]))
    elif kind == "barh":
        ax.barh(cats, vals)
    elif kind == "line":
        ax.plot(cats, vals, marker=rng.choice(["o", "s", "", "^"]))
    else:
        ax.pie(vals, labels=cats, autopct=rng.choice([None, "%1.0f%%"]))
    if kind != "pie":
        (ax.set_xlabel if kind == "barh" else ax.set_ylabel)(metric)
        if rng.random() < 0.5:
            ax.grid(True, alpha=0.3)
    title = rng.choice(["%s by segment" % metric, "Weekly report", "Dashboard", "%s" % metric, ""])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="jpg", pil_kwargs={"quality": 88})
    plt.close(fig)
    return {"cats": cats, "vals": vals, "metric": metric, "kind": kind}


def _chart_questions(f, rng):
    cats, vals, kind = f["cats"], f["vals"], f["kind"]
    rngv = max(vals) - min(vals)
    margin = 0.08 * max(rngv, 1e-9)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    qs = []
    if vals[order[-1]] - vals[order[-2]] >= margin:
        qs.append(({"type": "choice", "instructions": "Which category has the highest value in the chart?",
                    "criteria": cats}, order[-1]))
    if vals[order[1]] - vals[order[0]] >= margin and kind != "pie":
        qs.append(({"type": "choice", "instructions": "Which category has the lowest value in the chart?",
                    "criteria": cats}, order[0]))
    a, b = rng.sample(range(len(cats)), 2)
    if abs(vals[a] - vals[b]) >= margin:
        qs.append(({"type": "noul", "instructions": "Is the value for %s greater than the value for %s?" % (cats[a], cats[b]),
                    "criteria": None}, int(vals[a] > vals[b])))
    if kind == "line" and abs(vals[-1] - vals[0]) >= margin:
        qs.append(({"type": "noul", "instructions": "Does the series end higher than it starts?", "criteria": None},
                    int(vals[-1] > vals[0])))
    if kind in ("bar", "barh", "line") and rngv > 0:
        i = rng.randrange(len(cats))
        pos = (vals[i] - min(vals)) / rngv
        lvl = min(4, int(pos * 5))
        edge = abs(pos * 5 - round(pos * 5))
        if edge >= 0.15 or pos in (0.0, 1.0):  # not on a bin boundary
            qs.append(({"type": "score", "instructions": "Relative to the other categories in this chart, how large is %s?"
                        % cats[i], "criteria": LEVELS}, lvl))
    return qs


def build_charts(root, n, seed=0, name="chart_qa", score_only=False):
    """``score_only``: keep only the score question ("how large is X", 5 levels) -> ``chart_score``."""
    tmp = os.path.join(root, ".tmp-" + name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    rng = random.Random(seed)
    files = {s: open(os.path.join(tmp, s + ".jsonl"), "w") for s in ("train", "val")}
    counts, labels = Counter(), defaultdict(Counter)
    for i in range(n):
        p = "images/chart-%06d.jpg" % i
        f = _chart(rng, os.path.join(tmp, p))
        qs = _chart_questions(f, rng)
        if score_only:
            qs = [x for x in qs if x[0]["type"] == "score"]
        if not qs:
            os.remove(os.path.join(tmp, p))
            continue
        q, lab = rng.choice(qs)
        split = "val" if i % 10 == 0 else "train"
        if score_only and labels[split]["score:%s" % lab] >= (2600 if split == "train" else 260):
            os.remove(os.path.join(tmp, p))  # extremes (the min / max category) qualify far more often: cap each level
            continue
        rec = {"id": "%s-%s-%d" % (name, split, counts[split]), "image": p, "state_text": None, "question": q,
               "label": lab, "image_license": "generated (VisionJev)"}
        files[split].write(json.dumps(rec) + "\n")
        counts[split] += 1
        labels[split]["%s:%s" % (q["type"], lab)] += 1
    for fh in files.values():
        fh.close()
    _finish(root, name, tmp, counts, labels, "generated charts (matplotlib), exact labels from the data",
            {"validator": {"skip_content_leakage": True}})


def _condition(img, cond, rng):
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    w, h = img.size
    if cond == "fog":
        haze = Image.new("RGB", img.size, (220, 220, 225))
        return Image.blend(img, haze, rng.uniform(0.45, 0.65)).filter(ImageFilter.GaussianBlur(1.2))
    if cond == "dark":
        return ImageEnhance.Brightness(img).enhance(rng.uniform(0.18, 0.35))
    if cond == "motion_blur":
        k = rng.choice([9, 11, 13])
        out = img
        for d in range(1, k // 2 + 1):  # a horizontal smear
            out = Image.blend(out, img.transform(img.size, Image.AFFINE, (1, 0, d * 2, 0, 1, 0)), 1 / (d + 1))
        return out
    if cond == "rain":
        out = img.copy()
        dr = ImageDraw.Draw(out)
        for _ in range(int(w * h / 900)):
            x, y = rng.randrange(w), rng.randrange(h)
            L = rng.randint(8, 18)
            dr.line((x, y, x + L // 4, y + L), fill=(200, 200, 210), width=1)
        return ImageEnhance.Brightness(out.filter(ImageFilter.GaussianBlur(0.6))).enhance(0.85)
    if cond == "glare":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(rng.uniform(1.8, 2.3))).enhance(0.7)
    if cond == "low_res":
        s = rng.uniform(0.08, 0.14)
        return img.resize((max(16, int(w * s)), max(16, int(h * s)))).resize((w, h), Image.NEAREST)
    raise ValueError(cond)


COND_Q = {"fog": "Which photo is foggier or hazier?", "dark": "Which photo was taken in darker conditions?",
          "motion_blur": "Which photo shows motion blur?", "rain": "Which photo shows rain streaks?",
          "glare": "Which photo is overexposed?", "low_res": "Which photo has lower resolution?"}


def build_conditions(root, n, seed=0):
    from PIL import Image

    name, src = "photo_condition", "oi_train_images"
    tmp = os.path.join(root, ".tmp-" + name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    rng = random.Random(seed)
    meta = {}
    for sp in ("train", "val"):
        for l in open(os.path.join(root, "oi_presence", sp + ".jsonl")):
            r = json.loads(l)
            meta[os.path.basename(r["image"])[:-4]] = r.get("attribution")
    ids = sorted(meta)
    rng.shuffle(ids)
    files = {s: open(os.path.join(tmp, s + ".jsonl"), "w") for s in ("train", "val")}
    counts, labels = Counter(), defaultdict(Counter)
    for i, iid in enumerate(ids[:n]):
        split = "val" if int(hashlib.md5(iid.encode()).hexdigest(), 16) % 100 < 5 else "train"
        with Image.open(os.path.join(root, src, "images", iid + ".jpg")) as im:
            img = im.convert("RGB")
        img.thumbnail((640, 640))
        cond = rng.choice(sorted(COND_Q))
        a = "images/%s-clean.jpg" % iid
        b = "images/%s-%s.jpg" % (iid, cond)
        img.save(os.path.join(tmp, a), "JPEG", quality=90)
        _condition(img, cond, rng).save(os.path.join(tmp, b), "JPEG", quality=90)
        gold = rng.randrange(2)
        imgs = [b, a] if gold == 0 else [a, b]
        rec = {"id": "%s-%s-%d" % (name, split, counts[split]), "images": imgs, "state_text": None, "label": gold,
               "question": {"type": "choice", "instructions": COND_Q[cond],
                            "criteria": ["the first photo", "the second photo"]},
               "image_license": "CC BY 2.0", "attribution": dict(meta[iid] or {}, modified="condition applied (VisionJev)")}
        files[split].write(json.dumps(rec) + "\n")
        counts[split] += 1
        labels[split][gold] += 1
    for fh in files.values():
        fh.close()
    _finish(root, name, tmp, counts, labels, "Open Images photos (oi_train_images) with a synthetic condition applied",
            {"validator": {"skip_content_leakage": True}}, license_images="CC BY 2.0 photos, modified")


def _finish(root, name, tmp, counts, labels, source, extra, license_images="generated"):
    m = {"dataset": name, "source": source, "revision": "v1",
         "license": {"annotations": "generated in-house, exact labels", "images": license_images, "commercial": True,
                     "verified": "generated in-house"},
         "counts": dict(counts), "label_counts": {s: {str(k): v for k, v in c.items()} for s, c in labels.items()}, **extra}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    final = os.path.join(root, name)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    print("%s: %s" % (name, dict(counts)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--charts", type=int, default=12000)
    ap.add_argument("--conditions", type=int, default=6000)
    ap.add_argument("--chart-scores", type=int, default=0, help="also build chart_score (score questions only)")
    a = ap.parse_args()
    if a.chart_scores:
        build_charts(a.root, a.chart_scores, seed=1, name="chart_score", score_only=True)
    if a.charts:
        build_charts(a.root, a.charts)
    if a.conditions:
        build_conditions(a.root, a.conditions)
