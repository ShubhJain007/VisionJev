#!/usr/bin/env python3
"""Data validator for prepared decision-model datasets (CONTEXT.md §12–13).

Runs on prepared JSONL *before* training. Cheap, deterministic, no model.
Exits non-zero on any violation and always prints a summary.

Usage:
    python3 data_validator.py data/vqa/aokvqa data/vqa/vqav2_yesno        # dataset dirs: every <split>.jsonl
    python3 data_validator.py data/vqa/aokvqa/train.jsonl eval:held.jsonl  # files, split inferred or split:path

Format: the one the repo loader reads (``laya.vlm_train.jsonl_example``), one question per line:
    {"id", "image" | "images", "state_text", "question": {"type", "instructions", "criteria"},
     "label": int, "target"?, "value"?, "next_target"?, "image_license"?}
``label`` indexes the rendered options (choice: criteria order; score: level; noul: 0 = false, 1 = true).
Image paths are relative to the JSONL's directory.

Checks
  per row   schema (known keys, types), label is a valid index, choice/score criteria unique and >= 2
            (the loader silently DROPS rows that break these), soft target shape/sum/argmax == label,
            value in [0, 1], images open, unique ids.
  per file  noul class balance, choice gold-position histogram ~uniform.
  license   the dataset's manifest.json declares license.commercial = true (else fail, unless
            --allow-noncommercial); for COCO-filtered sets every row's image_license is a commercial one.
  global    image leakage across splits (and across datasets), by exact bytes and by perceptual hash (dHash,
            needs Pillow) so a photo re-encoded by a second source is still caught.

Not covered here (needs a model): the shuffled-image test — see the model evaluator.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

try:  # full decode check when Pillow is available; header sniff otherwise
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

# ---------------------------------------------------------------------------
# Schema (laya.vlm_train.jsonl_example / laya.visionjev_data)
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {"question", "label"}
OPTIONAL_KEYS = {"id", "image", "images", "state_text", "target", "value", "next_target",
                 "episode", "step", "image_license", "dataset", "workflow", "attribution",
                 "rationale", "kind", "provenance"}  # rationale loss text; generator kind; teacher provenance
QUESTION_TYPES = {"choice", "noul", "score"}
QUESTION_KEYS = {"type", "instructions", "criteria"}
SPLIT_NAMES = ("train", "val", "test", "eval", "calib")
MIN_OPTIONS = 2
SOFT_SUM_TOL = 1e-3
#: image licenses a commercial COCO-filtered row may carry (laya.visionjev_data.COCO_LICENSE_NAMES, ids 4-8)
COMMERCIAL_IMAGE_LICENSES = {"CC BY 2.0", "CC BY-SA 2.0", "CC BY-ND 2.0", "No known copyright restrictions",
                             "United States Government Work"}

IMAGE_MAGIC = (
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"GIF87a", b"GIF89a",     # GIF
    b"BM",                    # BMP
    b"II*\x00", b"MM\x00*",   # TIFF
)


@dataclass
class Thresholds:
    noul_min_minority: float = 0.2   # minority class share must be >= this
    noul_min_count: int = 50         # only judge balance with this many labels
    position_tol: float = 0.5        # each position share within (1±tol)/n
    position_min_count: int = 20     # per-position expected count to judge


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
@dataclass
class Report:
    errors: dict = field(default_factory=lambda: defaultdict(list))
    warnings: list = field(default_factory=list)
    overrides: dict = field(default_factory=dict)  # abs jsonl path -> manifest "validator" block

    def error(self, check: str, where: str, msg: str) -> None:
        self.errors[check].append(f"{where}: {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def n_errors(self) -> int:
        return sum(len(v) for v in self.errors.values())


@dataclass
class FileStats:
    path: str
    split: str
    rows: int = 0
    text_only_rows: int = 0
    type_counts: Counter = field(default_factory=Counter)
    noul_by_q: dict = field(default_factory=lambda: defaultdict(Counter))
    noul_overall: Counter = field(default_factory=Counter)
    option_counts: Counter = field(default_factory=Counter)
    positions: dict = field(default_factory=lambda: defaultdict(Counter))
    score_levels: dict = field(default_factory=lambda: defaultdict(Counter))
    image_licenses: Counter = field(default_factory=Counter)
    image_hashes: dict = field(default_factory=dict)  # hash -> first image path
    image_dhashes: dict = field(default_factory=lambda: defaultdict(set))  # perceptual hash -> image files
    ids: set = field(default_factory=set)
    license: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)  # manifest "validator": per-dataset thresholds, e.g. real base rates


# ---------------------------------------------------------------------------
# Row validation
# ---------------------------------------------------------------------------
def _is_str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def options_of(q: dict):
    """The rendered option list (choice/score: criteria keys or items; noul: [false, true]), or None if malformed."""
    if q.get("type") == "noul":
        return ["false", "true"]
    crit = q.get("criteria")
    if isinstance(crit, dict):
        return list(crit.keys())
    if isinstance(crit, list):
        return crit
    return None


def validate_question(q, where: str, rep: Report):
    """Return the option list if the question is well formed, else None (after recording errors)."""
    if not isinstance(q, dict):
        rep.error("schema", where, "'question' is not an object")
        return None
    unknown = set(q) - QUESTION_KEYS
    if unknown:
        rep.error("schema", where, f"question has unknown keys {sorted(unknown)}")
    qtype = q.get("type")
    if qtype not in QUESTION_TYPES:
        rep.error("schema", where, f"invalid question type {qtype!r}")
        return None
    if not _is_str(q.get("instructions")):
        rep.error("schema", where, "question missing non-empty 'instructions'")
        return None
    if qtype == "noul":
        crit = q.get("criteria")
        # allowed: no criteria, or descriptions of the two outcomes ({"false": ..., "true": ...}, rendered by the repo)
        if crit not in (None, [], {}) and not (isinstance(crit, dict) and set(crit) <= {"false", "true"}
                                                 and all(_is_str(v) for v in crit.values())):
            rep.error("schema", where, "noul 'criteria' may only describe the outcomes: {\"false\": str, \"true\": str}")
            return None
        return options_of(q)
    opts = options_of(q)
    if opts is None or not all(_is_str(str(o)) for o in opts):
        rep.error("schema", where, f"{qtype} question needs 'criteria' as a list (or dict) of non-empty options")
        return None
    opts = [str(o) for o in opts]
    if len(opts) < MIN_OPTIONS:
        rep.error("schema", where, f"{qtype} question has {len(opts)} option(s); need >= {MIN_OPTIONS}")
        return None
    if len(set(opts)) != len(opts):
        dups = [c for c, n in Counter(opts).items() if n > 1]
        rep.error("options", where, f"duplicate options {dups} (the loader would silently drop this row)")
        return None
    return opts


def validate_distribution(name: str, dist, k: int, label, where: str, rep: Report) -> None:
    if not isinstance(dist, list) or not all(_is_num(p) for p in dist):
        rep.error("soft", where, f"'{name}' must be a list of numbers")
        return
    if len(dist) != k:
        rep.error("soft", where, f"'{name}' has length {len(dist)}, expected {k}")
        return
    if any(p < 0 for p in dist) or abs(sum(dist) - 1.0) > SOFT_SUM_TOL:
        rep.error("soft", where, f"'{name}' is not a distribution (sum={sum(dist):.4f})")
        return
    # Human gold overrules the teacher: the soft target's argmax must agree with the label.
    if label is not None:
        top = max(range(k), key=lambda i: dist[i])
        if dist[top] > dist[label] + 1e-9:
            rep.error("soft", where, f"'{name}' argmax {top} disagrees with label {label}")


def dhash(im) -> int | None:
    """64-bit difference hash: the same photo re-encoded or resized by another source lands within a few bits. None
    for flat images (no gradient), whose hash says nothing about content."""
    px = im.convert("L").resize((9, 8), Image.BILINEAR).tobytes()
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return None if bits in (0, (1 << 64) - 1) else bits


THUMBS: dict = {}  # image file -> grayscale thumbnail, to confirm dHash candidates (keyed by FILE: text-dense pages
#                    often share a dHash, so a per-hash thumbnail would compare the wrong page)
THUMB_SIDE = 64


def thumbnail(im) -> bytes:
    return im.convert("L").resize((THUMB_SIDE, THUMB_SIDE), Image.BILINEAR).tobytes()


def correlation(a: bytes, b: bytes) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va and vb else 0.0


def check_image(path: str, where: str, rep: Report, cache: dict):
    """Return ``(sha256, dhash)`` of an openable image (dhash None without Pillow), else None (and record an error)."""
    if path in cache:
        return cache[path]
    digest = None
    if not os.path.isfile(path):
        rep.error("image", where, f"image not found: {path}")
    else:
        try:
            with open(path, "rb") as f:
                data = f.read()
            dh = None
            if Image is not None:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:  # verify() leaves the image unusable; reopen to hash the pixels
                    dh = dhash(im)
                    if dh is not None:
                        dh = (dh, thumbnail(im))
            elif not (data.startswith(IMAGE_MAGIC) or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")):
                raise ValueError("unrecognized image header")
            digest = (hashlib.sha256(data).hexdigest(), dh)
        except Exception as e:  # noqa: BLE001 — any failure to open is a violation
            rep.error("image", where, f"cannot open image {path}: {e}")
    cache[path] = digest
    return digest


def validate_row(row, where: str, stats: FileStats, rep: Report, image_root: str, cache: dict,
                 check_images: bool) -> None:
    if not isinstance(row, dict):
        rep.error("schema", where, "row is not a JSON object")
        return
    missing = REQUIRED_KEYS - set(row)
    unknown = set(row) - REQUIRED_KEYS - OPTIONAL_KEYS
    if missing:
        rep.error("schema", where, f"missing keys {sorted(missing)}")
    if unknown:
        rep.error("schema", where, f"unknown keys {sorted(unknown)} (schema drift?)")
    if missing:
        return

    rid = row.get("id")
    if rid is not None:
        if rid in stats.ids:
            rep.error("schema", where, f"duplicate id {rid!r}")
        stats.ids.add(rid)
    if row.get("state_text") is not None and not isinstance(row["state_text"], str):
        rep.error("schema", where, "'state_text' must be a string or null")

    paths = []
    if row.get("image"):
        paths = [row["image"]]
    elif row.get("images"):
        paths = row["images"]
    if "image" in row and row["image"] is not None and not _is_str(row["image"]):
        rep.error("schema", where, "'image' must be a non-empty path string (omit for text-only)")
        paths = []
    if not isinstance(paths, list) or not all(_is_str(p) for p in paths):
        rep.error("schema", where, "'images' must be a list of path strings")
        paths = []
    if not paths:
        stats.text_only_rows += 1
    elif check_images:
        for p in paths:
            digest = check_image(p if os.path.isabs(p) else os.path.join(image_root, p), where, rep, cache)
            if digest:
                stats.image_hashes.setdefault(digest[0], p)
                if digest[1] is not None:
                    full = os.path.normpath(p if os.path.isabs(p) else os.path.join(image_root, p))
                    stats.image_dhashes[digest[1][0]].add(full)
                    THUMBS[full] = digest[1][1]
    if "image_license" in row:
        stats.image_licenses[row["image_license"]] += 1

    opts = validate_question(row["question"], where, rep)
    if opts is None:
        return
    q, label, k = row["question"], row["label"], len(opts)
    if not _is_int(label) or not 0 <= label < k:
        rep.error("label", where, f"label must be an int index in [0, {k - 1}], got {label!r} "
                                  "(the loader would silently drop this row)")
        return
    for name in ("target", "next_target"):
        if row.get(name) is not None:
            validate_distribution(name, row[name], k, label if name == "target" else None, where, rep)
    if row.get("value") is not None and not (_is_num(row["value"]) and 0.0 <= row["value"] <= 1.0):
        rep.error("schema", where, f"'value' must be a number in [0, 1], got {row['value']!r}")

    qtype = q["type"]
    stats.type_counts[qtype] += 1
    if qtype == "noul":
        stats.noul_by_q[q["instructions"]][label] += 1
        stats.noul_overall[label] += 1
    else:
        stats.option_counts[k] += 1
        if qtype == "choice":
            stats.positions[k][label] += 1
        else:
            stats.score_levels[k][label] += 1


# ---------------------------------------------------------------------------
# File / cross-file checks
# ---------------------------------------------------------------------------
def expand_args(args):
    """Dataset dirs -> their <split>.jsonl files; ``split:path`` and plain files pass through."""
    out = []
    for arg in args:
        if os.path.isdir(arg):
            files = sorted(glob.glob(os.path.join(arg, "*.jsonl")))
            if not files:
                out.append(arg)  # reported as "no jsonl files"
            out += files
        else:
            out.append(arg)
    return out


def infer_split(arg: str):
    if ":" in arg and not os.path.exists(arg):
        split, path = arg.split(":", 1)
        return split, path
    stem = os.path.basename(arg).lower()
    for name in SPLIT_NAMES:
        if name in stem:
            return name, arg
    return "train", arg


def load_license(path: str, rep: Report) -> dict:
    """The ``license`` block of the dataset's manifest.json (next to the JSONL), or {} with an error."""
    manifest = os.path.join(os.path.dirname(os.path.abspath(path)), "manifest.json")
    if not os.path.isfile(manifest):
        rep.error("license", path, "no manifest.json with a license block next to this file")
        return {}
    try:
        with open(manifest, encoding="utf-8") as f:
            lic = json.load(f).get("license")
    except (OSError, json.JSONDecodeError) as e:
        rep.error("license", manifest, f"unreadable manifest: {e}")
        return {}
    try:
        with open(manifest, encoding="utf-8") as f:
            ov = json.load(f).get("validator") or {}
        rep.overrides[os.path.abspath(path)] = ov if isinstance(ov, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    if not isinstance(lic, dict) or "commercial" not in lic:
        rep.error("license", manifest, "manifest has no license block with a 'commercial' flag")
        return {}
    return lic


def check_license(stats: FileStats, allow_noncommercial: bool, rep: Report) -> None:
    lic = stats.license
    if not lic:
        return
    if not lic.get("commercial") and not allow_noncommercial:
        rep.error("license", stats.path, f"dataset is not cleared for commercial use ({lic.get('annotations')})")
    if lic.get("image_filter") == "coco":
        missing = stats.rows - sum(stats.image_licenses.values())
        if missing:
            rep.error("license", stats.path, f"{missing} COCO-filtered row(s) carry no image_license")
        bad = {k: v for k, v in stats.image_licenses.items() if k not in COMMERCIAL_IMAGE_LICENSES}
        if bad and not allow_noncommercial:
            rep.error("license", stats.path, f"non-commercial image licenses present: {bad}")


def validate_file(path: str, split: str, rep: Report, check_images: bool, cache: dict,
                  check_licenses: bool) -> FileStats:
    stats = FileStats(path=path, split=split)
    if not os.path.isfile(path):
        rep.error("file", path, "file not found" if not os.path.isdir(path) else "no .jsonl files in directory")
        return stats
    if check_licenses:
        stats.license = load_license(path, rep)
    root = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            stats.rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                rep.error("schema", where, f"invalid JSON: {e}")
                continue
            validate_row(row, where, stats, rep, root, cache, check_images)
    if stats.rows == 0:
        rep.error("file", path, "file has no rows")
    return stats


def check_noul_balance(stats: FileStats, th: Thresholds, rep: Report) -> None:
    min_minority = rep.overrides.get(os.path.abspath(stats.path), {}).get("noul_min_minority", th.noul_min_minority)

    def judge(name: str, c: Counter) -> None:
        total = c[0] + c[1]
        if total < th.noul_min_count:
            return
        minority = min(c[0], c[1]) / total
        if minority < min_minority:
            rep.error("noul_balance", stats.path,
                      f"{name}: {c[1]} true / {c[0]} false (minority {minority:.1%} < "
                      f"{min_minority:.0%}) — mix in negatives")

    for q, c in stats.noul_by_q.items():
        judge(f"noul question {q!r}", c)
    judge("noul overall", stats.noul_overall)


def check_positions(stats: FileStats, th: Thresholds, rep: Report) -> None:
    for n, hist in stats.positions.items():
        total = sum(hist.values())
        if total / n < th.position_min_count:
            rep.warn(f"{stats.path}: {n}-option choice has only {total} rows; position histogram not judged")
            continue
        lo, hi = (1 - th.position_tol) / n, (1 + th.position_tol) / n
        sd = math.sqrt(total * (1 / n) * (1 - 1 / n))  # binomial sd of one position's count under no bias
        # outside the band AND |z| > 3.5: with ~30 position bins per validation run, a lone 3-sigma bin is noise
        bad = {i: hist[i] / total for i in range(n)
               if not lo <= hist[i] / total <= hi and abs(hist[i] - total / n) > 3.5 * sd}
        if bad:
            shares = ", ".join(f"pos{i}={s:.1%}" for i, s in sorted(bad.items()))
            rep.error("position_bias", stats.path,
                      f"{n}-option gold positions not ~uniform (expect {1 / n:.1%} each, "
                      f"allowed [{lo:.1%}, {hi:.1%}]): {shares}")


DHASH_MAX_DISTANCE = 8  # bits of 64: re-encodes/resizes of one photo land within ~0-4, unrelated photos near 32 but
#                         with ~1e8 pairs the tail reaches 8, so a candidate must also pass the thumbnail check
MIN_THUMB_CORRELATION = 0.98  # 64x64 grayscale correlation. Measured: the same COCO photo re-encoded by two sources
#                               >= 0.998; different document pages from one template (arXiv "figure available" stubs)
#                               0.92-0.96, dense text pages that merely share a dHash 0.2-0.7
DHASH_BANDS = 9         # 64 bits in 9 bands of <= 8 bits: two hashes within 8 bits share at least one band exactly


def _near_pairs(a: dict, b: dict, max_dist: int):
    """Pairs (ha, hb) with Hamming distance <= max_dist, via exact matches on bit bands (pigeonhole)."""
    width = -(-64 // DHASH_BANDS)
    def bands(h):
        return [(i, (h >> (i * width)) & ((1 << width) - 1)) for i in range(DHASH_BANDS)]
    index = defaultdict(list)
    for hb in b:
        for key in bands(hb):
            index[key].append(hb)
    for ha in a:
        seen = set()
        for key in bands(ha):
            for hb in index.get(key, ()):
                if hb not in seen:
                    seen.add(hb)
                    if bin(ha ^ hb).count("1") <= max_dist:
                        yield ha, hb


def check_leakage(all_stats: list, rep: Report, max_dist: int = DHASH_MAX_DISTANCE) -> None:
    """Same image in two splits, by exact bytes (sha256) or by content: dHash within ``max_dist`` bits finds candidates
    (the same photo encoded or resized differently by two sources, e.g. COCO via A-OKVQA and via VQAv2), a thumbnail
    correlation >= ``MIN_THUMB_CORRELATION`` between the two files confirms them. One violation per pair of files."""
    reported = set()

    def report(a, b, kind, fa, fb):
        pair = (fa, fb)
        if pair not in reported:
            reported.add(pair)
            rep.error("leakage", f"{a}/{b}", f"same image ({kind}) in both splits: {fa} == {fb}")

    by_split = defaultdict(dict)  # split -> sha -> "jsonl (image)"
    for s in all_stats:
        for h, img in s.image_hashes.items():
            by_split[s.split].setdefault(h, f"{s.path} ({img})")
    splits = sorted(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            for h in sorted(by_split[a].keys() & by_split[b].keys()):
                report(a, b, "bytes", by_split[a][h], by_split[b][h])

    dsplit = defaultdict(lambda: defaultdict(set))  # split -> dhash -> image files
    for s in all_stats:
        # generated sets (templated renders) opt out via their manifest: each split is generated independently, so
        # leakage is impossible by construction, while two renders of one template look alike at 64 px whatever the
        # text says (false positives, and thousands of shared hashes that made this check quadratic)
        if rep.overrides.get(os.path.abspath(s.path), {}).get("skip_content_leakage"):
            continue
        for h, files in s.image_dhashes.items():
            dsplit[s.split][h] |= files
    splits = sorted(dsplit)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            for ha, hb in _near_pairs(dsplit[a], dsplit[b], max_dist):
                fas, fbs = sorted(dsplit[a][ha]), sorted(dsplit[b][hb])
                for fa, fb in _confirmed(fas, fbs):
                    if fa != fb:
                        report(a, b, "content", fa, fb)


_NORM: dict = {}


def _confirmed(fas, fbs):
    """Pairs (fa, fb) whose thumbnail correlation >= MIN_THUMB_CORRELATION: one matrix product per group with numpy
    (zero-mean unit-norm thumbnails, so a dot product is the correlation), the pure-Python loop without it."""
    try:
        import numpy as np
    except ImportError:
        return [(fa, fb) for fa in fas for fb in fbs if correlation(THUMBS[fa], THUMBS[fb]) >= MIN_THUMB_CORRELATION]

    def norm(f):
        if f not in _NORM:
            v = np.frombuffer(THUMBS[f], dtype=np.uint8).astype(np.float32)
            v -= v.mean()
            n = float(np.linalg.norm(v))
            _NORM[f] = v / n if n else v
        return _NORM[f]

    A = np.stack([norm(f) for f in fas])
    B = np.stack([norm(f) for f in fbs])
    ii, jj = np.nonzero(A @ B.T >= MIN_THUMB_CORRELATION)
    return [(fas[i], fbs[j]) for i, j in zip(ii, jj)]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _hist(c: Counter, keys=None) -> str:
    keys = sorted(c) if keys is None else keys
    return " ".join(f"{k}:{c[k]}" for k in keys)


def print_summary(all_stats: list, rep: Report, max_per_check: int, top_k: int = 5) -> None:
    out = print
    out("=" * 72)
    out("DATA VALIDATOR SUMMARY")
    out("=" * 72)
    for s in all_stats:
        out(f"\n{s.path}  [split={s.split}]")
        if s.license:
            out(f"  license: annotations={s.license.get('annotations')}  images={s.license.get('images')}  "
                f"commercial={s.license.get('commercial')}")
        out(f"  rows: {s.rows}  (text-only: {s.text_only_rows}, unique images: {len(s.image_hashes)})")
        out(f"  questions by type: {_hist(s.type_counts) or '-'}")
        if s.image_licenses:
            out(f"  image licenses: {dict(s.image_licenses.most_common())}")
        if s.option_counts:
            out(f"  option-count distribution: {_hist(s.option_counts)}")
        for n in sorted(s.positions):
            out(f"  choice gold position ({n} opts): {_hist(s.positions[n], range(n))}")
        for n in sorted(s.score_levels):
            out(f"  score gold level ({n} levels): {_hist(s.score_levels[n], range(n))}")
        if s.noul_overall:
            c = s.noul_overall
            out(f"  noul overall: true:{c[1]} false:{c[0]}  ({len(s.noul_by_q)} distinct instructions)")
            for q, c in sorted(s.noul_by_q.items(), key=lambda kv: -sum(kv[1].values()))[:top_k]:
                out(f"    {q[:60]!r}: true:{c[1]} false:{c[0]}")

    if rep.warnings:
        out(f"\nWARNINGS ({len(rep.warnings)})")
        for w in rep.warnings:
            out(f"  - {w}")

    out("")
    if not rep.n_errors:
        out("RESULT: PASS — no violations")
        return
    out(f"RESULT: FAIL — {rep.n_errors} violation(s)")
    for check in sorted(rep.errors):
        errs = rep.errors[check]
        out(f"\n[{check}] {len(errs)}")
        for e in errs[:max_per_check]:
            out(f"  - {e}")
        if len(errs) > max_per_check:
            out(f"  ... {len(errs) - max_per_check} more")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(files, check_images=True, th: Thresholds | None = None, max_per_check=20, quiet=False,
        check_licenses=True, allow_noncommercial=False):
    th = th or Thresholds()
    THUMBS.clear()
    _NORM.clear()
    rep = Report()
    cache: dict = {}
    all_stats = []
    for arg in expand_args(files):
        split, path = infer_split(arg)
        stats = validate_file(path, split, rep, check_images, cache, check_licenses)
        if check_licenses:
            check_license(stats, allow_noncommercial, rep)
        check_noul_balance(stats, th, rep)
        check_positions(stats, th, rep)
        all_stats.append(stats)
    if check_images:
        check_leakage(all_stats, rep)
    if not quiet:
        print_summary(all_stats, rep, max_per_check)
    return rep, all_stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="dataset dirs or JSONL files (optionally split:path)")
    p.add_argument("--no-image-check", action="store_true", help="skip opening/hashing images (disables leakage check)")
    p.add_argument("--no-license-check", action="store_true", help="skip the manifest license gate")
    p.add_argument("--allow-noncommercial", action="store_true", help="research runs: don't fail on NC licenses")
    p.add_argument("--noul-min-minority", type=float, default=Thresholds.noul_min_minority)
    p.add_argument("--noul-min-count", type=int, default=Thresholds.noul_min_count)
    p.add_argument("--position-tol", type=float, default=Thresholds.position_tol)
    p.add_argument("--position-min-count", type=int, default=Thresholds.position_min_count)
    p.add_argument("--max-per-check", type=int, default=20, help="violations printed per check")
    a = p.parse_args(argv)
    th = Thresholds(a.noul_min_minority, a.noul_min_count, a.position_tol, a.position_min_count)
    rep, _ = run(a.paths, not a.no_image_check, th, a.max_per_check, check_licenses=not a.no_license_check,
                 allow_noncommercial=a.allow_noncommercial)
    return 1 if rep.n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
