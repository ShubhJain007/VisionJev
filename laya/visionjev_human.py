"""Human-labelled public sets as typed decisions (Hypothesis/V6.3.md section 3.4; licences re-read on the primary
source 2026-09-30, see ``SOURCES[...]["sources"]``). Nothing here is generated or labelled by a language model: labels
are expert programs, human annotations or human ratings, and every judged response comes from a person or an open
research system (no closed-model output).

A. Arithmetic over tables / finance / dates (exact labels)
* ``finqa_choice``     (choice) FinQA: report text + table + question -> the number. The expert program is executed
                       here (the result must equal the released ``exe_ans``); options are rank-uniform; distractors are
                       single-mistake results first (operands swapped, stopped one step early, another number of the
                       program, the neighbouring table cell = wrong row / year, percent vs ratio), then a geometric
                       ladder of near values. ``rationale`` = the program as worked steps with intermediate values.
* ``convfinqa_choice`` (choice) ConvFinQA: the same, for a turn of a conversation with the earlier turns (questions and
                       answers) in the state. Every final turn, plus earlier turns that compute something.
* ``tatqa_choice``     (choice) TAT-QA: arithmetic (derivation parsed, executed and checked against the answer; same
                       mistake distractors), count, and span answers that are table cells (distractors: cells of the
                       same row / column). The derivation is the rationale.
* ``tabfact_noul``     (noul) TabFact: table + statement -> supported? One entailed and one refuted statement per table.
* ``timeqa_choice``    (choice) TimeQA (hard): article + time-scoped question; distractors are the same entity's answers
                       for OTHER time scopes in the same article; "no answer for that time" is gold on the unanswerable
                       questions and a distractor on a share of the others.

B. Judging with human labels
* ``summeval_pair``    (choice, 2) two system summaries of one article: which is more consistent / relevant (expert mean
                       differs by >= 1 and no expert prefers the other one). Position- and length-balanced.
* ``summeval_score``   (score, 5) coherence / consistency / fluency / relevance with the three experts' vote shares as
                       the soft target.
* ``frank_factual``    (noul + choice) FRANK, CNN/DM part: is this summary sentence factually consistent (soft target =
                       annotator share); for sentences with an agreed error, which type of error.
* ``mqm_pair``         (choice, 2) WMT MQM: source + two translations of one segment: which has the lower MQM penalty
                       (>= 5 points = one major error, no rater reversed). Position- and length-balanced.
* ``mqm_major_error``  (noul) does the translation contain a major error (soft target = rater share where 3 raters).
* ``faithdial_noul``   (noul) FaithDial: knowledge + dialogue + response -> faithful to the knowledge?
* ``begin_attribution``(choice, 3) BEGIN (CMU-DoG and Wizard-of-Wikipedia parts): attribution class of a response.
* ``feedbackqa_score`` (score, 4) FeedbackQA: question + answer passage -> human rating, rater shares as soft target.

Not built: ASAP 2.0 (its PERSUADE essays cannot be told apart in the release; see ``SKIPPED``).

Rows dropped by a filter are counted in ``DROPS`` and written to the manifest (``dropped``); raw GitHub files are
cached under ``<data>/raw/human`` at pinned commits, Hub files in the normal Hub cache at pinned revisions.

    python -m laya.visionjev_human all
"""
import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT, RAW_ROOT

RAW = os.path.join(RAW_ROOT, "human")
VERIFIED = "2026-09-30 primary source (GitHub LICENSE / README, Hub card); Hypothesis/V6.3.md section 3.4"
MAX_TOKENS = 3500
TOKENIZER = "Qwen/Qwen3-VL-4B-Instruct"
TRAIN_CAP, VAL_CAP = 10000, 1000

GITHUB = {  # repository -> pinned commit (read 2026-09-30)
    "czyssrs/FinQA": "0f16e2867befa6840783e58be38c9efb9229d742",
    "czyssrs/ConvFinQA": "cf3eed2d5984960bf06bb8145bcea5e80b0222a6",
    "wenhuchen/Table-Fact-Checking": "2ab782ba42b5808076ac91fec846473aa5315a79",
    "wenhuchen/Time-Sensitive-QA": "38b05989070c1168b2bef3d5a2656afeeba763dc",
    "Yale-LILY/SummEval": "81b59ad53d63cb6009764240853c91235a44e238",
    "artidoro/frank": "80a88fb12cc0bfc17ff6ee2c5ebb0c4f4dd8a5f4",
    "google/wmt-mqm-human-evaluation": "29acd6999aaea3586378c9dcb58d8cc4820cede3",
    "google/BEGIN-dataset": "6e65f0d5a1ed1b9574643a7640ff1678d4e979b5",
}
HUB = {  # Hub dataset -> pinned revision
    "next-tat/TAT-QA": "c96247f5077eac447f63527fd3dcfdc58bb56d6a",
    "McGill-NLP/FaithDial": "7a414e80725eac766f2602676dc8b39f80b061e4",
    "McGill-NLP/feedbackQA": "41ea0e39f062f9ca791fd5ec95c364a22150b56e",
    "abisee/cnn_dailymail": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
}
SUMMEVAL_URL = "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl"
SKIPPED = {
    "asap2_score": "ASAP 2.0 (github scrosseye/ASAP_2.0 @060ab22, CC BY 4.0 per README): 12,871 of its essays come from "
                   "PERSUADE 2.0 (non-commercial) and no column marks them (same id pattern and prompts as the new "
                   "essays; only the 4,005 'Car-free cities' / 'Electoral college' essays are identifiable), so the "
                   "PERSUADE essays cannot be excluded from the release alone.",
}

DROPS: Dict[str, Counter] = defaultdict(Counter)  # set name -> {"<split>: <reason>": rows}
NOTES: Dict[str, dict] = defaultdict(dict)  # set name -> facts measured while building (written to the manifest)


def _drop(name: str, split: str, why: str, n: int = 1) -> None:
    DROPS[name]["%s: %s" % (split, why)] += n


# ---- files ------------------------------------------------------------------------------------------------------------


def _fetch(url: str, rel: str) -> str:
    path = os.path.join(RAW, rel)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(url, timeout=900) as r, open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(path + ".part", path)
    return path


def _gh(repo: str, path: str, folder: str) -> str:
    url = "https://raw.githubusercontent.com/%s/%s/%s" % (repo, GITHUB[repo], path)
    return _fetch(url, os.path.join(folder, os.path.basename(path)))


def _hub(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=HUB[repo])


_TOK = None


def _ntok(text: str) -> int:
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(TOKENIZER)
    return len(_TOK(text, add_special_tokens=False)["input_ids"])


def _fits(text: str, ntok: Optional[Callable[[str], int]] = None) -> bool:
    """State within the budget. A token covers at least one byte, so short texts skip the tokenizer."""
    return len(text.encode("utf-8")) <= MAX_TOKENS or (ntok or _ntok)(text) <= MAX_TOKENS


def _split(key: str, pct: int) -> str:
    return "val" if int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100 < pct else "train"


def _state(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _table_text(rows: List[List[str]]) -> str:
    return "\n".join(" | ".join(str(c).strip() for c in row) for row in rows)


# ---- programs over numbers (FinQA DSL; TAT-QA derivations are converted to it) -------------------------------------------

OPS2 = ("add", "subtract", "multiply", "divide", "exp", "greater")
TOPS = ("table_sum", "table_average", "table_max", "table_min")
SYM = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "exp": "^"}
TOP_WORD = {"table_sum": "sum", "table_average": "average", "table_max": "largest value", "table_min": "smallest value"}
_STEP = re.compile(r"([a-z_]+)\(([^()]*)\)")
Step = Tuple[str, str, str]


def _g(v) -> str:
    """A value as written in a rationale: integers bare, else up to 5 decimals (7 below 1, so ratios keep digits)."""
    if isinstance(v, str):
        return v
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return str(int(round(v)))
    return ("%.*f" % (5 if abs(v) >= 1 else 7, v)).rstrip("0").rstrip(".")


def lit_value(text: str) -> Optional[float]:
    """A literal of a FinQA program (the official ``str_to_num``): ``5,829`` ``23.6%`` (= 0.236) ``const_100``
    ``const_m1``. None for a step reference or anything else."""
    t = text.strip().replace(",", "")
    try:
        return float(t)
    except ValueError:
        pass
    if t.endswith("%"):
        try:
            return float(t[:-1]) / 100.0
        except ValueError:
            return None
    if t.startswith("const_"):
        c = t[6:]
        try:
            return -1.0 if c == "m1" else float(c)
        except ValueError:
            return None
    return None


def cell_num(cell) -> Optional[float]:
    """The number a table cell shows: ``$ -23158 ( 23158 )`` -> -23158, ``(3.1)`` -> -3.1, ``9.4%`` -> 9.4."""
    s = re.sub(r"[$€£]", "", str(cell)).strip()
    m = re.fullmatch(r"\(\s*([\d,]*\.?\d+)\s*\)\s*%?", s)
    if m:
        return -float(m.group(1).replace(",", ""))
    m = re.match(r"(-?)\s*(\d[\d,]*(?:\.\d+)?|\.\d+)", s)
    if not m:
        return None
    try:
        return float(m.group(1) + m.group(2).replace(",", ""))
    except ValueError:
        return None


def parse_program(program: str) -> Optional[List[Step]]:
    """``subtract(5829, 5735), divide(#0, 5735)`` -> [(op, arg1, arg2), ...]; None if it is not such a program."""
    prog, steps, pos = program.strip(), [], 0
    for m in _STEP.finditer(prog):
        if prog[pos:m.start()].strip(" ,"):
            return None
        parts = m.group(2).rsplit(",", 1)
        if m.group(1) not in OPS2 + TOPS or len(parts) != 2:
            return None
        steps.append((m.group(1), parts[0].strip(), parts[1].strip()))
        pos = m.end()
    if prog[pos:].strip(" ,") or not steps:
        return None
    return steps


def _row_nums(cells) -> Optional[List[float]]:
    """A table row as numbers, as the official evaluator reads it (``$`` dropped, the text before ``(`` kept)."""
    if not cells:
        return None
    out = []
    for c in cells:
        v = lit_value(str(c).replace("$", "").strip().split("(")[0].strip())
        if v is None:
            return None
        out.append(v)
    return out


def run_program(steps: List[Step], table=None) -> Optional[list]:
    """The value of every step (floats; ``greater`` gives "yes" / "no"), or None if the program cannot be executed."""
    rows = {str(r[0]): r[1:] for r in (table or []) if r}
    vals: list = []

    def arg(t):
        if t.startswith("#"):
            k = int(t[1:]) if t[1:].isdigit() else -1
            return vals[k] if 0 <= k < len(vals) else None
        return lit_value(t)

    for op, a, b in steps:
        try:
            if op in TOPS:
                nums = _row_nums(rows.get(a))
                if not nums:
                    return None
                v = {"table_sum": sum(nums), "table_average": sum(nums) / len(nums), "table_max": max(nums),
                     "table_min": min(nums)}[op]
            else:
                x, y = arg(a), arg(b)
                if x is None or y is None or isinstance(x, str) or isinstance(y, str):
                    return None
                if op == "greater":
                    v = "yes" if x > y else "no"
                else:
                    v = {"add": lambda: x + y, "subtract": lambda: x - y, "multiply": lambda: x * y,
                         "divide": lambda: x / y, "exp": lambda: x ** y}[op]()
                    if isinstance(v, complex) or not math.isfinite(v):
                        return None
        except (ZeroDivisionError, OverflowError, ValueError):
            return None
        vals.append(v)
    return vals


def prune_steps(steps: List[Step]) -> List[Step]:
    """Only the steps the final result depends on, renumbered (a conversation turn's program carries the steps of
    earlier turns even when it does not use them)."""
    ref = lambda t: int(t[1:]) if t.startswith("#") and t[1:].isdigit() else None  # noqa: E731
    need, stack = set(), [len(steps) - 1]
    while stack:
        i = stack.pop()
        if i not in need:
            need.add(i)
            stack += [ref(t) for t in steps[i][1:] if ref(t) is not None and ref(t) < i]
    idx = {old: new for new, old in enumerate(sorted(need))}
    fix = lambda t: "#%d" % idx[ref(t)] if ref(t) in idx else t  # noqa: E731
    return [(steps[i][0], fix(steps[i][1]), fix(steps[i][2])) for i in sorted(need)]


def worked_steps(steps: List[Step], vals: list, table=None) -> str:
    """The program as worked steps with every intermediate value."""
    rows = {str(r[0]): r[1:] for r in (table or []) if r}

    def show(t):
        if t.startswith("#"):
            return _g(vals[int(t[1:])])
        if t.startswith("const_"):
            return _g(lit_value(t))
        return t

    parts = []
    for i, ((op, a, b), v) in enumerate(zip(steps, vals)):
        if op in TOPS:
            parts.append("Step %d: %s of the row '%s' (%s) = %s" % (i + 1, TOP_WORD[op], a,
                                                                 ", ".join(_g(x) for x in _row_nums(rows[a])), _g(v)))
        elif op == "greater":
            parts.append("Step %d: is %s greater than %s? %s" % (i + 1, show(a), show(b), v))
        else:
            parts.append("Step %d: %s %s %s = %s" % (i + 1, show(a), SYM[op], show(b), _g(v)))
    return ". ".join(parts) + "." if parts else ""


def _plain_lit(t: str) -> bool:
    return not t.startswith("#") and not t.startswith("const_") and lit_value(t) is not None


def _neighbours(grid: List[List[Optional[float]]], text: str, limit: int = 6) -> List[str]:
    """Literals a careless reader would take instead of ``text``: the other numbers of its table row (the wrong year /
    column) and the cells just above and below (the wrong row)."""
    pct = text.strip().endswith("%")
    try:
        shown = float(text.strip().rstrip("%").replace(",", ""))
    except ValueError:
        return []
    out = []
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v is None or abs(abs(v) - abs(shown)) > 1e-9 or (v == 0) != (shown == 0):
                continue
            sign = 1.0 if v == shown else -1.0
            cand = [row[c2] for c2 in range(len(row)) if c2 != c]
            cand += [grid[r2][c] for r2 in (r - 1, r + 1) if 0 <= r2 < len(grid) and c < len(grid[r2])]
            for x in cand:
                if x is not None and abs(x - v) > 1e-12:
                    out.append(_g(sign * x) + ("%" if pct else ""))
    return list(dict.fromkeys(out))[:limit]


def program_mistakes(steps: List[Step], vals: list, table=None) -> List[float]:
    """Results of the program with ONE mistake: operands of a subtraction / division swapped (sign, wrong base),
    stopped a step early (forgot to divide / to convert), one number replaced by another number of the program, or by
    the neighbouring table cell (wrong row / year), a table aggregate over the neighbouring row or with the wrong
    aggregate."""
    out: List[float] = []

    def add(new):
        v = run_program(new, table) if new else None
        if v and isinstance(v[-1], float):
            out.append(v[-1])

    def put(i, p, text):
        s = list(steps[i])
        s[1 + p] = text
        if s[1] == s[2]:  # x - x, x / x: a degenerate 0 or 1 nobody would offer
            return []
        return steps[:i] + [tuple(s)] + steps[i + 1:]

    for i, (op, a, b) in enumerate(steps):
        if op in ("subtract", "divide"):
            add(steps[:i] + [(op, b, a)] + steps[i + 1:])
    out += [v for v in vals[:-1] if isinstance(v, float)]
    lits = [(i, p, steps[i][1 + p]) for i in range(len(steps)) for p in (0, 1)
            if steps[i][0] in OPS2 and _plain_lit(steps[i][1 + p])]
    distinct = list(dict.fromkeys(t for _, _, t in lits))
    grid = [[cell_num(c) for c in row] for row in (table or [])]
    for i, p, t in lits:
        for u in distinct:
            if u != t:
                add(put(i, p, u))
        for u in _neighbours(grid, t):
            add(put(i, p, u))
    names = [str(r[0]) for r in (table or []) if r]
    for i, (op, a, b) in enumerate(steps):
        if op in TOPS:
            for op2 in TOPS:
                if op2 != op:
                    add(steps[:i] + [(op2, a, b)] + steps[i + 1:])
            if a in names:
                k = names.index(a)
                for j in (k - 1, k + 1):
                    if 0 <= j < len(names):
                        add(steps[:i] + [(op, names[j], b)] + steps[i + 1:])
    return out


SIGN_PLAN = (0.20, 0.04, 0.30)  # positive gold: P(one opposite-sign distractor), P(two); negative gold: P(one)


def sign_plan(negative: bool, share: float, rng) -> int:
    """How many of the three distractors get the sign opposite to the gold. ``share`` is the share of negative golds
    (among questions where a sign mistake can be derived). The probabilities are set so that in every mixed-sign
    option set the gold is negative with probability (negative options) / (options): most answers are positive, and
    with sign-flip distractors only on those, "the lone negative option is never the answer" would be a shortcut."""
    if not 0 < share < 1:
        return 0
    a, e, c = SIGN_PLAN
    t = min(1.0, (1 - c) * share / ((1 - share) * (a / 3 + e)))
    a, e = a * t, e * t
    probs = ({3: (1 - share) * a / (3 * share), 2: (1 - share) * e / share, 1: c} if negative else
             {1: a, 2: e, 3: share * c / (3 * (1 - share))})
    x = rng.random()
    for k, pr in probs.items():
        if x < pr:
            return k
        x -= pr
    return 0


BAND, SPAN = 8.0, 40.0  # a mistake is used if within BAND x of the gold; no option further than SPAN x from it
NEAR_STEPS = (0.04, 0.06, 0.08, 0.1, 0.15)


def _nice(x: float) -> float:
    """``x`` rounded to 1, 2 or 5 times a power of ten."""
    if x <= 0:
        return 0.0
    base = 10.0 ** math.floor(math.log10(x))
    return min((1, 2, 5, 10), key=lambda m: abs(math.log(m * base / x))) * base


def numeric_choice(ins: str, gold: float, mistakes: List[float], fmt: Callable[[float], str], rng, integer: bool = False,
                   lo=None, unit: float = 1.0, keep: Optional[List[float]] = None, neg_share: float = 0.0,
                   stats: Optional[Counter] = None, scale: float = 1.0):
    """Four numeric options in which nothing but the computation points at the gold.

    The options form a SHAPE built from the offsets of real single-mistake values. With two mistakes at offsets d1, d2
    from the gold it is the parallelogram ``x + {0, d1, d2, d1 + d2}``; with one, a ladder ``x + {0, d, 2d, 3d}`` or two
    pairs ``{A, A + e, A + d, A + d + e}`` (``e`` a near step); with none, a near ladder ``x + {0, e, 2e, 3e}``. Sums
    are tried first (they keep the digits pattern of the real numbers: a whole number stays whole), the same shapes
    with ratios instead of offsets second. The gold is put on a corner drawn uniformly and a shape is used only if all
    four placements are valid. So the gold's rank is uniform, and neither "the option the others cluster around", "the
    two close options" nor "the outlier" identifies it: a gold with its real mistakes and a mistake with its mirrored
    values look the same. A real mistake value is among the options on about 3 corners of 4.

    Mistakes further than ``BAND`` x from the gold are not used (``keep`` exempts the percent-vs-ratio mistake, 100x by
    nature). Near steps are about 4-15% of the value, rounded to 1 / 2 / 5 display units (``unit`` is one unit of the
    last shown digit, ``scale`` the display factor). Sign: where a sign mistake is derivable, ``sign_plan`` draws how
    many distractors are negated.
    """
    g = fmt(gold)
    neg = lambda v: fmt(v).startswith("-")  # noqa: E731
    flip_ok = gold != 0 and lo is None and any(math.isfinite(m) and fmt(m) != fmt(0.0) and neg(m) != neg(gold) for m in mistakes)
    if stats is not None and gold != 0:
        stats["sign mistake derivable"] += flip_ok
        stats["sign mistake derivable, gold negative"] += flip_ok and neg(gold)
    s = rng.choice(NEAR_STEPS)

    def step(ref: float, times: float = 1.0) -> float:
        return max(unit, times * _nice(s * abs(ref) * scale) / scale)

    def usable(vals: List[float], free: bool) -> bool:
        if not all(math.isfinite(v) for v in vals) or len({fmt(v) for v in vals}) < 4:
            return False
        for v in vals[1:]:
            if (lo is not None and v < lo) or (gold != 0 and neg(v) != neg(gold)):
                return False
            if not free and gold != 0 and not abs(gold) / SPAN <= abs(v) <= abs(gold) * SPAN:
                return False
        return True

    # a shape is a function: corner (0-3) -> [gold, the other three values]
    def add_shape(off):
        return lambda c: [gold] + [gold + o - off[c] for k, o in enumerate(off) if k != c]

    def mul_shape(fac):
        return lambda c: [gold] + [gold * f / fac[c] for k, f in enumerate(fac) if k != c]

    def pairs_shape(d):
        """Two roots ``d`` apart, each with a partner one near step above; the step is set by the smaller root, so it
        says nothing about which root is the gold's."""
        def at(c):
            partner, way = c in (1, 3), (d if c < 2 else -d)
            e = step(min(abs(gold), abs(gold + way)))
            root = gold - e if partner else gold
            other = root + way
            return [gold, root if partner else root + e, other, other + e]
        return at

    ratios, offsets, kept = [], [], set()
    for m in dict.fromkeys(list(mistakes) + list(keep or [])):
        if not math.isfinite(m) or m == 0 or gold == 0:
            continue
        r = abs(m) / abs(gold)
        free = m in (keep or [])
        if fmt(gold * r) != g and (free or 1 / BAND <= r <= BAND) and fmt(gold * r) not in kept:
            kept.add(fmt(gold * r))
            ratios.append((r, free))
            offsets.append(gold * r - gold)
    order = list(range(len(ratios)))
    rng.shuffle(order)
    shapes = []  # (shape, exempt from SPAN), tried in this order
    pairs = [(a, b) for a in order for b in order if a < b]
    rng.shuffle(pairs)
    for a, b in pairs[:8]:
        free = ratios[a][1] or ratios[b][1]
        if not free:
            shapes.append((add_shape([0.0, offsets[a], offsets[b], offsets[a] + offsets[b]]), False))
        shapes.append((mul_shape([1.0, ratios[a][0], ratios[b][0], ratios[a][0] * ratios[b][0]]), free))
    for a in order:
        (r, free), d = ratios[a], offsets[a]
        ladder = [(add_shape([0.0, d, 2 * d, 3 * d]), False), (mul_shape([1.0, r, r * r, r ** 3]), free)]
        two = [(pairs_shape(d), False), (mul_shape([1.0, r, 1 + s, r * (1 + s)]), free)]
        if free:
            ladder, two = ladder[1:], two[1:]
        shapes += (ladder + two) if rng.random() < 0.5 else (two + ladder)
    e = step(gold)
    shapes += [(add_shape([k * e * t for k in range(4)]), False) for t in (1, 2, 5, 10)]
    shapes += [(mul_shape([(1 + t) ** k for k in range(4)]), False) for t in (s, 0.15, 0.3, 0.6, 1.0)]
    vals = None
    for shape, free in shapes:
        placed = [shape(c) for c in range(4)]
        if all(usable(v, free) for v in placed):
            vals = placed[rng.randrange(4)]
            break
    if vals is None:  # last resort (e.g. a count near its lower bound): a unit ladder on the corners that are valid
        placed = [v for v in (add_shape([k * unit for k in range(4)])(c) for c in range(4)) if usable(v, False)]
        if not placed:
            return None
        vals = rng.choice(placed)
    if flip_ok:
        for k in rng.sample([1, 2, 3], sign_plan(neg(gold), neg_share, rng)):
            vals[k] = -vals[k]
    opts = [g] + [fmt(v) for v in vals[1:]]
    if len(set(opts)) < 4:
        return None
    return ts._fixed_choice(ins, opts, 0, rng)


def percent_ratio_mistake(steps: List[Step], vals: list, scale: float, rng, p: float = 0.2) -> List[float]:
    """The percent-vs-ratio mistake, offered on a share ``p`` of the questions it applies to: a ratio shown as a
    percentage (``scale`` 100) read without the conversion, or the value before a final ``* 100`` step."""
    if rng.random() >= p or not vals or isinstance(vals[-1], str):
        return []
    if scale == 100.0:
        return [vals[-1] / 100.0]
    if len(steps) > 1 and steps[-1][0] == "multiply" and "const_100" in steps[-1][1:] and isinstance(vals[-2], float):
        return [vals[-2]]
    return []


def own_decimals(x: float) -> int:
    """Decimals for showing ``x`` by ITS OWN size (about four significant digits). Every option is formatted by its own
    size: a precision chosen from the gold's size would give away which options can be the gold."""
    a = abs(x)
    return 0 if a >= 1000 else 1 if a >= 100 else 2 if a >= 1 else 3 if a >= 0.1 else 4 if a >= 0.01 else 5


def _zero(text: str) -> str:
    return text[1:] if text.startswith("-") and not re.search(r"[1-9]", text) else text


# ---- FinQA / ConvFinQA ------------------------------------------------------------------------------------------------


def _answer_num(answer: str) -> Optional[Tuple[float, int]]:
    m = re.search(r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)", str(answer).replace("$", "").replace(" ", ""))
    if not m:
        return None
    t = m.group(0).replace(",", "")
    return float(t), len(t.split(".")[1]) if "." in t else 0


def _close(a: float, x: float, dec: int) -> bool:
    return abs(a - x) <= 0.51 * 10 ** -dec + 0.011 * abs(x)


def fin_format(gold: float, answer: str, question: str, steps: List[Step], lookup: Optional[str] = None):
    """(scale, decimals, suffix) for showing the options, or None when the annotator's answer string disagrees with
    the program result (label noise). A ratio is shown as a percentage when the annotator wrote it as one (or, with no
    answer string, when the question asks for a percentage and the program ends in a ratio). ``decimals`` is 0 for
    whole-number arithmetic, the literal's own for a number read off the report, else None = ``own_decimals``."""
    if lookup and lookup.strip().endswith("%"):
        return 100.0, len(lookup.strip()[:-1].split(".")[1]) if "." in lookup else 0, "%"
    an = _answer_num(answer)
    pct_q = bool(re.search(r"percent|%", question, re.I))
    has100 = bool(steps) and steps[-1][0] == "multiply" and "const_100" in steps[-1][1:]  # already converted
    divides = any(op == "divide" for op, _, _ in steps)
    mode = "plain"
    if an is not None:
        a, dec = an
        has_pct = "%" in str(answer)
        if has_pct and _close(a, gold * 100, dec) and not _close(a, gold, dec):
            mode = "ratio"
        elif _close(a, gold, dec):
            mode = "pct" if has_pct else "plain"
        elif _close(a, gold * 100, dec):
            mode = "ratio" if (has_pct or pct_q) else "plain"
        elif not any(_close(a, gold * f, dec) for f in (0.01, 1e3, 1e-3, 1e6, 1e-6)):
            return None
    elif pct_q:
        mode = "ratio" if (divides and not has100 and abs(gold) < 10) else "pct"
    if mode == "ratio":
        return 100.0, None, "%"
    lits = [t for op, a, b in steps if op in OPS2 for t in (a, b) if _plain_lit(t)] + ([lookup] if lookup else [])
    ints = all(re.fullmatch(r"-?[\d,]+", t.strip()) for t in lits) and all(
        op in ("add", "subtract", "multiply", "table_sum", "table_max", "table_min") for op, _, _ in steps)
    if lookup and "." in lookup:
        dec = len(lookup.strip().rstrip("%").split(".")[1])
    elif ints and abs(gold - round(gold)) < 1e-9:
        dec = 0
    else:
        dec = None
    return 1.0, dec, "%" if mode == "pct" else ""


def fin_item(question: str, program: str, table, rng, exe_ans=None, answer: str = "", neg_share: float = 0.15,
             stats: Optional[Counter] = None):
    """One FinQA-style question -> ``(question dict, label, rationale, kind)``, or a drop reason (str). ``neg_share``
    and ``stats``: see ``numeric_choice`` (the builders measure the share in a first pass)."""
    prog, lookup = str(program).strip(), None
    if "(" not in prog:
        if lit_value(prog) is None or prog.startswith("const_"):
            return "program is not executable"
        lookup, steps, vals, gold = prog, [], [], lit_value(prog)
    else:
        steps = parse_program(prog)
        if steps is None:
            return "program is not parseable"
        vals = run_program(steps, table)
        if vals is None:
            return "program is not executable"
        gold = vals[-1]
    if exe_ans is not None:
        same = (str(exe_ans).strip().lower() == gold) if isinstance(gold, str) else (
            not isinstance(exe_ans, str) and abs(gold - float(exe_ans)) <= 1e-4 * max(1.0, abs(gold)) + 1e-5)
        if not same:
            return "executed program differs from the released exe_ans"
    if steps:
        steps = prune_steps(steps)
        vals = run_program(steps, table)
    work = worked_steps(steps, vals, table)
    if isinstance(gold, str):
        q, lab = ts._fixed_choice(question, ["yes", "no"], ["yes", "no"].index(gold), rng)
        return q, lab, "%s Answer: %s." % (work, gold), "compare"
    f = fin_format(gold, answer, question, steps, lookup)
    if f is None:
        return "annotator's answer string disagrees with the program result"
    scale, dec, suffix = f

    def fmt(v):
        return _zero("%.*f" % (own_decimals(v * scale) if dec is None else dec, v * scale)) + suffix

    unit = 10.0 ** -(own_decimals(gold * scale) if dec is None else dec) / scale
    if lookup:  # a number read off the table: the other options are the cells next to it
        grid = [[cell_num(c) for c in row] for row in (table or [])]
        cells = list(dict.fromkeys(fmt(v) for v in (lit_value(t) for t in _neighbours(grid, lookup, 12)) if v is not None))
        cells = [c for c in cells if c != fmt(gold)]
        rng.shuffle(cells)
        if len(cells) < 2:
            return "fewer than three distinct options"
        q, lab = ts._fixed_choice(question, [fmt(gold)] + cells[:3], 0, rng)
        return q, lab, "Read from the report: %s." % fmt(gold), "lookup"
    mistakes = program_mistakes(steps, vals, table)
    out = numeric_choice(question, gold, mistakes, fmt, rng, integer=dec == 0 and scale == 1.0, unit=unit, scale=scale,
                         keep=percent_ratio_mistake(steps, vals, scale, rng), neg_share=neg_share, stats=stats)
    if out is None:
        return "fewer than three distinct options"
    return out[0], out[1], "%s Answer: %s." % (work, fmt(gold)), "program"


def _fin_state(ex: dict, history=None) -> str:
    obj = {"text_before_table": " ".join(ex["pre_text"]), "table": _table_text(ex["table"]),
           "text_after_table": " ".join(ex["post_text"])}
    if history:
        obj["conversation_so_far"] = history
    return _state(obj)


def _finqa_parts() -> Dict[str, list]:
    return {p: json.load(open(_gh("czyssrs/FinQA", "dataset/%s.json" % p, "finqa"), encoding="utf-8"))
            for p in ("train", "dev", "test")}


def _convfinqa_parts() -> Dict[str, list]:
    z = zipfile.ZipFile(_gh("czyssrs/ConvFinQA", "data.zip", "convfinqa"))
    return {p: json.load(z.open("data/%s.json" % p)) for p in ("train", "dev")}


def _fin_val_docs() -> set:
    """Report pages that are validation / test material in FinQA or ConvFinQA (the two share their reports)."""
    fq, cq = _finqa_parts(), _convfinqa_parts()
    return {x["filename"] for p in ("dev", "test") for x in fq[p]} | {x["filename"] for x in cq["dev"]}


def _neg_share(name: str, stats: Counter) -> float:
    """Share of negative golds among the questions where a sign mistake is derivable (first pass of a builder)."""
    share = stats["sign mistake derivable, gold negative"] / max(1, stats["sign mistake derivable"])
    NOTES[name]["negative_gold_share_where_sign_mistake_derivable"] = round(share, 4)
    return share


def _fin_rows(name: str, jobs: List[dict], rng) -> Iterator[ts.Row]:
    stats: Counter = Counter()
    for j in jobs:  # first pass: only to measure the sign statistics the option builder needs
        fin_item(j["question"], j["program"], j["ex"]["table"], random.Random(0), j["exe_ans"], j["answer"], stats=stats)
    share = _neg_share(name, stats)
    for j in jobs:
        ex, split = j["ex"], j["split"]
        item = fin_item(j["question"], j["program"], ex["table"], rng, j["exe_ans"], j["answer"], neg_share=share)
        if isinstance(item, str):
            _drop(name, split, item)
            continue
        state = _fin_state(ex, j.get("history"))
        if not _fits(state):
            _drop(name, split, "state over %d tokens" % MAX_TOKENS)
            continue
        q, lab, why, kind = item
        yield split, state, q, lab, None, {"rationale": why, "kind": j.get("prefix", "") + kind,
                                           "provenance": {"doc": ex["filename"], "source_id": j["source_id"]}}


def finqa_choice(rng) -> Iterator[ts.Row]:
    name, parts, val_docs, jobs = "finqa_choice", _finqa_parts(), _fin_val_docs(), []
    for part, split in (("train", "train"), ("dev", "val"), ("test", "val")):
        for ex in parts[part]:
            if split == "train" and ex["filename"] in val_docs:
                _drop(name, split, "report page is also validation / test material")
                continue
            qa = ex["qa"]
            jobs.append({"ex": ex, "split": split, "question": qa["question"].strip(), "program": qa["program"],
                         "exe_ans": qa.get("exe_ans"), "answer": qa.get("answer", ""), "source_id": ex["id"]})
    return _fin_rows(name, jobs, rng)


def convfinqa_choice(rng) -> Iterator[ts.Row]:
    name, parts, val_docs, jobs = "convfinqa_choice", _convfinqa_parts(), _fin_val_docs(), []
    for part, split in (("train", "train"), ("dev", "val")):
        for ex in parts[part]:
            if split == "train" and ex["filename"] in val_docs:
                _drop(name, split, "report page is also validation / test material")
                continue
            ann = ex["annotation"]
            turns, progs, answers = ann["dialogue_break"], ann["turn_program"], ann["exe_ans_list"]
            if not (len(turns) == len(progs) == len(answers)):
                _drop(name, split, "conversation fields have different lengths")
                continue
            for t in range(len(turns)):
                last = t == len(turns) - 1
                if t == 0:
                    if last:
                        _drop(name, split, "conversation has a single turn (no history)")
                    continue
                if not last and "(" not in progs[t]:
                    continue  # an earlier turn that only reads a number: kept as history, not asked
                jobs.append({"ex": ex, "split": split, "question": turns[t].strip(), "program": progs[t],
                             "exe_ans": answers[t], "answer": "", "source_id": "%s#turn%d" % (ex["id"], t),
                             "history": [{"question": turns[k], "answer": _g(answers[k])} for k in range(t)],
                             "prefix": "final_turn_" if last else "earlier_turn_"})
    return _fin_rows(name, jobs, rng)


# ---- TAT-QA -----------------------------------------------------------------------------------------------------------
SCALE_SUFFIX = {"percent": "%", "thousand": " thousand", "million": " million", "billion": " billion"}


def derivation_steps(derivation: str) -> Optional[List[Step]]:
    """A TAT-QA derivation (``(44.1 - 56.7)/56.7``) as program steps; None if it is not plain arithmetic."""
    s = str(derivation).strip()
    for a, b in (("−", "-"), ("–", "-"), ("×", "*"), ("÷", "/"), ("[", "("), ("]", ")"), ("$", ""), ("%", "")):
        s = s.replace(a, b)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    if not s or not re.fullmatch(r"[\d.\s+\-*/()]+", s):
        return None
    try:
        tree = ast.parse(s.strip(), mode="eval")
    except SyntaxError:
        return None
    steps: List[Step] = []
    ops = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide"}

    def walk(node) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.get_source_segment(s.strip(), node) or repr(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner = walk(node.operand)
            if isinstance(node.op, ast.UAdd):
                return inner
            if not inner.startswith("#"):
                return inner[1:] if inner.startswith("-") else "-" + inner
            steps.append(("multiply", inner, "const_m1"))
            return "#%d" % (len(steps) - 1)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            a, b = walk(node.left), walk(node.right)
            steps.append((ops[type(node.op)], a, b))
            return "#%d" % (len(steps) - 1)
        raise ValueError("not arithmetic")

    try:
        walk(tree.body)
    except (ValueError, RecursionError):
        return None
    return steps or None


def _norm_cell(s) -> str:
    return re.sub(r"[$,\s]", "", str(s)).lower()


def _show_cell(s) -> str:
    return re.sub(r"^\$\s*", "", str(s).strip())


def tat_item(table: List[List[str]], q: dict, rng, neg_share: float = 0.15, stats: Optional[Counter] = None):
    """One TAT-QA question -> ``(question dict, label, rationale or None, kind)``, or a drop reason (str)."""
    atype, question, deriv = q["answer_type"], q["question"].strip(), str(q.get("derivation") or "").strip()
    suffix = SCALE_SUFFIX.get(q.get("scale") or "", "")
    if atype == "arithmetic":
        text = str(q["answer"])
        try:
            ans = float(text)
        except (TypeError, ValueError):
            return "arithmetic answer is not a number"
        if "e" in text.lower():
            return "arithmetic answer is not a number"
        dec = len(text.split(".")[1]) if "." in text else 0

        def fmt(v):
            return _zero("{:,.{d}f}".format(v, d=dec)) + suffix

        steps = derivation_steps(deriv)
        if steps is None:
            return "derivation is not plain arithmetic"
        vals = run_program(steps)
        if vals is None:
            return "derivation is not executable"
        mult = next((m for m in (1, 100) if abs(vals[-1] * m - ans) <= 0.51 * 10 ** -dec + 1e-9), None)
        if mult is None:
            return "derivation does not reproduce the answer"
        if q.get("scale") == "percent" and mult == 1 and "%" not in deriv and any(op == "divide" for op, _, _ in steps):
            return "percent scale on a ratio that was not multiplied by 100"
        if mult == 100:
            steps = steps + [("multiply", "#%d" % (len(steps) - 1), "const_100")]
            vals = run_program(steps)
        out = numeric_choice(question, ans, program_mistakes(steps, vals, table), fmt, rng, integer=dec == 0,
                             unit=10.0 ** -dec, keep=percent_ratio_mistake(steps, vals, 1.0, rng), neg_share=neg_share,
                             stats=stats)
        if out is None:
            return "fewer than three distinct options"
        return out[0], out[1], "%s Answer: %s." % (worked_steps(steps, vals), fmt(ans)), "arithmetic"
    if atype == "count":
        try:
            ans = int(str(q["answer"]))
        except (TypeError, ValueError):
            return "count answer is not an integer"
        out = numeric_choice(question, float(ans), [], lambda v: "%d" % round(v), rng, integer=True, lo=0)
        if out is None:
            return "fewer than three distinct options"
        items = [x.strip() for x in deriv.split("##") if x.strip()]
        why = ("Counted: %s. Answer: %d." % ("; ".join(items), ans)) if len(items) == ans and ans else (
            "%s Answer: %d." % (deriv, ans) if deriv else None)
        return out[0], out[1], why, "count"
    if atype == "span":
        answers = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
        if len(answers) != 1:
            return "span answer has several parts"
        gold = _norm_cell(answers[0])
        hits = [(r, c) for r, row in enumerate(table) for c, cell in enumerate(row) if _norm_cell(cell) == gold and gold]
        if not hits:
            return "span answer is not a table cell (no real distractors)"
        r, c = hits[0]
        if r == 0:
            pool = [x for x in table[0][1:]]
        elif c == 0:
            pool = [row[0] for row in table[1:] if row]
        else:
            same_row = [x for i, x in enumerate(table[r]) if i not in (0, c)]
            same_col = [row[c] for i, row in enumerate(table) if i not in (0, r) and c < len(row)]
            rng.shuffle(same_row)
            rng.shuffle(same_col)
            kind = cell_num(table[r][c]) is None
            pool = [x for x in same_row + same_col if (cell_num(x) is None) == kind]
        opts, seen = [_show_cell(table[r][c])], {gold}
        if r == 0 or c == 0:
            rng.shuffle(pool)
        for x in pool:
            if _norm_cell(x) and _norm_cell(x) not in seen and len(opts) < 4:
                seen.add(_norm_cell(x))
                opts.append(_show_cell(x))
        if len(opts) < 3:
            return "fewer than three distinct options"
        ql, lab = ts._fixed_choice(question, opts, 0, rng)
        return ql, lab, ("%s, so the answer is %s." % (deriv, opts[0])) if deriv else None, "span"
    return "multi-span answer (not converted)"


def tatqa_choice(rng) -> Iterator[ts.Row]:
    name, parts = "tatqa_choice", []
    for part, split in (("train", "train"), ("dev", "val")):
        parts.append((split, json.load(open(_hub("next-tat/TAT-QA", "tatqa_dataset_%s.json" % part), encoding="utf-8"))))
    stats: Counter = Counter()
    for split, data in parts:  # first pass: the sign statistics the option builder needs
        for ex in data:
            for q in ex["questions"]:
                tat_item(ex["table"]["table"], q, random.Random(0), stats=stats)
    share = _neg_share(name, stats)
    for split, data in parts:
        for ex in data:
            table = ex["table"]["table"]
            paras = " ".join(p["text"].strip() for p in sorted(ex["paragraphs"], key=lambda p: p["order"]))
            state = _state({"table": _table_text(table), "text": paras})
            fits = _fits(state)
            for q in ex["questions"]:
                item = tat_item(table, q, rng, neg_share=share)
                if isinstance(item, str):
                    _drop(name, split, item)
                    continue
                if not fits:
                    _drop(name, split, "state over %d tokens" % MAX_TOKENS)
                    continue
                ql, lab, why, kind = item
                extra = {"kind": kind, "provenance": {"doc": ex["table"]["uid"], "source_id": q["uid"]}}
                if why:
                    extra["rationale"] = why
                yield split, state, ql, lab, None, extra


# ---- TabFact ----------------------------------------------------------------------------------------------------------
TABFACT_Q = ("Is the statement supported by the table?", "Does the table entail the statement?")


def _tabfact() -> dict:
    """Tables, statements and the official table-level split, extracted once from the pinned repository archive."""
    path = os.path.join(RAW, "tabfact", "compact.json")
    if not os.path.exists(path):
        repo = "wenhuchen/Table-Fact-Checking"
        tar = _fetch("https://codeload.github.com/%s/tar.gz/%s" % (repo, GITHUB[repo]), "tabfact/repo.tar.gz")
        out = {"tables": {}, "statements": {}, "split": {}}
        with tarfile.open(tar) as t:
            for m in t:
                p = m.name.split("/", 1)[1] if "/" in m.name else m.name
                if p.startswith("data/all_csv/") and m.isfile():
                    out["tables"][os.path.basename(p)] = t.extractfile(m).read().decode("utf-8")
                elif p in ("collected_data/r1_training_all.json", "collected_data/r2_training_all.json"):
                    out["statements"].update(json.loads(t.extractfile(m).read()))
                elif p in ("data/train_id.json", "data/val_id.json", "data/test_id.json"):
                    for tid in json.loads(t.extractfile(m).read()):
                        out["split"][tid] = os.path.basename(p).split("_")[0]
        with open(path + ".part", "w", encoding="utf-8") as f:
            json.dump(out, f)
        os.replace(path + ".part", path)
    return json.load(open(path, encoding="utf-8"))


def tabfact_rows(table_id: str, csv_text: str, entry, rng) -> List[Tuple[str, int]]:
    """(state, label) for one entailed and one refuted statement of a table (when it has both)."""
    statements, labels, caption = entry
    table = _table_text([line.split("#") for line in csv_text.strip().split("\n")])
    out = []
    for want in (1, 0):
        cands = [s for s, l in zip(statements, labels) if int(l) == want]
        if cands:
            out.append((_state({"table_caption": caption, "table": table, "statement": rng.choice(cands).strip()}), want))
    return out


def tabfact_noul(rng) -> Iterator[ts.Row]:
    name, d = "tabfact_noul", _tabfact()
    for tid in sorted(d["statements"]):
        part = d["split"].get(tid)
        if part is None or tid not in d["tables"]:
            _drop(name, "train", "table has no split or no file")
            continue
        split = "train" if part == "train" else "val"
        for state, lab in tabfact_rows(tid, d["tables"][tid], d["statements"][tid], rng):
            if not _fits(state):
                _drop(name, split, "state over %d tokens" % MAX_TOKENS)
                continue
            yield split, state, ts._noul(rng.choice(TABFACT_Q)), lab, None, {"provenance": {"doc": tid}}


# ---- TimeQA -----------------------------------------------------------------------------------------------------------
NO_ANSWER = "no answer for that time"
P_NO_ANSWER_OPTION = 0.45


def _alias(a: str, b: str) -> bool:
    x, y = re.sub(r"\W+", " ", a.lower()).strip(), re.sub(r"\W+", " ", b.lower()).strip()
    return not x or not y or x in y or y in x


def timeqa_item(question: str, targets: List[str], group_answers: List[str], rng):
    """Time-scoped question -> ``(question dict, label, kind)`` or a drop reason. Distractors are the entity's answers
    for other time scopes; none of them may be (a shorter or longer form of) a valid answer for this scope."""
    targets = [t for t in targets if t and t.strip()]
    pool = [a for a in dict.fromkeys(group_answers) if a.strip() and not any(_alias(a, t) for t in targets)]
    distinct = []
    for a in pool:  # an answer and its longer form must not both be offered
        if not any(_alias(a, b) for b in distinct):
            distinct.append(a)
    rng.shuffle(distinct)
    if not targets:
        if len(distinct) < 2:
            return "fewer than two answers from other time scopes"
        q, lab = ts._fixed_choice(question, [NO_ANSWER] + distinct[:3], 0, rng)
        return q, lab, "unanswerable"
    with_none = rng.random() < P_NO_ANSWER_OPTION
    opts = [targets[0]] + distinct[:3 - with_none] + ([NO_ANSWER] if with_none else [])
    if len(opts) < 3:
        return "fewer than two answers from other time scopes"
    q, lab = ts._fixed_choice(question, opts, 0, rng)
    return q, lab, "answerable"


def _timeqa_lines(fname: str):
    path = _gh("wenhuchen/Time-Sensitive-QA", "dataset/" + fname, "timeqa")
    with (gzip.open(path, "rt", encoding="utf-8") if fname.endswith(".gzip") else open(path, encoding="utf-8")) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def timeqa_choice(rng) -> Iterator[ts.Row]:
    name = "timeqa_choice"
    files = (("train.hard.json.gzip", "train"), ("dev.hard.json", "val"))
    groups, val_docs, fits = defaultdict(list), set(), {}
    for fname, split in files:
        for ex in _timeqa_lines(fname):
            doc, rel = ex["idx"].split("#")[:2]
            groups[(split, doc, rel)] += [t for t in ex["targets"] if t]
            if split == "val":
                val_docs.add(doc)
    for fname, split in files:
        for ex in _timeqa_lines(fname):
            doc, rel = ex["idx"].split("#")[:2]
            if split == "train" and doc in val_docs:
                _drop(name, split, "article is also in the validation file")
                continue
            item = timeqa_item(ex["question"].strip(), ex["targets"], groups[(split, doc, rel)], rng)
            if isinstance(item, str):
                _drop(name, split, item)
                continue
            if doc not in fits:
                fits[doc] = _fits(_state({"article": ex["context"]}))
            if not fits[doc]:
                _drop(name, split, "state over %d tokens" % MAX_TOKENS)
                continue
            q, lab, kind = item
            yield split, _state({"article": ex["context"]}), q, lab, None, {
                "kind": kind, "provenance": {"doc": doc, "source_id": ex["idx"]}}


# ---- pairs and votes ----------------------------------------------------------------------------------------------------


def clear_margin(a: Dict, b: Dict, margin: float, higher_is_better: bool = True) -> int:
    """+1 if ``a`` is clearly better than ``b``, -1 if ``b`` is, else 0. ``a`` / ``b`` map annotator -> score. Clear =
    the means differ by at least ``margin`` and no annotator who scored both prefers the other one."""
    if not a or not b:
        return 0
    sign = 1.0 if higher_is_better else -1.0
    diff = sign * (sum(a.values()) / len(a) - sum(b.values()) / len(b))
    if abs(diff) < margin - 1e-9:
        return 0
    way = 1 if diff > 0 else -1
    for r in set(a) & set(b):
        if sign * (a[r] - b[r]) * way < 0:
            return 0
    return way


def balance_pairs(cands: List[tuple], rng, cap: int) -> List[Tuple[tuple, int]]:
    """``cands`` = (better text, worse text, ...). Keep as many pairs where the better one is LONGER as where it is
    shorter, and within each half put the better one first exactly half the time: neither length nor position tells.
    Returns (candidate, position of the better one)."""
    longer = [c for c in cands if len(c[0]) > len(c[1])]
    shorter = [c for c in cands if len(c[0]) < len(c[1])]
    n = min(len(longer), len(shorter), cap // 2)
    out = []
    for bucket in (longer, shorter):
        rng.shuffle(bucket)
        out += [(c, i % 2) for i, c in enumerate(bucket[:n])]
    rng.shuffle(out)
    return out


def vote_target(votes: List[int], k: int, rng) -> Tuple[int, List[float]]:
    """Annotator votes over ``k`` ordered levels -> (label, vote shares). The label is the most voted level; a tie goes
    to the tied level nearest the mean vote (at random if two are equally near)."""
    counts = Counter(votes)
    target = [round(counts.get(i, 0) / len(votes), 6) for i in range(k)]  # tied levels keep exactly equal shares
    top = max(counts.values())
    tied = [i for i in sorted(counts) if counts[i] == top]
    mean = sum(votes) / len(votes)
    best = min(abs(i - mean) for i in tied)
    return rng.choice([i for i in tied if abs(abs(i - mean) - best) < 1e-9]), target


# ---- SummEval ---------------------------------------------------------------------------------------------------------
SUMMEVAL_DIMS = {
    "coherence": "How coherent is the summary? Judge whether its sentences build into a well-structured, well-organised "
                 "whole rather than a heap of loosely related facts.",
    "consistency": "How factually consistent is the summary with the article? Judge whether every statement in it is "
                   "supported by the article.",
    "fluency": "How fluent is the summary? Judge the quality of its individual sentences: grammar, formatting and "
               "readability.",
    "relevance": "How relevant is the summary? Judge whether it keeps the important content of the article and leaves "
                 "out redundant or unimportant detail.",
}
SUMMEVAL_LEVELS = ["1 (very poor)", "2 (poor)", "3 (fair)", "4 (good)", "5 (excellent)"]
SUMMEVAL_PAIR_Q = {
    "consistency": "Which summary is more factually consistent with the article (fewer statements the article does not "
                   "support)?",
    "relevance": "Which summary is more relevant: keeps the important content of the article and leaves out the rest?",
}
SUMMEVAL_MARGIN, SUMMEVAL_PAIRS_PER_ARTICLE, SUMMEVAL_VAL_PCT = 1.0, 30, 15


def _summeval() -> List[dict]:
    import pyarrow.parquet as pq

    rows = [json.loads(l) for l in open(_fetch(SUMMEVAL_URL, "summeval/model_annotations.aligned.jsonl"), encoding="utf-8")]
    need = {r["id"].split("-")[-1] for r in rows}
    t = pq.read_table(_hub("abisee/cnn_dailymail", "3.0.0/test-00000-of-00001.parquet"), columns=["id", "article"])
    art = {i: a for i, a in zip(t.column("id").to_pylist(), t.column("article").to_pylist()) if i in need}
    out = []
    for r in rows:
        key = r["id"].split("-")[-1]
        if key not in art or len(r["expert_annotations"]) < 3:
            continue
        out.append({"id": r["id"], "article": art[key], "summary": r["decoded"].strip(), "model": r["model_id"],
                    "expert": r["expert_annotations"]})
    return out


def summeval_pairs(items: List[dict], dim: str, margin: float = SUMMEVAL_MARGIN) -> List[tuple]:
    """(better summary, worse summary) for every pair of summaries of ONE article with a clear expert margin on
    ``dim``."""
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a["summary"] == b["summary"]:
                continue
            way = clear_margin({k: e[dim] for k, e in enumerate(a["expert"])}, {k: e[dim] for k, e in enumerate(b["expert"])},
                               margin)
            if way:
                out.append((a["summary"], b["summary"]) if way > 0 else (b["summary"], a["summary"]))
    return out


def summeval_pair(rng) -> Iterator[ts.Row]:
    name, by_art = "summeval_pair", defaultdict(list)
    for r in _summeval():
        by_art[r["id"]].append(r)
    cands = {"train": [], "val": []}
    for aid in sorted(by_art):
        split = _split(aid, SUMMEVAL_VAL_PCT)
        for dim in SUMMEVAL_PAIR_Q:
            pairs = summeval_pairs(by_art[aid], dim)
            rng.shuffle(pairs)
            _drop(name, split, "over %d pairs for one article and criterion" % SUMMEVAL_PAIRS_PER_ARTICLE,
                  max(0, len(pairs) - SUMMEVAL_PAIRS_PER_ARTICLE))
            cands[split] += [(a, b, aid, dim) for a, b in pairs[:SUMMEVAL_PAIRS_PER_ARTICLE]]
    for split, cap in (("train", TRAIN_CAP), ("val", VAL_CAP)):
        kept = balance_pairs(cands[split], rng, cap)
        _drop(name, split, "left out to balance length and position (or over the cap)", len(cands[split]) - len(kept))
        for (better, worse, aid, dim), pos in kept:
            a, b = (better, worse) if pos == 0 else (worse, better)
            state = _state({"article": by_art[aid][0]["article"], "summary_A": a, "summary_B": b})
            if not _fits(state):
                _drop(name, split, "state over %d tokens" % MAX_TOKENS)
                continue
            q = {"type": "choice", "instructions": SUMMEVAL_PAIR_Q[dim], "criteria": ["summary A", "summary B"]}
            yield split, state, q, pos, None, {"kind": dim, "provenance": {"doc": aid}}


def summeval_score(rng) -> Iterator[ts.Row]:
    name = "summeval_score"
    for r in _summeval():
        split = _split(r["id"], SUMMEVAL_VAL_PCT)
        state = _state({"article": r["article"], "summary": r["summary"]})
        if not _fits(state):
            _drop(name, split, "state over %d tokens" % MAX_TOKENS, len(SUMMEVAL_DIMS))
            continue
        for dim, ins in SUMMEVAL_DIMS.items():
            lab, target = vote_target([int(e[dim]) - 1 for e in r["expert"]], 5, rng)
            q = {"type": "score", "instructions": ins, "criteria": SUMMEVAL_LEVELS}
            yield split, state, q, lab, target, {"kind": dim, "provenance": {"doc": r["id"], "source_id": r["model"]}}


# ---- FRANK ------------------------------------------------------------------------------------------------------------
FRANK_ERRORS = {
    "RelE": "wrong relation: the action or predicate is not what the article says",
    "EntE": "wrong entity: the subject or object of the statement is wrong",
    "CircE": "wrong circumstance: the time, place, manner or amount is wrong",
    "CorefE": "wrong reference: a pronoun or reference points to the wrong or no antecedent",
    "LinkE": "wrong link: statements are connected by a cause or order the article does not give",
    "OutE": "not in the article: the statement cannot be verified from the article",
    "GramE": "ungrammatical: the sentence is too garbled to have a clear meaning",
}
FRANK_Q = ("Is the marked summary sentence factually consistent with the article?",
           "Does the article support everything the marked summary sentence says?")


def frank_sentence(annotations: Dict[str, List[str]]) -> Tuple[float, Optional[str]]:
    """(share of annotators who found NO factual error, the error type at least two annotators agree on or None)."""
    votes = [set(v) for v in annotations.values()]
    ok = sum(1 for v in votes if v <= {"NoE"}) / len(votes)
    types = Counter(t for v in votes for t in v if t in FRANK_ERRORS).most_common(2)
    agreed = types[0][0] if types and types[0][1] >= 2 and (len(types) == 1 or types[1][1] < types[0][1]) else None
    return ok, agreed


def frank_factual(rng) -> Iterator[ts.Row]:
    name = "frank_factual"
    data = json.load(open(_gh("artidoro/frank", "data/human_annotations_sentence.json", "frank"), encoding="utf-8"))
    noul, choice = defaultdict(list), []
    for ex in data:
        split = "val" if ex["split"] == "valid" else "train"  # FRANK has valid / test only: test is the larger part
        if len(ex["hash"]) != 40:  # CNN/DM ids are 40-hex story hashes; the others are XSum (BBC) articles
            _drop(name, split, "XSum article (licence of the BBC articles not stated)", len(ex["summary_sentences"]))
            continue
        summary = " ".join(ex["summary_sentences"])
        for sent, ann in zip(ex["summary_sentences"], ex["summary_sentences_annotations"]):
            if not ann:
                continue
            state = _state({"article": ex["article"], "summary": summary, "marked_sentence": sent})
            if not _fits(state):
                _drop(name, split, "state over %d tokens" % MAX_TOKENS)
                continue
            ok, agreed = frank_sentence(ann)
            extra = {"provenance": {"doc": ex["hash"], "source_id": ex["model_name"]}}
            noul[(split, int(ok > 0.5))].append((state, [round(1 - ok, 4), round(ok, 4)], extra))
            if ok < 0.5 and agreed:
                choice.append((split, state, agreed, extra))
    for split in ("train", "val"):  # balance the yes / no part here (the set also holds choice rows)
        pos, neg = noul[(split, 1)], noul[(split, 0)]
        n = min(len(pos), len(neg))
        _drop(name, split, "consistent sentences left out to balance yes / no", len(pos) - n)
        _drop(name, split, "inconsistent sentences left out to balance yes / no", len(neg) - n)
        for lab, rows in ((1, pos), (0, neg)):
            rng.shuffle(rows)
            for state, target, extra in rows[:n]:
                yield split, state, ts._noul(rng.choice(FRANK_Q)), lab, target, dict(extra, kind="consistent")
    for split, state, agreed, extra in choice:
        others = rng.sample([k for k in FRANK_ERRORS if k != agreed], 3)
        q, lab = ts._fixed_choice("The marked summary sentence is not factually consistent with the article. What kind "
                                  "of error is it?", [FRANK_ERRORS[agreed]] + [FRANK_ERRORS[k] for k in others], 0, rng)
        yield split, state, q, lab, None, dict(extra, kind="error_type")


# ---- WMT MQM ----------------------------------------------------------------------------------------------------------
MQM_FILES = [  # (path in the repository, split, source language, target language)
    ("newstest2020/ende/mqm_newstest2020_ende.tsv", "train", "English", "German"),
    ("newstest2020/zhen/mqm_newstest2020_zhen.tsv", "train", "Chinese", "English"),
    ("newstest2021/ende/mqm_newstest2021_ende.tsv", "train", "English", "German"),
    ("newstest2021/zhen/mqm_newstest2021_zhen.tsv", "train", "Chinese", "English"),
    ("ted/ende/mqm_ted_ende.tsv", "train", "English", "German"),
    ("ted/zhen/mqm_ted_zhen.tsv", "train", "Chinese", "English"),
    ("generalMT2022/ende/mqm_generalMT2022_ende.tsv", "val", "English", "German"),
    ("generalMT2022/zhen/mqm_generalMT2022_zhen.tsv", "val", "Chinese", "English"),
]
CLOSED_SYSTEM = re.compile(r"gpt|openai|claude|anthropic|gemini|bard|palm|command-?r|cohere", re.I)
MQM_MARGIN, MQM_PAIRS_PER_SEGMENT = 5.0, 3
MQM_PAIR_Q = "Which translation of the source is better (fewer and less severe translation errors)?"
MQM_MAJOR_Q = ("Does the translation contain a major error (one that changes the meaning or makes the text hard to "
               "understand)?", "Is there a major translation error in this translation?")


def mqm_weight(category: str, severity: str) -> float:
    """MQM penalty of one annotated error (Freitag et al. 2021): major 5, minor 1, minor punctuation 0.1,
    non-translation 25, neutral / no error 0."""
    c, s = category.strip().lower(), severity.strip().lower()
    if c.startswith("non-translation"):
        return 25.0
    if s == "major":
        return 5.0
    if s == "minor":
        return 0.1 if c == "fluency/punctuation" else 1.0
    return 0.0


def mqm_segments(rows: List[dict]) -> Dict[tuple, dict]:
    """Rating rows -> {(doc, seg_id): {"source", "systems": {system: {"target", "penalty": {rater: points}, "major":
    {rater: bool}}}}}. Error-span marks are removed; outputs with an attention-check row or a source error are left
    out."""
    segs: Dict[tuple, dict] = {}
    bad = set()
    clean = lambda s: re.sub(r"\s+", " ", re.sub(r"</?v>", "", s)).strip()  # noqa: E731
    for r in rows:
        key = (r["doc"], r["seg_id"])
        seg = segs.setdefault(key, {"source": clean(r["source"]), "systems": {}})
        out = seg["systems"].setdefault(r["system"], {"target": clean(r["target"]), "penalty": defaultdict(float),
                                                     "major": defaultdict(bool)})
        sev, cat = r["severity"].strip().lower(), r["category"].strip().lower()
        if "hotw" in sev or cat.startswith("source error"):
            bad.add((key, r["system"]))
        out["penalty"][r["rater"]] += mqm_weight(cat, sev)
        out["major"][r["rater"]] |= (sev == "major" or cat.startswith("non-translation"))
    for key, system in bad:
        segs[key]["systems"].pop(system, None)
    return segs


def _mqm_rows(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 9:
                out.append(dict(zip(head[:9], p[:9])))
    return out


def _mqm(name: str):
    """(split, source language, target language, file tag, segments) per file; closed-model systems removed."""
    for path, split, src, tgt in MQM_FILES:
        rows = _mqm_rows(_gh("google/wmt-mqm-human-evaluation", path, "mqm"))
        closed = {r["system"] for r in rows if CLOSED_SYSTEM.search(r["system"])}
        if closed:
            _drop(name, split, "ratings of closed-model systems %s" % sorted(closed), sum(r["system"] in closed for r in rows))
        yield split, src, tgt, os.path.basename(path)[4:-4], mqm_segments([r for r in rows if r["system"] not in closed])


def mqm_pairs(seg: dict, margin: float = MQM_MARGIN) -> List[tuple]:
    """(better translation, worse translation) for the outputs of two systems with a clear margin in MQM penalty."""
    systems, seen = [], set()
    for s in sorted(seg["systems"]):
        o = seg["systems"][s]
        if o["target"] and o["target"] not in seen:  # identical outputs of two systems: keep one
            seen.add(o["target"])
            systems.append(o)
    out = []
    for i in range(len(systems)):
        for j in range(i + 1, len(systems)):
            a, b = systems[i], systems[j]
            way = clear_margin(dict(a["penalty"]), dict(b["penalty"]), margin, higher_is_better=False)
            if way:
                out.append((a["target"], b["target"]) if way > 0 else (b["target"], a["target"]))
    return out


def mqm_pair(rng) -> Iterator[ts.Row]:
    name = "mqm_pair"
    cands = {"train": [], "val": []}
    val_sources = set()
    for split, src, tgt, tag, segs in _mqm(name):
        for key in sorted(segs):
            seg = segs[key]
            if split == "val":
                val_sources.add(seg["source"])
            pairs = mqm_pairs(seg)
            rng.shuffle(pairs)
            _drop(name, split, "over %d pairs for one segment" % MQM_PAIRS_PER_SEGMENT, max(0, len(pairs) - MQM_PAIRS_PER_SEGMENT))
            cands[split] += [(a, b, seg["source"], src, tgt, "%s/%s/%s" % (tag, key[0], key[1]))
                             for a, b in pairs[:MQM_PAIRS_PER_SEGMENT]]
    n = len(cands["train"])
    cands["train"] = [c for c in cands["train"] if c[2] not in val_sources]
    _drop(name, "train", "source segment also in the validation files", n - len(cands["train"]))
    for split, cap in (("train", TRAIN_CAP), ("val", VAL_CAP)):
        kept = balance_pairs(cands[split], rng, cap)
        _drop(name, split, "left out to balance length and position (or over the cap)", len(cands[split]) - len(kept))
        for (better, worse, source, src, tgt, sid), pos in kept:
            a, b = (better, worse) if pos == 0 else (worse, better)
            state = _state({"source_language": src, "target_language": tgt, "source": source, "translation_A": a,
                            "translation_B": b})
            q = {"type": "choice", "instructions": MQM_PAIR_Q, "criteria": ["translation A", "translation B"]}
            yield split, state, q, pos, None, {"kind": "%s-%s" % (src, tgt), "provenance": {"doc": sid}}


def mqm_major_error(rng) -> Iterator[ts.Row]:
    name = "mqm_major_error"
    rows = defaultdict(list)
    val_sources = set()
    for split, src, tgt, tag, segs in _mqm(name):
        for key in sorted(segs):
            seg, seen = segs[key], set()
            if split == "val":
                val_sources.add(seg["source"])
            for s in sorted(seg["systems"]):
                o = seg["systems"][s]
                if not o["target"] or o["target"] in seen:
                    continue
                seen.add(o["target"])
                flags = [bool(o["major"][r]) for r in sorted(o["penalty"])]
                p = sum(flags) / len(flags)
                state = _state({"source_language": src, "target_language": tgt, "source": seg["source"],
                                "translation": o["target"]})
                target = [round(1 - p, 4), round(p, 4)] if len(flags) > 1 else None
                rows[(split, src + "-" + tgt, int(p >= 0.5))].append(
                    (state, target, seg["source"], "%s/%s/%s" % (tag, key[0], key[1])))
    for key in sorted(rows):
        if key[0] == "train":
            k = len(rows[key])
            rows[key] = [r for r in rows[key] if r[2] not in val_sources]
            _drop(name, "train", "source segment also in the validation files", k - len(rows[key]))
    for (split, lp, lab), rs in sorted(rows.items()):
        n = min(len(rs), len(rows.get((split, lp, 1 - lab), [])))  # error rates differ by language pair: balance inside each
        rng.shuffle(rs)
        _drop(name, split, "left out to balance yes / no inside %s" % lp, len(rs) - n)
        for state, target, _, sid in rs[:n]:
            yield split, state, ts._noul(rng.choice(MQM_MAJOR_Q)), lab, target, {"kind": lp, "provenance": {"doc": sid}}


# ---- FaithDial / BEGIN ------------------------------------------------------------------------------------------------
FAITH_Q = ("Is the response faithful to the knowledge: does it say only what the knowledge supports?",
           "Is everything the response claims supported by the knowledge?")


def faithdial_rows(utt: dict) -> List[Tuple[dict, int]]:
    """(state, label) rows of one FaithDial turn: the annotator-edited faithful response (yes) and, when the original
    Wizard response was marked as hallucination, that original (no)."""
    if "Generic" in utt["BEGIN"] or not (utt.get("response") or "").strip():
        return []
    base = {"knowledge": utt["knowledge"], "dialogue": utt["history"][-3:]}
    out = [(dict(base, response=utt["response"].strip()), 1)]
    orig = (utt.get("original_response") or "").strip()
    if orig and "Hallucination" in utt["BEGIN"] and orig != utt["response"].strip():
        out.append((dict(base, response=orig), 0))
    return out


def faithdial_noul(rng) -> Iterator[ts.Row]:
    name = "faithdial_noul"
    load = lambda f: json.load(open(_hub("McGill-NLP/FaithDial", "data/%s.json" % f), encoding="utf-8"))  # noqa: E731
    train = load("train")
    train_knowledge = {u["knowledge"] for d in train for u in d["utterances"]}
    parts = [("train", "train", train)] + [("val", f, load(f)) for f in ("valid_topic_split", "test_topic_split")]
    for split, fname, dialogs in parts:
        for d in dialogs:
            for i, u in enumerate(d["utterances"]):
                if split == "val" and u["knowledge"] in train_knowledge:
                    _drop(name, split, "knowledge snippet also in train")
                    continue
                rows = faithdial_rows(u)
                if not rows:
                    _drop(name, split, "generic or empty response")
                for obj, lab in rows:
                    yield split, _state(obj), ts._noul(rng.choice(FAITH_Q)), lab, None, {
                        "kind": "edited" if lab else "original",
                        "provenance": {"doc": "%s/%s" % (fname, d["dialog_idx"]), "source_id": str(i)}}


BEGIN_LABELS = {
    "Fully attributable": "fully attributable to the knowledge",
    "Not fully attributable": "not fully attributable to the knowledge",
    "Generic": "generic (too vague to check against the knowledge)",
}
BEGIN_Q = "How does the response relate to the knowledge it should be grounded in?"
BEGIN_VAL_PCT, BEGIN_MAX_SHARE = 15, 0.45


def begin_attribution(rng) -> Iterator[ts.Row]:
    """BEGIN has no train split (dev / test are a 10 / 90 row split that shares knowledge documents), so the rows of
    both files are split here by knowledge document."""
    name, rows = "begin_attribution", defaultdict(list)
    for part, tag in (("cmu-dog", "cmu"), ("wow", "wow")):
        for f in ("begin_dev_%s.tsv" % tag, "begin_test_%s.tsv" % tag):
            with open(_gh("google/BEGIN-dataset", "%s/%s" % (part, f), "begin"), encoding="utf-8") as fh:
                for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                    split = _split(r["knowledge"], BEGIN_VAL_PCT)
                    if r["begin_label"] not in BEGIN_LABELS or not r["response"].strip():
                        _drop(name, split, "no label or empty response")
                        continue
                    state = _state({"knowledge": r["knowledge"], "previous_turn": r["message"], "response": r["response"]})
                    rows[(split, r["begin_label"])].append((state, {
                        "kind": tag, "provenance": {"doc": hashlib.md5(r["knowledge"].encode()).hexdigest()[:12],
                                                    "source_id": r["model_name"].lower()}}))
    for split in ("train", "val"):
        major = "Not fully attributable"
        rest = sum(len(rows[(split, l)]) for l in BEGIN_LABELS if l != major)
        keep = min(len(rows[(split, major)]), int(rest * BEGIN_MAX_SHARE / (1 - BEGIN_MAX_SHARE)))
        rng.shuffle(rows[(split, major)])
        _drop(name, split, "'not fully attributable' rows left out (class capped at %d%%)" % (100 * BEGIN_MAX_SHARE),
              len(rows[(split, major)]) - keep)
        rows[(split, major)] = rows[(split, major)][:keep]
        for label in BEGIN_LABELS:
            for state, extra in rows[(split, label)]:
                q, lab = ts._fixed_choice(BEGIN_Q, list(BEGIN_LABELS.values()), list(BEGIN_LABELS).index(label), rng)
                yield split, state, q, lab, None, extra


# ---- FeedbackQA -------------------------------------------------------------------------------------------------------
FEEDBACK_LEVELS = ["bad", "could be improved", "acceptable", "excellent"]
FEEDBACK_Q = "How well does the answer respond to the question? Judge whether it gives the information asked for."
FEEDBACK_VAL_PCT = 25


def feedback_vote(ratings: List[str], rng) -> Optional[Tuple[int, List[float]]]:
    """Rater labels -> (label, vote shares), or None when two raters are two or more levels apart with no majority."""
    votes = [FEEDBACK_LEVELS.index(r.strip().lower()) for r in ratings]
    if max(Counter(votes).values()) == 1 and max(votes) - min(votes) >= 2:
        return None
    return vote_target(votes, len(FEEDBACK_LEVELS), rng)


def feedbackqa_score(rng) -> Iterator[ts.Row]:
    """The release is split by question, and one passage is rated for several questions; passages are split here as
    well (hash), so no passage is on both sides: train rows keep train passages, validation rows validation passages."""
    name = "feedbackqa_score"
    for part, split in (("train", "train"), ("valid", "val"), ("test", "val")):
        for ex in json.load(open(_hub("McGill-NLP/feedbackQA", "data/feedback_%s.json" % part), encoding="utf-8")):
            pas = ex["passage"]
            pid = "%s/%s" % (pas["source"], pas["passage_id"])
            if _split(pid, FEEDBACK_VAL_PCT) != split:
                _drop(name, split, "passage belongs to the other split")
                continue
            try:
                vote = feedback_vote(ex["rating"], rng)
            except ValueError:
                _drop(name, split, "unknown rating label")
                continue
            if vote is None:
                _drop(name, split, "raters two or more levels apart, no majority")
                continue
            ref = pas["reference"]
            answer = "\n".join([ref.get("page_title") or ""] + list(ref.get("section_headers") or []) +
                               [ref.get("section_content") or ""]).strip()
            state = _state({"question": ex["question"], "answer": answer})
            if not _fits(state):
                _drop(name, split, "state over %d tokens" % MAX_TOKENS)
                continue
            q = {"type": "score", "instructions": FEEDBACK_Q, "criteria": FEEDBACK_LEVELS}
            yield split, state, q, vote[0], vote[1], {"kind": ex.get("domain") or pas["source"], "provenance": {"doc": pid}}


# ---- registry ---------------------------------------------------------------------------------------------------------


def _g_src(repo: str, licence: str) -> dict:
    return {"source": "github.com/" + repo, "revision": GITHUB[repo], "license": licence}


def _h_src(repo: str, licence: str) -> dict:
    return {"source": "huggingface.co/datasets/" + repo, "revision": HUB[repo], "license": licence}


_FINQA = _g_src("czyssrs/FinQA", "MIT (GitHub LICENSE); Hub card ibm-research/finqa CC BY 4.0")
_CONV = _g_src("czyssrs/ConvFinQA", "MIT (GitHub LICENSE)")
_MQM = _g_src("google/wmt-mqm-human-evaluation", "Apache-2.0 (GitHub LICENSE)")
_CNNDM = _h_src("abisee/cnn_dailymail", "Apache-2.0 (Hub card) for the articles")
SOURCES: Dict[str, dict] = {
    "finqa_choice": {"fn": finqa_choice, "repo": "github.com/czyssrs/FinQA", "sources": [_FINQA, _CONV],
                     "license": "MIT (GitHub czyssrs/FinQA LICENSE; Hub card ibm-research/finqa: CC BY 4.0)",
                     "use": "finance arithmetic over report text + table; expert programs executed"},
    "convfinqa_choice": {"fn": convfinqa_choice, "repo": "github.com/czyssrs/ConvFinQA", "sources": [_CONV, _FINQA],
                         "license": "MIT (GitHub czyssrs/ConvFinQA LICENSE)",
                         "use": "finance arithmetic in a conversation (history in the state)"},
    "tatqa_choice": {"fn": tatqa_choice, "repo": "next-tat/TAT-QA",
                     "sources": [_h_src("next-tat/TAT-QA", "CC BY 4.0 (Hub card; GitHub NExTplusplus/TAT-QA README)")],
                     "license": "CC BY 4.0 (Hub card next-tat/TAT-QA and GitHub README)",
                     "use": "arithmetic / count / cell lookup over a table + paragraphs"},
    "tabfact_noul": {"fn": tabfact_noul, "repo": "github.com/wenhuchen/Table-Fact-Checking", "balance": True,
                     "sources": [_g_src("wenhuchen/Table-Fact-Checking",
                                        "MIT (GitHub LICENSE); Hub card wenhu/tab_fact CC BY 4.0; tables from Wikipedia")],
                     "license": "CC BY 4.0 (Hub card wenhu/tab_fact); GitHub LICENSE MIT; tables are Wikipedia text (CC BY-SA)",
                     "use": "table fact verification"},
    "timeqa_choice": {"fn": timeqa_choice, "repo": "github.com/wenhuchen/Time-Sensitive-QA",
                      "sources": [_g_src("wenhuchen/Time-Sensitive-QA", "BSD-3-Clause (GitHub LICENSE, README)")],
                      "license": "BSD-3-Clause (GitHub wenhuchen/Time-Sensitive-QA); articles are Wikipedia text (CC BY-SA)",
                      "use": "time-scoped reading: answer valid at the asked time"},
    "summeval_pair": {"fn": summeval_pair, "repo": "github.com/Yale-LILY/SummEval",
                      "sources": [_g_src("Yale-LILY/SummEval", "MIT (GitHub LICENSE); annotations file " + SUMMEVAL_URL), _CNNDM],
                      "license": "MIT (GitHub Yale-LILY/SummEval) annotations and system summaries; CNN/DailyMail "
                                 "articles Apache-2.0 (Hub card abisee/cnn_dailymail)",
                      "use": "judge: which summary is more consistent / relevant (expert margin)"},
    "summeval_score": {"fn": summeval_score, "repo": "github.com/Yale-LILY/SummEval",
                       "sources": [_g_src("Yale-LILY/SummEval", "MIT (GitHub LICENSE); annotations file " + SUMMEVAL_URL), _CNNDM],
                       "license": "MIT (GitHub Yale-LILY/SummEval) annotations and system summaries; CNN/DailyMail "
                                  "articles Apache-2.0 (Hub card abisee/cnn_dailymail)",
                       "use": "judge: rate a summary 1-5 on one criterion, expert vote shares as soft target"},
    "frank_factual": {"fn": frank_factual, "repo": "github.com/artidoro/frank",
                      "sources": [_g_src("artidoro/frank", "MIT (GitHub LICENSE)"), _CNNDM],
                      "license": "MIT (GitHub artidoro/frank); CNN/DailyMail articles Apache-2.0 (Hub card "
                                 "abisee/cnn_dailymail); XSum rows left out (Hub card licence: unknown)",
                      "use": "judge: is a summary sentence factual; which kind of factual error"},
    "mqm_pair": {"fn": mqm_pair, "repo": "github.com/google/wmt-mqm-human-evaluation", "sources": [_MQM],
                 "license": "Apache-2.0 (GitHub google/wmt-mqm-human-evaluation)",
                 "use": "judge: which translation has the lower MQM penalty (professional translators)"},
    "mqm_major_error": {"fn": mqm_major_error, "repo": "github.com/google/wmt-mqm-human-evaluation", "sources": [_MQM],
                        "license": "Apache-2.0 (GitHub google/wmt-mqm-human-evaluation)", "balance": True,
                        "use": "judge: does the translation contain a major error"},
    "faithdial_noul": {"fn": faithdial_noul, "repo": "McGill-NLP/FaithDial", "balance": True,
                       "sources": [_h_src("McGill-NLP/FaithDial", "MIT (Hub card; GitHub McGill-NLP/FaithDial LICENSE)")],
                       "license": "MIT (Hub card McGill-NLP/FaithDial and GitHub LICENSE)",
                       "use": "judge: is a dialogue response faithful to its knowledge"},
    "begin_attribution": {"fn": begin_attribution, "repo": "github.com/google/BEGIN-dataset",
                          "sources": [_g_src("google/BEGIN-dataset", "CC BY 4.0 (cmu-dog/LICENSE, wow/LICENSE); "
                                                                     "topicalchat part not used")],
                          "license": "CC BY 4.0 (GitHub google/BEGIN-dataset cmu-dog/ and wow/ LICENSE); TopicalChat part not used",
                          "use": "judge: attribution class of a knowledge-grounded response"},
    "feedbackqa_score": {"fn": feedbackqa_score, "repo": "McGill-NLP/feedbackQA",
                         "sources": [_h_src("McGill-NLP/feedbackQA", "Apache-2.0 (Hub card)")],
                         "license": "Apache-2.0 (Hub card McGill-NLP/feedbackQA)",
                         "use": "judge: rate an answer passage for a question, rater shares as soft target"},
}
for _spec in SOURCES.values():
    _spec.setdefault("train_cap", TRAIN_CAP)
    _spec.setdefault("val_cap", VAL_CAP)


def build(name: str, root: str = DATA_ROOT, log=print) -> dict:
    """``visionjev_textsets.build`` plus what this module knows: the pinned revisions and licences actually read, the
    rows each filter dropped, and the state length in tokens."""
    spec = SOURCES[name]
    DROPS.pop(name, None)
    NOTES.pop(name, None)
    man = ts.build(name, root, spec=spec, log=log)
    tokens = {}
    for split in ("train", "val"):
        path = os.path.join(root, name, split + ".jsonl")
        if os.path.exists(path):
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            step = max(1, len(rows) // 500)
            n = [_ntok(r["state_text"] or "") for r in rows[::step]]
            tokens[split] = {"median": int(statistics.median(n)), "max_of_sample": max(n), "sampled": len(n)}
    man["revision"] = spec["sources"][0]["revision"]
    man["sources"] = spec["sources"]
    man["license"]["verified"] = VERIFIED
    man["dropped"] = dict(sorted(DROPS[name].items()))
    man["notes"] = dict(NOTES[name])
    man["state_tokens"] = tokens
    man["max_state_tokens"] = MAX_TOKENS
    with open(os.path.join(root, name, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    log("%s: dropped %s" % (name, man["dropped"]))
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (list(SOURCES) if a.names == ["all"] else a.names):
        try:
            build(n, a.root, log=lambda m: print(m, flush=True))
        except Exception as e:  # one broken source must not stop the rest
            print("%s: FAILED %r" % (n, e), flush=True)
