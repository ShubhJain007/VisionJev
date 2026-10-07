"""Commercially usable text and code decision sets (licenses checked 2026-09-25, see CONTEXT.md section 9).

One generic builder: every source is a function that yields ``(split, state_text, question, label, target|None)``;
the builder caps the splits, balances ``noul`` sets that need it, writes ``<root>/<name>/{train,val}.jsonl`` and a
manifest with the license. Many-class sources become ``choice`` questions over a *sampled* option set (the gold plus
distractors drawn from the real label space, 4-8 options, never generated), which also varies the option count.

Sets marked ``eval_only`` (e.g. MMLU) are written as val only: they are standard benchmarks, so training on them would
spoil the comparison. Names in ``SOURCES``; build with

    python -m laya.visionjev_textsets banking77 clinc_intent clinc_scope civil_toxicity ...
    python -m laya.visionjev_textsets all
"""
import argparse
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple

from .visionjev_data import DATA_ROOT, _hub_revision

Row = Tuple[str, str, dict, int, Optional[List[float]]]
TRAIN_CAP, VAL_CAP = 8000, 1000


def _ds(repo, config=None, split="train", streaming=False, revision=None):
    import datasets

    return datasets.load_dataset(repo, config, split=split, streaming=streaming, revision=revision)


def _human(label: str) -> str:
    return re.sub(r"[_\-]+", " ", label).strip()


def _sampled_choice(ins: str, gold: str, space: List[str], rng: random.Random, kmin=4, kmax=8, text: str = ""):
    """Gold + k-1 real labels. With ``text``, half the distractors are HARD: labels sharing the most words with the
    text, so keyword matching alone cannot find the gold (audit: random distractors let word overlap score ~0.65)."""
    k = min(len(space), rng.randint(kmin, kmax))
    others = [s for s in space if s != gold]
    hard = []
    if text:
        tw = set(re.findall(r"[a-z0-9]+", text.lower()))
        ranked = sorted(others, key=lambda s: -len(set(re.findall(r"[a-z0-9]+", s.lower())) & tw))
        hard = [s for s in ranked[: 3 * k] if set(re.findall(r"[a-z0-9]+", s.lower())) & tw]
        hard = rng.sample(hard, min(len(hard), (k - 1) // 2 + 1))
    rest = [s for s in others if s not in hard]
    opts = [gold] + hard + rng.sample(rest, k - 1 - len(hard))
    rng.shuffle(opts)
    return {"type": "choice", "instructions": ins, "criteria": opts}, opts.index(gold)


def _fixed_choice(ins: str, opts: List[str], gold: int, rng: random.Random, shuffle=True):
    order = list(range(len(opts)))
    if shuffle:
        rng.shuffle(order)
    return {"type": "choice", "instructions": ins, "criteria": [opts[i] for i in order]}, order.index(gold)


def _noul(ins: str) -> dict:
    return {"type": "noul", "instructions": ins, "criteria": None}


# ---- text -----------------------------------------------------------------------------------------------------


BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/%s.csv"


def banking77(rng) -> Iterator[Row]:
    """The Hub repo is a loading script (no parquet copy); read PolyAI's CSVs (GitHub, CC BY 4.0) directly."""
    import csv
    import io
    import urllib.request

    data = {}
    for part in ("train", "test"):
        text = urllib.request.urlopen(BANKING77 % part, timeout=60).read().decode("utf-8")
        data[part] = list(csv.DictReader(io.StringIO(text)))
    space = sorted({_human(r["category"]) for r in data["train"]})
    ins = ("Which intent best describes this banking customer's message?", "Route this message: what is the customer asking about?")
    for part, split in (("train", "train"), ("test", "val")):
        for r in data[part]:
            q, lab = _sampled_choice(rng.choice(ins), _human(r["category"]), space, rng, text=r["text"])
            yield split, json.dumps({"message": r["text"]}), q, lab, None


def _clinc(rng, scope: bool) -> Iterator[Row]:
    feats = _ds("clinc/clinc_oos", "plus", split="train").features["intent"].names
    oos = feats.index("oos")
    space = [_human(n) for i, n in enumerate(feats) if i != oos]
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in _ds("clinc/clinc_oos", "plus", split=hub):
            state = json.dumps({"user_request": ex["text"]})
            if scope:
                q = _noul(rng.choice(("Is this request something a general-purpose virtual assistant supports?",
                                      "Is this request within the assistant's supported services?")))
                yield split, state, q, int(ex["intent"] != oos), None
            elif ex["intent"] != oos:
                q, lab = _sampled_choice("Which intent does this request express?", _human(feats[ex["intent"]]), space, rng, text=ex["text"])
                yield split, state, q, lab, None


def clinc_intent(rng):
    return _clinc(rng, scope=False)


def clinc_scope(rng):
    return _clinc(rng, scope=True)


def civil_toxicity(rng) -> Iterator[Row]:
    """Toxicity with the rater fraction as a soft target (human vote shares: calibration data)."""
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in _ds("google/civil_comments", split=hub, streaming=True):
            p = float(ex["toxicity"])
            q = _noul(rng.choice(("Is this comment toxic (rude, disrespectful or unreasonable)?",
                                  "Would most readers find this comment toxic?")))
            yield split, json.dumps({"comment": ex["text"][:1500]}), q, int(p >= 0.5), [round(1 - p, 4), round(p, 4)]


def sms_spam(rng) -> Iterator[Row]:
    for i, ex in enumerate(_ds("ucirvine/sms_spam", split="train")):
        split = "val" if i % 10 == 0 else "train"
        yield split, json.dumps({"sms": ex["sms"]}), _noul("Is this text message spam?"), int(ex["label"]), None


def prompt_injection(rng) -> Iterator[Row]:
    for hub, split in (("train", "train"), ("test", "val")):
        for ex in _ds("deepset/prompt-injections", split=hub):
            q = _noul(rng.choice(("Does this input try to override or inject instructions into the AI system?",
                                  "Is this a prompt-injection attempt?")))
            yield split, json.dumps({"user_input": ex["text"]}), q, int(ex["label"]), None


def jailbreak(rng) -> Iterator[Row]:
    for hub, split in (("train", "train"), ("test", "val")):
        for ex in _ds("jackhhao/jailbreak-classification", split=hub):
            q = _noul("Is this prompt a jailbreak attempt (trying to get the model to ignore its rules)?")
            yield split, json.dumps({"prompt": ex["prompt"][:2000]}), q, int(ex["type"] == "jailbreak"), None


def mmlu(rng) -> Iterator[Row]:
    for ex in _ds("cais/mmlu", "all", split="test"):
        q, lab = _fixed_choice(ex["question"], [str(c) for c in ex["choices"]], int(ex["answer"]), rng)
        yield "val", json.dumps({"subject": _human(ex["subject"])}), q, lab, None


def commonsense_qa(rng) -> Iterator[Row]:
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in _ds("tau/commonsense_qa", split=hub):
            labels = ex["choices"]["label"]
            if ex["answerKey"] not in labels:
                continue
            q, lab = _fixed_choice(ex["question"], ex["choices"]["text"], labels.index(ex["answerKey"]), rng)
            yield split, "", q, lab, None


def openbookqa(rng) -> Iterator[Row]:
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in _ds("allenai/openbookqa", "main", split=hub):
            labels = ex["choices"]["label"]
            q, lab = _fixed_choice(ex["question_stem"], ex["choices"]["text"], labels.index(ex["answerKey"]), rng)
            yield split, "", q, lab, None


def arc(rng) -> Iterator[Row]:
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for hub, split in (("train", "train"), ("validation", "val")):
            for ex in _ds("allenai/ai2_arc", cfg, split=hub):
                labels = ex["choices"]["label"]
                if ex["answerKey"] not in labels:
                    continue
                q, lab = _fixed_choice(ex["question"], ex["choices"]["text"], labels.index(ex["answerKey"]), rng)
                yield split, json.dumps({"difficulty": cfg.split("-")[1].lower()}), q, lab, None


def boolq(rng) -> Iterator[Row]:
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in _ds("google/boolq", split=hub):
            q = _noul(ex["question"].rstrip("?") + "?")
            yield split, json.dumps({"passage": ex["passage"][:1500]}), q, int(ex["answer"]), None


def snli(rng) -> Iterator[Row]:
    opts = ["entailment: the hypothesis must be true", "neutral: the hypothesis may or may not be true",
            "contradiction: the hypothesis must be false"]
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in _ds("stanfordnlp/snli", split=hub):
            if ex["label"] not in (0, 1, 2):
                continue
            q, lab = _fixed_choice("Given the premise, is the hypothesis true?", opts, ex["label"], rng, shuffle=False)
            yield split, json.dumps({"premise": ex["premise"], "hypothesis": ex["hypothesis"]}), q, lab, None


def support_intent(rng) -> Iterator[Row]:
    rows = list(_ds("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"))
    space = sorted({_human(r["intent"]) for r in rows})
    for i, ex in enumerate(rows):
        split = "val" if i % 10 == 0 else "train"
        q, lab = _sampled_choice("Which support intent does this customer message express?", _human(ex["intent"]), space, rng, text=ex["instruction"])
        yield split, json.dumps({"customer_message": ex["instruction"]}), q, lab, None


# ---- code -----------------------------------------------------------------------------------------------------


def code_contests(rng) -> Iterator[Row]:
    """Will this (real, judged) solution pass all tests? One passing and one failing Python 3 solution per problem."""
    for hub, split in (("train", "train"), ("valid", "val"), ("test", "val")):
        for ex in _ds("deepmind/code_contests", split=hub, streaming=True):
            good = [s for l, s in zip(ex["solutions"]["language"], ex["solutions"]["solution"]) if l == 3]
            bad = [s for l, s in zip(ex["incorrect_solutions"]["language"], ex["incorrect_solutions"]["solution"]) if l == 3]
            for sols, lab in ((good, 1), (bad, 0)):
                if sols:
                    code = rng.choice(sols)[:2500]
                    state = json.dumps({"problem": ex["description"][:1800], "solution_python": code})
                    q = _noul(rng.choice(("Will this solution pass all of the problem's tests?",
                                          "Is this program a correct solution to the problem?")))
                    yield split, state, q, lab, None


def code_choice(rng) -> Iterator[Row]:
    """Which of three real, judged Python 3 solutions is correct (1 passing, 2 failing)? Only short solutions
    (<= 600 chars) are used, so no truncation can hide the bug that makes one fail."""
    for hub, split in (("train", "train"), ("valid", "val"), ("test", "val")):
        for ex in _ds("deepmind/code_contests", split=hub, streaming=True):
            good = [s for l, s in zip(ex["solutions"]["language"], ex["solutions"]["solution"]) if l == 3 and len(s) <= 600]
            bad = [s for l, s in zip(ex["incorrect_solutions"]["language"], ex["incorrect_solutions"]["solution"])
                   if l == 3 and len(s) <= 600]
            bad = list(dict.fromkeys(bad))
            if not good or len(bad) < 2:
                continue
            sols = [rng.choice(good)] + rng.sample(bad, 2)
            order = [0, 1, 2]
            rng.shuffle(order)
            state = {"problem": ex["description"][:900]}
            for letter, i in zip("ABC", order):
                state["solution_" + letter] = sols[i]
            q = {"type": "choice", "instructions": "Which solution correctly solves the problem?",
                 "criteria": ["solution A", "solution B", "solution C"]}
            yield split, json.dumps(state), q, order.index(0), None


def code_defect(rng) -> Iterator[Row]:
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in _ds("google/code_x_glue_cc_defect_detection", split=hub):
            q = _noul("Does this C function contain a security vulnerability?")
            yield split, json.dumps({"function_c": ex["func"][:3000]}), q, int(ex["target"]), None


SOURCES: Dict[str, dict] = {
    "banking77": {"train_cap": 12000, "fn": banking77, "repo": "PolyAI/banking77", "license": "CC BY 4.0 (GitHub PolyAI-LDN/task-specific-datasets)",
                  "use": "router / intent"},
    "clinc_intent": {"train_cap": 16000, "fn": clinc_intent, "repo": "clinc/clinc_oos", "license": "CC BY 3.0", "use": "router / intent"},
    "clinc_scope": {"fn": clinc_scope, "repo": "clinc/clinc_oos", "license": "CC BY 3.0",
                    "use": "out-of-scope detection (escalation)", "balance": True},
    "civil_toxicity": {"train_cap": 30000, "fn": civil_toxicity, "repo": "google/civil_comments", "license": "CC0-1.0",
                       "use": "moderation, human vote-share soft targets", "balance": True},
    "sms_spam": {"fn": sms_spam, "repo": "ucirvine/sms_spam", "license": "CC BY 4.0 (UCI)", "use": "spam triage",
                 "balance": True},
    "prompt_injection": {"fn": prompt_injection, "repo": "deepset/prompt-injections", "license": "Apache-2.0",
                         "use": "guardrail (text)"},
    "jailbreak": {"fn": jailbreak, "repo": "jackhhao/jailbreak-classification", "license": "Apache-2.0",
                  "use": "guardrail (text)"},
    "mmlu": {"fn": mmlu, "repo": "cais/mmlu", "license": "MIT", "use": "knowledge (eval only)", "eval_only": True},
    "commonsense_qa": {"train_cap": 12000, "fn": commonsense_qa, "repo": "tau/commonsense_qa", "license": "MIT", "use": "commonsense"},
    "openbookqa": {"fn": openbookqa, "repo": "allenai/openbookqa", "license": "Apache-2.0 (GitHub allenai/OpenBookQA)",
                   "use": "science knowledge"},
    "arc": {"fn": arc, "repo": "allenai/ai2_arc", "license": "CC BY-SA 4.0 (share-alike)", "use": "science reasoning"},
    "boolq": {"train_cap": 12000, "fn": boolq, "repo": "google/boolq", "license": "CC BY-SA 3.0 (share-alike)", "use": "reading comprehension"},
    "snli": {"train_cap": 30000, "fn": snli, "repo": "stanfordnlp/snli", "license": "CC BY-SA 4.0 (share-alike)", "use": "fact verification / NLI"},
    "support_intent": {"train_cap": 25000, "fn": support_intent, "repo": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                       "license": "CDLA-Sharing-1.0 (share-alike)", "use": "customer support routing"},
    "code_contests": {"train_cap": 20000, "fn": code_contests, "repo": "deepmind/code_contests", "license": "CC BY 4.0",
                      "use": "code: will this solution pass?", "balance": True},
    "code_choice": {"fn": code_choice, "repo": "deepmind/code_contests", "license": "CC BY 4.0",
                    "use": "code: which real solution is correct?"},
    "code_defect": {"train_cap": 22000, "fn": code_defect, "repo": "google/code_x_glue_cc_defect_detection",
                    "license": "C-UDA (Computational Use of Data Agreement)", "use": "code: vulnerability triage"},
}


def build(name: str, root: str = DATA_ROOT, seed: int = 0, train_cap: int = TRAIN_CAP, val_cap: int = VAL_CAP, log=print,
          spec: Optional[dict] = None):
    """``spec`` defaults to ``SOURCES[name]``; other modules (e.g. ``visionjev_codeexec``) pass their own."""
    spec = spec or SOURCES[name]
    train_cap, val_cap = spec.get("train_cap", train_cap), spec.get("val_cap", val_cap)  # per-source overrides
    rng = random.Random(seed)
    rows = {"train": [], "val": []}
    seen = {"train": 0, "val": 0}
    for item in spec["fn"](rng):
        # rows are (split, state, question, label, target) or, with extra record fields such as a rationale for the
        # rationale loss, (split, state, question, label, target, extra)
        split, state, q, label, target = item[:5]
        extra = item[5] if len(item) > 5 else None
        seen[split] += 1
        crit = q.get("criteria")
        if isinstance(crit, list) and (len(set(crit)) != len(crit) or any(not str(c).strip() for c in crit)):
            seen[split + "_dropped_duplicate_or_blank_options"] = seen.get(split + "_dropped_duplicate_or_blank_options", 0) + 1
            continue
        rows[split].append((state, q, label, target, extra))
        if len(rows["train"]) >= 6 * train_cap and len(rows["val"]) >= 6 * val_cap:
            break  # enough to sample from (streaming sources are large)
    out = {}
    for split, cap in (("train", 0 if spec.get("eval_only") else train_cap), ("val", val_cap)):
        rs = rows[split]
        rng.shuffle(rs)
        if spec.get("balance"):  # noul: take as many of each class as the rarer one allows
            pos = [r for r in rs if r[2] == 1]
            neg = [r for r in rs if r[2] == 0]
            n = min(len(pos), len(neg), cap // 2)
            rs = pos[:n] + neg[:n]
            rng.shuffle(rs)
        out[split] = rs[:cap]
    tmp, final = os.path.join(root, ".tmp-" + name), os.path.join(root, name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    labels = {}
    for split, rs in out.items():
        if not rs:
            continue
        with open(os.path.join(tmp, split + ".jsonl"), "w") as f:
            for i, (state, q, label, target, extra) in enumerate(rs):
                rec = {"id": "%s-%s-%d" % (name, split, i), "state_text": state or None, "question": q, "label": label}
                if target is not None:
                    rec["target"] = target
                if extra:
                    rec.update(extra)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        labels[split] = dict(Counter(r[2] for r in rs))
    manifest = {"dataset": name, "source": spec["repo"], "revision": _hub_revision(spec["repo"]),
                "license": {"annotations": spec["license"], "images": "none (text-only)", "commercial": True,
                            "verified": "2026-09-25 HF card / upstream; see CONTEXT.md section 9"},
                "use": spec["use"], "eval_only": bool(spec.get("eval_only")), "seed": seed,
                "counts": {k: len(v) for k, v in out.items()}, "rows_seen": seen, "label_counts": labels,
                "validator": {"noul_min_minority": 0.05}}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if os.path.exists(final):
        old = os.path.join(root, ".old-%s-%d" % (name, int(time.time())))
        os.rename(final, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    log("%s: %s (labels %s)" % (name, manifest["counts"], labels))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    names = list(SOURCES) if a.names == ["all"] else a.names
    for n in names:
        try:
            build(n, a.root, log=lambda m: print(m, flush=True))
        except Exception as e:  # one broken source must not stop the rest
            print("%s: FAILED %r" % (n, e), flush=True)
