"""Hard reasoning decisions for the short hard-reasoning delta (Hypothesis/V6.1.md §4.3, §4.6; PROBLEMS D2, D14, E8).

Licensed reasoning sets (licences read on the Hub 2026-09-29 or stated by the project):
* ``bb_reasoning``    (choice) BIG-bench (tasksource/bigbench, Apache-2.0): date understanding, temporal sequences,
                      logical deduction, tracking shuffled objects, penguins in a table, colored objects, navigate,
                      causal judgment, disambiguation, formal fallacies, logical fallacy detection, hyperbaton,
                      elementary math, analytic entailment. At most 8 options (gold + sampled distractors).
* ``proofwriter_owa`` (choice, 3) ProofWriter open-world (tasksource/proofwriter; AI2, CC BY 4.0): facts + rules ->
                      is the statement true, false or unknown? Depth >= 1 only; depth 3-5 kept in full, the rest sampled.
* ``quality_long``    (choice, 4) QuALITY (emozilla/quality; NYU, CC BY 4.0): questions over 2-8k-token articles; the
                      long-document case (read with the 4,096-token budget, cut in the middle).
* ``wanli_nli``       (choice, 3) WANLI (CC BY 4.0): entailment / neutral / contradiction on adversarial pairs.
* ``helpsteer_score`` (score, 5) HelpSteer2 (CC BY 4.0): human helpfulness rating 0-4 of a response (judge-style).
* ``helpsteer_pair``  (choice, 2) HelpSteer2: which of two responses to the same prompt is more helpful (ratings
                      differ by >= 2, so the label is clear).

    python -m laya.visionjev_hard all
"""
import argparse
import ast
import hashlib
import json
from collections import defaultdict
from typing import Dict, Iterator

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT

BB_TASKS = ["date_understanding", "temporal_sequences", "logical_deduction", "tracking_shuffled_objects",
            "penguins_in_a_table", "reasoning_about_colored_objects", "navigate", "causal_judgment",
            "disambiguation_qa", "formal_fallacies_syllogisms_negation", "logical_fallacy_detection", "hyperbaton",
            "elementary_math_qa", "analytic_entailment", "logical_args", "logic_grid_puzzle"]


def _split(key: str, pct: int = 8) -> str:
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < pct else "train"


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


BB_PER_TASK = {"train": 2500, "val": 120}  # large tasks (hyperbaton, elementary math) must not crowd out the small hard ones


def bb_reasoning(rng) -> Iterator[ts.Row]:
    for task in BB_TASKS:
        for hub, split in (("train", "train"), ("validation", "val")):
            n = 0
            try:
                rows = ts._ds("tasksource/bigbench", task, split=hub)
            except Exception as e:  # a task missing from the mirror must not stop the others
                print("bigbench %s/%s: %r" % (task, hub, e), flush=True)
                continue
            for ex in rows:
                opts, scores = _lit(ex["multiple_choice_targets"]), _lit(ex["multiple_choice_scores"])
                if not opts or len(opts) != len(scores) or sum(scores) != 1:
                    continue
                gold = scores.index(1)
                others = [i for i in range(len(opts)) if i != gold]
                keep = [gold] + rng.sample(others, min(7, len(others)))
                q, lab = ts._fixed_choice("Which option correctly answers the problem?", [str(opts[i]) for i in keep],
                                          0, rng)
                state = {"task": task.replace("_", " "), "problem": ex["inputs"].strip()}
                yield split, json.dumps(state, ensure_ascii=False), q, lab, None
                n += 1
                if n >= BB_PER_TASK[split]:
                    break


def proofwriter_owa(rng) -> Iterator[ts.Row]:
    lab_of = {"True": 0, "False": 1, "Unknown": 2}
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in ts._ds("tasksource/proofwriter", split=hub):
            if "OWA" not in ex["id"] or ex["answer"] not in lab_of:
                continue
            d = int(ex["QDep"] or 0)
            if d < 1 or (d < 3 and rng.random() > 0.35):
                continue
            q, lab = ts._fixed_choice("Using only these facts and rules, is the following statement true, false, or "
                                      "unknown (cannot be determined)? Statement: %s" % ex["question"],
                                      ["true", "false", "unknown"], lab_of[ex["answer"]], rng)
            yield split, json.dumps({"facts_and_rules": ex["theory"]}), q, lab, None


def quality_long(rng) -> Iterator[ts.Row]:
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in ts._ds("emozilla/quality", split=hub):
            opts = _lit(ex["options"])
            a = int(ex["answer"])
            if not opts or not 0 <= a < len(opts):
                continue
            q, lab = ts._fixed_choice(ex["question"].strip(), [str(o) for o in opts], a, rng)
            yield split, json.dumps({"article": ex["article"]}, ensure_ascii=False), q, lab, None


def wanli_nli(rng) -> Iterator[ts.Row]:
    names = ["entailment", "neutral", "contradiction"]
    for hub, split in (("train", "train"), ("test", "val")):
        for ex in ts._ds("alisawuffles/WANLI", split=hub):
            if ex["gold"] not in names:
                continue
            q, lab = ts._fixed_choice("Does the premise entail the hypothesis, contradict it, or neither (neutral)?",
                                      names, names.index(ex["gold"]), rng)
            yield split, json.dumps({"premise": ex["premise"], "hypothesis": ex["hypothesis"]}), q, lab, None


LEVELS = ["not helpful at all", "slightly helpful", "partly helpful", "helpful", "very helpful"]


def _helpsteer():
    out = []
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in ts._ds("nvidia/HelpSteer2", split=hub):
            out.append((split, ex))
    return out


def helpsteer_score(rng) -> Iterator[ts.Row]:
    for split, ex in _helpsteer():
        h = int(ex["helpfulness"])
        if not 0 <= h <= 4 or len(ex["response"]) > 6000:
            continue
        q = {"type": "score", "instructions": "How helpful is the response to the request? Judge correctness, "
             "completeness and whether it follows the request, not length or polish.", "criteria": LEVELS}
        yield split, json.dumps({"request": ex["prompt"][:3000], "response": ex["response"]}, ensure_ascii=False), q, h, None


def helpsteer_pair(rng) -> Iterator[ts.Row]:
    by_prompt = defaultdict(list)
    for split, ex in _helpsteer():
        by_prompt[(split, ex["prompt"])].append(ex)
    for (split, prompt), exs in by_prompt.items():
        if len(exs) < 2:
            continue
        a, b = exs[0], exs[1]
        ha, hb = int(a["helpfulness"]), int(b["helpfulness"])
        if abs(ha - hb) < 2 or len(a["response"]) + len(b["response"]) > 8000:
            continue
        if rng.random() < 0.5:
            a, b, ha, hb = b, a, hb, ha
        q = {"type": "choice", "instructions": "Which response better satisfies the request? Judge correctness and "
             "whether it does what was asked, not length or tone.", "criteria": ["response A", "response B"]}
        state = {"request": prompt[:3000], "response_A": a["response"], "response_B": b["response"]}
        yield split, json.dumps(state, ensure_ascii=False), q, int(hb > ha), None


LIC = "see module docstring"
SOURCES: Dict[str, dict] = {
    "bb_reasoning": {"fn": bb_reasoning, "repo": "tasksource/bigbench", "license": "Apache-2.0",
                     "use": "multi-step logic, dates, tracking, tables", "train_cap": 30000, "val_cap": 1500},
    "proofwriter_owa": {"fn": proofwriter_owa, "repo": "tasksource/proofwriter", "license": "CC BY 4.0 (AI2 ProofWriter)",
                        "use": "deduction with true / false / unknown", "train_cap": 20000, "val_cap": 1000},
    "quality_long": {"fn": quality_long, "repo": "emozilla/quality", "license": "CC BY 4.0 (nyu-mll/quality)",
                     "use": "long-document reading", "train_cap": 10000, "val_cap": 1000},
    "wanli_nli": {"fn": wanli_nli, "repo": "alisawuffles/WANLI", "license": "CC BY 4.0",
                  "use": "adversarial NLI", "train_cap": 20000, "val_cap": 1000},
    "helpsteer_score": {"fn": helpsteer_score, "repo": "nvidia/HelpSteer2", "license": "CC BY 4.0",
                        "use": "judge: rate a response", "train_cap": 15000, "val_cap": 800},
    "helpsteer_pair": {"fn": helpsteer_pair, "repo": "nvidia/HelpSteer2", "license": "CC BY 4.0",
                       "use": "judge: pick the better response", "train_cap": 8000, "val_cap": 500},
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (list(SOURCES) if a.names == ["all"] else a.names):
        try:
            ts.build(n, a.root, spec=SOURCES[n], log=lambda m: print(m, flush=True))
        except Exception as e:
            print("%s: FAILED %r" % (n, e), flush=True)
