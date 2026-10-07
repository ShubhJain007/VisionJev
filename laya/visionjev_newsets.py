"""New decision families for robustness across setups (licenses checked on the Hub cards 2026-09-27):

Agents / tools
* ``tool_select``   (choice) available tools + a user request -> which tool should be called first (Hermes function
                    calling, Apache-2.0; ToolACE, Apache-2.0). Options are the tools actually offered.
* ``tool_relevant`` (noul) can any of these tools serve the request? Positives are the request with its own tools;
                    negatives pair it with the tool set of a request from a *different* category.
Safety
* ``aegis_safety``  (noul / choice) is this user message unsafe? (human labels); which category does it violate?
                    (Aegis 2.0, CC BY 4.0; single-category rows only).
* ``hate_score``    (score, 3 levels, soft) not hateful / unclear / hateful, the target being the annotators' vote
                    share per comment (Measuring Hate Speech, CC BY 4.0; comments with >= 2 annotators).
* ``salad_category`` (choice) harmful request -> harm category (Salad-Data base set, Apache-2.0; distractors from the
                    same top-level domain first).
Documents / privacy
* ``pii_detect``    (noul / choice) does this document contain a <PII type>? (from the annotated spans) / what kind
                    of document is it? (Gretel synthetic PII finance, multilingual, Apache-2.0).
Facts / reading / math
* ``vitaminc``      (choice, 3) does the evidence support / refute the claim, or is there not enough information?
                    (VitaminC, CC BY-SA 3.0; contrastive pairs).
* ``squad_answerable`` (noul) can the question be answered from the passage? (SQuAD 2.0, CC BY-SA 4.0).
* ``aqua_math``     (choice, 5 native options) algebra word problems (AQuA-RAT, Apache-2.0).
Sentiment / emotion
* ``go_emotions``   (choice) which emotion does this message express? (GoEmotions, Apache-2.0; single-label rows).
* ``review_polarity`` (choice, 2) product review sentiment (Amazon Polarity, Apache-2.0).

    python -m laya.visionjev_newsets all
"""
import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterator

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT
from .visionjev_diverse import _grouped_choice, _split

# ---- agents / tools ---------------------------------------------------------------------------------------------


def _hermes_rows():
    out = []
    for cfg in ("func_calling_singleturn", "func_calling"):
        for ex in ts._ds("NousResearch/hermes-function-calling-v1", cfg):
            try:
                tools = json.loads(ex["tools"]) if isinstance(ex["tools"], str) else ex["tools"]
            except ValueError:
                continue
            conv = ex["conversations"] if isinstance(ex["conversations"], list) else ast.literal_eval(ex["conversations"])
            human = next((c["value"] for c in conv if c["from"] == "human"), None)
            gpt = next((c["value"] for c in conv if c["from"] == "gpt"), "")
            m = re.search(r'"name":\s*"([^"]+)"', gpt) or re.search(r"'name':\s*'([^']+)'", gpt)
            if not human or not m:
                continue
            fns = [(t.get("function") or t) for t in tools]
            fns = [{"name": f["name"], "description": f.get("description", "")[:200]} for f in fns if f.get("name")]
            out.append({"request": human[:1500], "tools": fns, "gold": m.group(1), "category": ex.get("category") or "",
                        "key": ex["id"]})
    return out


def _toolace_rows():
    out = []
    for i, ex in enumerate(ts._ds("Team-ACE/ToolACE")):
        conv = ex["conversations"] if isinstance(ex["conversations"], list) else ast.literal_eval(ex["conversations"])
        if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            continue
        m = re.match(r"\s*\[\s*([^\(\[\]]+?)\s*\(", conv[1]["value"])  # names may contain spaces
        j = ex["system"].find("[")
        if not m or j < 0:
            continue
        try:  # the function list is the first JSON array; the format hint after it also contains brackets
            fns, _ = json.JSONDecoder().raw_decode(ex["system"][j:])
        except ValueError:
            continue
        fns = [{"name": f["name"], "description": str(f.get("description", ""))[:200]} for f in fns
               if isinstance(f, dict) and f.get("name")]
        out.append({"request": conv[0]["value"][:1500], "tools": fns, "gold": m.group(1),
                    "category": fns[0]["name"].split("_")[0] if fns else "", "key": "toolace-%d" % i})
    return out


def _tool_rows():
    rows = [r for r in _hermes_rows() + _toolace_rows()
            if len(r["tools"]) >= 2 and r["gold"] in {t["name"] for t in r["tools"]}
            and len({t["name"] for t in r["tools"]}) == len(r["tools"])]
    return rows


def tool_select(rng) -> Iterator[ts.Row]:
    for r in _tool_rows():
        tools = list(r["tools"])
        rng.shuffle(tools)
        opts = [t["name"] for t in tools][:8]
        if r["gold"] not in opts:
            opts = opts[:7] + [r["gold"]]
            rng.shuffle(opts)
        state = {"available_tools": [t for t in tools if t["name"] in opts], "user_request": r["request"]}
        q = {"type": "choice", "instructions": "Which tool should be called first to handle the request?", "criteria": opts}
        yield _split(r["key"]), json.dumps(state, ensure_ascii=False), q, opts.index(r["gold"]), None


def tool_relevant(rng) -> Iterator[ts.Row]:
    rows = _tool_rows()
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for r in rows:
        own = rng.random() < 0.5
        if own:
            tools = r["tools"]
        else:
            others = [c for c in by_cat if c != r["category"] and c]
            if not others:
                continue
            tools = rng.choice(by_cat[rng.choice(others)])["tools"]
        state = {"available_tools": tools[:8], "user_request": r["request"]}
        q = ts._noul(rng.choice(("Can any of the available tools serve this request?",
                                 "Is one of these tools the right one to call for this request?")))
        yield _split(r["key"]), json.dumps(state, ensure_ascii=False), q, int(own), None


# ---- safety -------------------------------------------------------------------------------------------------------


def aegis_safety(rng) -> Iterator[ts.Row]:
    rows = list(ts._ds("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")) + \
        [dict(r, _split="val") for r in ts._ds("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="validation")]
    cats = sorted({r["violated_categories"] for r in rows if r["violated_categories"] and "," not in r["violated_categories"]})
    for r in rows:
        if not r["prompt"] or r["prompt"] == "REDACTED":
            continue
        split = r.get("_split") or "train"
        state = json.dumps({"user_message": r["prompt"][:2000]}, ensure_ascii=False)
        c = r["violated_categories"]
        if r["prompt_label"] == "unsafe" and c and "," not in c and rng.random() < 0.5:
            q, lab = _grouped_choice("Which safety category does this message fall under?", c, [], cats, rng,
                                     k=rng.randint(4, 6))
            yield split, state, q, lab, None
        else:
            yield split, state, ts._noul("Is this user message unsafe?"), int(r["prompt_label"] == "unsafe"), None


def hate_score(rng) -> Iterator[ts.Row]:
    votes, text = defaultdict(Counter), {}
    for r in ts._ds("ucberkeley-dlab/measuring-hate-speech"):
        v = int(float(r["hatespeech"]))
        if v in (0, 1, 2):
            votes[r["comment_id"]][v] += 1
            text[r["comment_id"]] = r["text"]
    levels = ["not hateful", "unclear", "hateful"]
    for cid, c in votes.items():
        n = sum(c.values())
        if n < 2:
            continue
        target = [c[i] / n for i in range(3)]
        q = {"type": "score", "instructions": "How hateful is this comment?", "criteria": levels}
        yield _split(str(cid)), json.dumps({"comment": text[cid][:2000]}, ensure_ascii=False), q, \
            max(range(3), key=lambda i: target[i]), target


def salad_category(rng) -> Iterator[ts.Row]:
    rows = list(ts._ds("OpenSafetyLab/Salad-Data", "base_set"))
    strip = lambda s: re.sub(r"^O\d+:\s*", "", s or "").strip()  # noqa: E731
    by_top = defaultdict(set)
    for r in rows:
        by_top[strip(r["1-category"])].add(strip(r["2-category"]))
    allc = sorted({strip(r["2-category"]) for r in rows})
    for r in rows:
        gold = strip(r["2-category"])
        q, lab = _grouped_choice("Which kind of harmful request is this?", gold, sorted(by_top[strip(r["1-category"])]),
                                 allc, rng, k=rng.randint(4, 6))
        yield _split("%s-%s" % (r["qid"], r["question"][:40])), json.dumps({"request": r["question"]}, ensure_ascii=False), q, lab, None


# ---- documents / privacy ------------------------------------------------------------------------------------------


def pii_detect(rng) -> Iterator[ts.Row]:
    """PII questions are balanced *per type*: for each type, as many documents without it as with it (types like
    dates are in almost every document, so drawing the type per document let the question alone give the answer:
    audit question-only 0.86). Document-type questions use the remaining documents."""
    rows = list(ts._ds("gretelai/synthetic_pii_finance_multilingual")) + \
        [dict(r, _split="val") for r in ts._ds("gretelai/synthetic_pii_finance_multilingual", split="test")]
    types = Counter()
    for r in rows:
        r["_types"] = {s["label"] for s in json.loads(r["pii_spans"] or "[]")}
        types.update(r["_types"])
    common = [t for t, n in types.most_common() if n >= 200]
    doc_types = sorted({r["document_type"] for r in rows})
    rng.shuffle(rows)
    used = set()
    for split in ("train", "val"):
        pool = [i for i, r in enumerate(rows) if (r.get("_split") or "train") == split]
        per = max(1, (len(pool) * 6 // 10) // (2 * len(common)))  # ~60% of documents go to PII questions
        for t in common:
            with_t = [i for i in pool if i not in used and t in rows[i]["_types"]]
            without = [i for i in pool if i not in used and t not in rows[i]["_types"]]
            n = min(per, len(with_t), len(without))
            for i, lab in [(i, 1) for i in with_t[:n]] + [(i, 0) for i in without[:n]]:
                used.add(i)
                q = ts._noul("Does this document contain %s?" % _pii_phrase(t))
                yield split, json.dumps({"document": rows[i]["generated_text"][:2500]}, ensure_ascii=False), q, lab, None
        for i in pool:
            if i in used:
                continue
            r = rows[i]
            text = r["generated_text"][:2500]
            q, lab = _grouped_choice("What kind of document is this?", r["document_type"], [], doc_types, rng,
                                     k=rng.randint(4, 6), text=text)
            yield split, json.dumps({"document": text}, ensure_ascii=False), q, lab, None


def _pii_phrase(t: str) -> str:
    s = t.replace("_", " ")
    return ("an " if s[0] in "aeiou" else "a ") + s


# ---- facts / reading / math ------------------------------------------------------------------------------------------


def vitaminc(rng) -> Iterator[ts.Row]:
    names = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    opts = ["the evidence supports the claim", "the evidence refutes the claim", "not enough information"]
    for hub, split in (("train", "train"), ("validation", "val"), ("test", "val")):
        for ex in ts._ds("tals/vitaminc", split=hub, streaming=True):
            q, lab = ts._fixed_choice("Does the evidence support or refute the claim?", opts, names[ex["label"]], rng)
            yield split, json.dumps({"claim": ex["claim"], "evidence": ex["evidence"][:1500]}, ensure_ascii=False), q, lab, None


def squad_answerable(rng) -> Iterator[ts.Row]:
    for hub, split in (("train", "train"), ("validation", "val")):
        for ex in ts._ds("rajpurkar/squad_v2", split=hub):
            ans = ex["answers"]["text"] if isinstance(ex["answers"], dict) else []
            q = ts._noul(rng.choice(("Can this question be answered from the passage?",
                                     "Does the passage contain the answer to the question?")))
            yield split, json.dumps({"passage": ex["context"][:2000], "question": ex["question"]}, ensure_ascii=False), \
                q, int(bool(ans)), None


def aqua_math(rng) -> Iterator[ts.Row]:
    for hub, split in (("train", "train"), ("validation", "val"), ("test", "val")):
        for ex in ts._ds("deepmind/aqua_rat", "raw", split=hub):
            opts = ex["options"] if isinstance(ex["options"], list) else ast.literal_eval(ex["options"])
            clean = [re.sub(r"^[A-E]\)\s*", "", o).strip() for o in opts]
            letters = [o[0] for o in opts]
            if ex["correct"] not in letters or len(set(clean)) != len(clean):
                continue
            q, lab = ts._fixed_choice(ex["question"][:1500], clean, letters.index(ex["correct"]), rng)
            yield split, None, q, lab, None


GO_EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
               "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
               "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness",
               "surprise", "neutral"]


def go_emotions(rng) -> Iterator[ts.Row]:
    for hub, split in (("train", "train"), ("validation", "val"), ("test", "val")):
        for ex in ts._ds("google-research-datasets/go_emotions", "simplified", split=hub):
            if len(ex["labels"]) != 1:
                continue
            q, lab = _grouped_choice("Which emotion does this message express?", GO_EMOTIONS[ex["labels"][0]], [],
                                     GO_EMOTIONS, rng, k=rng.randint(4, 6))
            yield split, json.dumps({"message": ex["text"]}, ensure_ascii=False), q, lab, None


def review_polarity(rng) -> Iterator[ts.Row]:
    for hub, split in (("train", "train"), ("test", "val")):
        for i, ex in enumerate(ts._ds("fancyzhx/amazon_polarity", split=hub, streaming=True)):
            if i >= 60000:
                break
            q, lab = ts._fixed_choice("Is this review positive or negative?", ["negative", "positive"], int(ex["label"]), rng)
            yield split, json.dumps({"title": ex["title"], "review": ex["content"][:1500]}, ensure_ascii=False), q, lab, None


SOURCES: Dict[str, dict] = {
    "tool_select": {"fn": tool_select, "repo": "NousResearch/hermes-function-calling-v1",
                    "license": "Apache-2.0 (Hermes function calling; ToolACE)", "use": "agent: which tool to call",
                    "train_cap": 15000},
    "tool_relevant": {"fn": tool_relevant, "repo": "NousResearch/hermes-function-calling-v1",
                      "license": "Apache-2.0 (Hermes function calling; ToolACE)", "use": "agent: are these tools relevant",
                      "balance": True, "train_cap": 12000},
    "aegis_safety": {"fn": aegis_safety, "repo": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "license": "CC BY 4.0",
                     "use": "content safety (human labels)", "train_cap": 20000},
    "hate_score": {"fn": hate_score, "repo": "ucberkeley-dlab/measuring-hate-speech", "license": "CC BY 4.0",
                   "use": "hate speech score, annotator vote shares as soft targets", "train_cap": 20000},
    "salad_category": {"fn": salad_category, "repo": "OpenSafetyLab/Salad-Data", "license": "Apache-2.0",
                       "use": "harmful request taxonomy", "train_cap": 15000},
    "pii_detect": {"fn": pii_detect, "repo": "gretelai/synthetic_pii_finance_multilingual", "license": "Apache-2.0",
                   "use": "PII presence / document type, multilingual finance documents", "train_cap": 20000},
    "vitaminc": {"fn": vitaminc, "repo": "tals/vitaminc", "license": "CC BY-SA 3.0 (share-alike)",
                 "use": "fact verification against evidence", "train_cap": 25000},
    "squad_answerable": {"fn": squad_answerable, "repo": "rajpurkar/squad_v2", "license": "CC BY-SA 4.0 (share-alike)",
                         "use": "answerability / abstention", "balance": True, "train_cap": 20000},
    "aqua_math": {"fn": aqua_math, "repo": "deepmind/aqua_rat", "license": "Apache-2.0", "use": "math word problems",
                  "train_cap": 20000},
    "go_emotions": {"fn": go_emotions, "repo": "google-research-datasets/go_emotions", "license": "Apache-2.0",
                    "use": "emotion / tone", "train_cap": 15000},
    "review_polarity": {"fn": review_polarity, "repo": "fancyzhx/amazon_polarity", "license": "Apache-2.0",
                        "use": "review sentiment", "train_cap": 15000},
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (list(SOURCES) if a.names == ["all"] else a.names):
        try:
            ts.build(n, a.root, spec=SOURCES[n], log=lambda m: print(m, flush=True))
        except Exception as e:  # one broken source must not stop the rest
            print("%s: FAILED %r" % (n, e), flush=True)
