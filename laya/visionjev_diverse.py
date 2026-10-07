"""More diverse decision sets: customer support (multilingual intents, industry verticals), languages, code review.
Licenses checked on the Hub cards / upstream on 2026-09-27 (CONTEXT.md section 9). Built by the generic
``visionjev_textsets.build``; distractors are always real labels / texts of the same source, drawn from the *same
scenario / vertical / language* first, so keyword matching alone cannot pick the gold.

Customer support
* ``massive_intent``      (choice) MASSIVE 1.1 (CC BY 4.0): an assistant utterance in one of 20 languages -> intent;
                          distractors from the same scenario first (alarm_set vs alarm_query vs alarm_remove).
* ``support_verticals``   (choice) Bitext retail banking / insurance / telco / travel / e-commerce (CDLA-Sharing-1.0):
                          customer message -> intent, distractors from the same vertical and category first.
* ``minds14_intent``      (choice) MINDS-14 (CC BY 4.0): e-banking call transcripts, 14 languages -> intent.
* ``sgd_intent``          (choice) Schema-Guided Dialogue (CC BY-SA 4.0), 20 domains: the turn where a new intent
                          starts -> the schema's description of it (same service's other intents first).
* ``multiwoz_intent``     (choice) MultiWOZ 2.2 (MIT): same, restaurant / hotel / train / taxi / attraction.
Languages
* ``belebele``            (choice, 4) Belebele (CC BY-SA 4.0): passage + question in 24 languages; the split is by
                          passage, so a passage's translations never cross train / val.
* ``xcopa``               (choice, 2) XCOPA (CC BY 4.0): cause / effect in 11 languages; split by item.
* ``xwinograd``           (choice, 2) XWinograd (CC BY 4.0): which entity the blank refers to, 6 languages.
* ``tweet_sentiment_ml``  (choice, 3) multilingual tweet sentiment (CC BY 3.0), 8 languages.
Code
* ``commit_message``      (choice, 4) CommitPackFT (MIT; permissively licensed repos): a diff -> which commit subject
                          describes it; distractors are subjects of other commits in the same language, the three
                          sharing the most words with the diff.
* ``conala_snippet``      (choice, 4) CoNaLa curated (MIT): an intent -> which Python snippet does it.
Robotics (eval only)
* ``robospatial_home``    (noul) RoboSpatial-Home (Apache-2.0): spatial compatibility / configuration yes-no questions
                          on real home scenes (images written to ``robospatial_home/images``).

    python -m laya.visionjev_diverse massive_intent support_verticals belebele xcopa xwinograd ...
"""
import argparse
import difflib
import hashlib
import io
import json
import os
import random
import re
import subprocess
import tarfile
import urllib.request
from collections import defaultdict
from typing import Dict, Iterator, List

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT, RAW_ROOT

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_LOCALES = ["en-US", "de-DE", "fr-FR", "es-ES", "it-IT", "pt-PT", "nl-NL", "pl-PL", "ru-RU", "tr-TR", "ar-SA",
                   "hi-IN", "bn-BD", "ta-IN", "zh-CN", "ja-JP", "ko-KR", "vi-VN", "id-ID", "sw-KE"]
BITEXT = {"retail banking": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
          "insurance": "bitext/Bitext-insurance-llm-chatbot-training-dataset",
          "telco": "bitext/Bitext-telco-llm-chatbot-training-dataset",
          "travel": "bitext/Bitext-travel-llm-chatbot-training-dataset",
          "e-commerce": "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"}
BELEBELE_LANGS = ["eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "nld_Latn", "pol_Latn",
                  "rus_Cyrl", "ukr_Cyrl", "tur_Latn", "arb_Arab", "heb_Hebr", "hin_Deva", "ben_Beng", "tam_Taml",
                  "zho_Hans", "jpn_Jpan", "kor_Hang", "vie_Latn", "ind_Latn", "swh_Latn", "yor_Latn", "amh_Ethi"]
XCOPA_LANGS = ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]
XWINO_LANGS = ["en", "fr", "jp", "pt", "ru", "zh"]
TWEET_LANGS = ["arabic", "english", "french", "german", "hindi", "italian", "portuguese", "spanish"]
COMMIT_LANGS = ["python", "javascript", "typescript", "java", "go", "c", "c++", "rust", "ruby", "php", "shell",
                "kotlin", "swift", "sql", "yaml"]
VAL_PCT = 10


def _split(key: str, pct: int = VAL_PCT) -> str:
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < pct else "train"


def _grouped_choice(ins, gold, same, others, rng, k=5, text=""):
    """Gold + up to k-1 distractors: from ``same`` (the gold's group) first, then ``others``. With ``text``, the
    distractors are the labels sharing the most words with it (ties broken at random), so keyword matching finds
    them as readily as the gold (audit: Bitext messages name their intent)."""
    same = [x for x in dict.fromkeys(same) if x != gold]
    others = [x for x in dict.fromkeys(others) if x != gold and x not in same]
    if text:
        tw = _words(text)
        key = lambda s: (-len(_words(s) & tw), rng.random())  # noqa: E731
        same, others = sorted(same, key=key), sorted(others, key=key)
        n_same = min(len(same), (k - 1) // 2 + 1)
        opts = [gold] + same[:n_same] + others[:k - 1 - n_same]
        rng.shuffle(opts)
        return {"type": "choice", "instructions": ins, "criteria": opts}, opts.index(gold)
    n_same = min(len(same), rng.randint(1, k - 1))
    opts = [gold] + rng.sample(same, n_same) + rng.sample(others, min(len(others), k - 1 - n_same))
    rng.shuffle(opts)
    return {"type": "choice", "instructions": ins, "criteria": opts}, opts.index(gold)


# ---- customer support ---------------------------------------------------------------------------------------------


def _massive_rows():
    raw = os.path.join(RAW_ROOT, "massive", "amazon-massive-dataset-1.1.tar.gz")
    if not os.path.exists(raw):
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        urllib.request.urlretrieve(MASSIVE_URL, raw)
    with tarfile.open(raw) as tar:
        for m in tar.getmembers():
            loc = os.path.basename(m.name).replace(".jsonl", "")
            if m.isfile() and loc in MASSIVE_LOCALES:
                for line in tar.extractfile(m):
                    yield json.loads(line)


def massive_intent(rng) -> Iterator[ts.Row]:
    rows = list(_massive_rows())
    by_scen = defaultdict(set)
    for r in rows:
        by_scen[r["scenario"]].add(r["intent"])
    allint = sorted({r["intent"] for r in rows})
    rng.shuffle(rows)
    for r in rows:
        gold = ts._human(r["intent"])
        q, lab = _grouped_choice("Which request is the user making to the assistant?", gold,
                                 [ts._human(i) for i in sorted(by_scen[r["scenario"]])], [ts._human(i) for i in allint],
                                 rng, k=rng.randint(4, 6))
        yield ("train" if r["partition"] == "train" else "val"), json.dumps({"utterance": r["utt"]},
                                                                             ensure_ascii=False), q, lab, None


def support_verticals(rng) -> Iterator[ts.Row]:
    for vertical, repo in BITEXT.items():
        exs = list(ts._ds(repo))
        by_cat = defaultdict(set)
        for ex in exs:
            by_cat[ex["category"]].add(ex["intent"])
        allint = sorted({ex["intent"] for ex in exs})
        rng.shuffle(exs)
        for ex in exs[:9000]:
            q, lab = _grouped_choice("Which %s support intent does this customer message express?" % vertical,
                                     ts._human(ex["intent"]), [ts._human(i) for i in sorted(by_cat[ex["category"]])],
                                     [ts._human(i) for i in allint], rng, k=rng.randint(4, 6), text=ex["instruction"])
            yield _split(ex["instruction"]), json.dumps({"customer_message": ex["instruction"]}, ensure_ascii=False), q, lab, None


MINDS14_INTENTS = ["abroad", "address", "app error", "atm limit", "balance", "business loan", "card issues",
                   "cash deposit", "direct debit", "freeze", "high value payment", "joint account",
                   "latest transactions", "pay bill"]


def minds14_intent(rng) -> Iterator[ts.Row]:
    """MINDS-14 (CC BY 4.0): e-banking call transcripts in 14 languages -> intent (the audio is not used)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for i in range(2):
        t = pq.read_table(hf_hub_download("PolyAI/minds14", "all/train-%05d-of-00002.parquet" % i, repo_type="dataset"),
                          columns=["path", "transcription", "intent_class"])
        for path, text, c in zip(*(t.column(k).to_pylist() for k in ("path", "transcription", "intent_class"))):
            if not text or not text.strip():
                continue
            q, lab = _grouped_choice("What does the caller want help with?", MINDS14_INTENTS[c], [], MINDS14_INTENTS,
                                     rng, k=rng.randint(4, 6), text=text)
            yield _split(path or text), json.dumps({"call_transcript": text}, ensure_ascii=False), q, lab, None


SGD_RAW = "https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/%s/%s"
WOZ_RAW = "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/%s/dialogues_%03d.json"


def _fetch_json(url, cache):
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        urllib.request.urlretrieve(url, cache + ".tmp")
        os.rename(cache + ".tmp", cache)
    return json.load(open(cache))


def _dialogue_rows(dialogues, describe, rng, split, same_group):
    """The user turn where each new active intent first appears -> which intent (last 3 turns as context)."""
    for d in dialogues:
        seen, hist = set(), []
        for turn in d["turns"]:
            hist.append("%s: %s" % (turn["speaker"].upper(), turn["utterance"]))
            if turn["speaker"] != "USER":
                continue
            for fr in turn["frames"]:
                intent = fr.get("state", {}).get("active_intent", "NONE")
                if intent in ("NONE", None) or (fr["service"], intent) in seen:
                    continue
                seen.add((fr["service"], intent))
                gold = describe(fr["service"], intent)
                if gold is None:
                    continue
                same, others = same_group(fr["service"], intent)
                q, lab = _grouped_choice("What does the user want to do now?", gold, same, others, rng,
                                         k=rng.randint(4, 6), text="\n".join(hist[-3:]))
                yield split, json.dumps({"conversation": "\n".join(hist[-3:])}, ensure_ascii=False), q, lab, None


def sgd_intent(rng) -> Iterator[ts.Row]:
    """Schema-Guided Dialogue (CC BY-SA 4.0), 20 domains: options are the schema's intent descriptions."""
    base = os.path.join(RAW_ROOT, "sgd")
    for split, hub, n in (("train", "train", 127), ("val", "dev", 20)):
        schema = _fetch_json(SGD_RAW % (hub, "schema.json"), os.path.join(base, hub, "schema.json"))
        desc = {(s["service_name"], it["name"]): it["description"] for s in schema for it in s["intents"]}
        by_service = defaultdict(list)
        for (svc, _), d in desc.items():
            by_service[svc].append(d)
        alld = sorted(set(desc.values()))
        same_group = lambda svc, it: (by_service[svc], alld)  # noqa: E731
        for i in range(1, n + 1):
            dl = _fetch_json(SGD_RAW % (hub, "dialogues_%03d.json" % i), os.path.join(base, hub, "dialogues_%03d.json" % i))
            yield from _dialogue_rows(dl, lambda s, it: desc.get((s, it)), rng, split, same_group)


def multiwoz_intent(rng) -> Iterator[ts.Row]:
    """MultiWOZ 2.2 (MIT): restaurant / hotel / train / taxi / attraction ... intents."""
    base = os.path.join(RAW_ROOT, "multiwoz22")
    files = [("train", "train", i) for i in range(1, 18)] + [("val", "dev", i) for i in (1, 2)]
    dls = {(split, hub, i): _fetch_json(WOZ_RAW % (hub, i), os.path.join(base, hub, "dialogues_%03d.json" % i))
           for split, hub, i in files}
    intents = sorted({fr["state"]["active_intent"] for dl in dls.values() for d in dl for t in d["turns"]
                      for fr in t["frames"] if fr.get("state", {}).get("active_intent", "NONE") != "NONE"})
    human = {it: ts._human(it) for it in intents}
    same_group = lambda svc, it: ([human[x] for x in intents if x.endswith("_" + svc)], list(human.values()))  # noqa: E731
    for (split, hub, i), dl in dls.items():
        yield from _dialogue_rows(dl, lambda s, it: human.get(it), rng, split, same_group)


# ---- languages ------------------------------------------------------------------------------------------------------


def belebele(rng) -> Iterator[ts.Row]:
    for lang in BELEBELE_LANGS:
        for ex in ts._ds("facebook/belebele", lang, split="test"):
            opts = [ex["mc_answer%d" % i] for i in range(1, 5)]
            gold = int(ex["correct_answer_num"]) - 1
            q, lab = ts._fixed_choice(ex["question"], opts, gold, rng)
            yield _split(ex["link"], 15), json.dumps({"passage": ex["flores_passage"]}, ensure_ascii=False), q, lab, None


def xcopa(rng) -> Iterator[ts.Row]:
    for lang in XCOPA_LANGS:
        for hub in ("validation", "test"):
            for ex in ts._ds("cambridgeltl/xcopa", lang, split=hub):
                ins = "What was the cause?" if ex["question"] == "cause" else "What happened as a result?"
                q, lab = ts._fixed_choice(ins, [ex["choice1"], ex["choice2"]], int(ex["label"]), rng)
                yield _split("%s-%s" % (hub, ex["idx"]), 15), json.dumps({"premise": ex["premise"]}, ensure_ascii=False), q, lab, None


def xwinograd(rng) -> Iterator[ts.Row]:
    for lang in XWINO_LANGS:
        for i, ex in enumerate(ts._ds("Muennighoff/xwinograd", lang, split="test")):
            q, lab = ts._fixed_choice("Who or what does the blank (_) refer to?", [ex["option1"], ex["option2"]],
                                      int(ex["answer"]) - 1, rng)
            yield _split(ex["sentence"], 15), json.dumps({"sentence": ex["sentence"]}, ensure_ascii=False), q, lab, None


def tweet_sentiment_ml(rng) -> Iterator[ts.Row]:
    names = ["negative", "neutral", "positive"]
    for lang in TWEET_LANGS:
        for hub, split in (("train", "train"), ("validation", "val"), ("test", "val")):
            for ex in ts._ds("mteb/tweet_sentiment_multilingual", lang, split=hub):
                q, lab = ts._fixed_choice("What is the sentiment of this post?", names, int(ex["label"]), rng)
                yield split, json.dumps({"post": ex["text"]}, ensure_ascii=False), q, lab, None


# ---- code -----------------------------------------------------------------------------------------------------------


def _words(s: str) -> set:
    return set(w for w in re.findall(r"[a-z][a-z0-9_]{2,}", s.lower()))


def commit_message(rng) -> Iterator[ts.Row]:
    from huggingface_hub import hf_hub_download

    for lang in COMMIT_LANGS:
        path = hf_hub_download("bigcode/commitpackft", "data/%s/data.jsonl" % lang, repo_type="dataset")
        exs = []
        for line in open(path):
            ex = json.loads(line)
            diff = "".join(difflib.unified_diff(ex["old_contents"].splitlines(True), ex["new_contents"].splitlines(True),
                                                ex["old_file"], ex["new_file"], n=2))
            subj = ex["subject"].strip()
            if 40 <= len(diff) <= 2000 and 10 <= len(subj) <= 120:
                exs.append((ex["commit"], diff, subj))
        rng.shuffle(exs)
        exs = exs[:4000]
        subjects = [s for _, _, s in exs]
        for commit, diff, subj in exs:
            dw = _words(diff)  # the 3 other subjects (same language) sharing the most words with this diff
            opts = [subj] + sorted((s for s in subjects if s != subj), key=lambda s: (-len(_words(s) & dw), rng.random()))[:3]
            if len(set(opts)) < 4:
                continue
            rng.shuffle(opts)
            q = {"type": "choice", "instructions": "Which commit message describes this change?", "criteria": opts}
            yield _split(commit), json.dumps({"diff": diff}, ensure_ascii=False), q, opts.index(subj), None


def conala_snippet(rng) -> Iterator[ts.Row]:
    from huggingface_hub import hf_hub_download

    for fname, split in (("conala-paired-train.json", "train"), ("conala-paired-test.json", "val")):
        rows = [json.loads(l) for l in open(hf_hub_download("neulab/conala", "data/" + fname, repo_type="dataset"))]
        snippets = [r["snippet"] for r in rows]
        for r in rows:
            # the asker's own question: the rewritten intent quotes the snippet's variable names ('x', 'df'), which
            # alone picked the gold 0.56 of the time (audit)
            intent = r["intent"] or r["rewritten_intent"]
            if not intent:
                continue
            iw = _words(intent)  # the 3 snippets sharing the most words with the intent
            opts = [r["snippet"]] + sorted((s for s in snippets if s != r["snippet"]),
                                           key=lambda s: (-len(_words(s) & iw), rng.random()))[:3]
            if len(set(opts)) < 4:
                continue
            rng.shuffle(opts)
            q = {"type": "choice", "instructions": "Which Python snippet does this: %s" % intent, "criteria": opts}
            yield split, None, q, opts.index(r["snippet"]), None


SOURCES: Dict[str, dict] = {
    "massive_intent": {"train_cap": 60000, "fn": massive_intent, "repo": "AmazonScience/massive", "license": "CC BY 4.0 (MASSIVE 1.1)",
                       "use": "assistant / support intent, 20 languages"},
    "support_verticals": {"train_cap": 40000, "fn": support_verticals, "repo": "bitext/Bitext-telco-llm-chatbot-training-dataset",
                          "license": "CDLA-Sharing-1.0 (share-alike; 5 Bitext vertical datasets)",
                          "use": "customer support intent: banking, insurance, telco, travel, e-commerce"},
    "minds14_intent": {"train_cap": 9000, "fn": minds14_intent, "repo": "PolyAI/minds14", "license": "CC BY 4.0",
                       "use": "e-banking call intent, 14 languages (transcripts)"},
    "sgd_intent": {"train_cap": 40000, "fn": sgd_intent, "repo": "google-research-datasets/schema_guided_dstc8",
                   "license": "CC BY-SA 4.0 (share-alike; GitHub LICENSE.txt)", "use": "dialogue intent, 20 domains"},
    "multiwoz_intent": {"train_cap": 30000, "fn": multiwoz_intent, "repo": "pfb30/multi_woz_v22",
                        "license": "MIT (GitHub budzianowski/multiwoz LICENSE)", "use": "dialogue intent (booking)"},
    "belebele": {"train_cap": 20000, "fn": belebele, "repo": "facebook/belebele", "license": "CC BY-SA 4.0 (share-alike)",
                 "use": "reading comprehension, 24 languages"},
    "xcopa": {"fn": xcopa, "repo": "cambridgeltl/xcopa", "license": "CC BY 4.0", "use": "causal commonsense, 11 languages"},
    "xwinograd": {"fn": xwinograd, "repo": "Muennighoff/xwinograd", "license": "CC BY 4.0",
                  "use": "coreference commonsense, 6 languages"},
    "tweet_sentiment_ml": {"train_cap": 16000, "fn": tweet_sentiment_ml, "repo": "mteb/tweet_sentiment_multilingual", "license": "CC BY 3.0",
                           "use": "sentiment, 8 languages"},
    "commit_message": {"train_cap": 40000, "fn": commit_message, "repo": "bigcode/commitpackft",
                       "license": "MIT (dataset); code from permissively licensed repos",
                       "use": "code review: which message describes this diff, 15 languages"},
    "conala_snippet": {"fn": conala_snippet, "repo": "neulab/conala", "license": "MIT",
                       "use": "code: which snippet implements this intent"},
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
