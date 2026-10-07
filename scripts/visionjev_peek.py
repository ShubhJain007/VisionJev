"""Read a prepared dataset by eye: manifest summary, then N rows with the state pretty-printed, the question, every
option and the gold marked (soft targets and rationale shown when present).

    .venv/bin/python scripts/visionjev_peek.py agent_verify_pair            # 3 random val rows
    .venv/bin/python scripts/visionjev_peek.py agent_outcome -n 5 --split train --seed 7
    .venv/bin/python scripts/visionjev_peek.py agent_next_action --id agent_next_action-val-12
    .venv/bin/python scripts/visionjev_peek.py gen_rules --grep "cannot be determined"
"""
import argparse
import json
import os
import random
import textwrap

ROOT = os.environ.get("VISIONJEV_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "vqa"))


def _state(text, width):
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return textwrap.fill(str(text), width)
    if not isinstance(obj, dict):
        return json.dumps(obj, indent=1, ensure_ascii=False)
    out = []
    for k, v in obj.items():
        v = v if isinstance(v, str) else json.dumps(v, indent=1, ensure_ascii=False)
        out.append("[%s]\n%s" % (k, v.replace("\r", "")))
    return "\n\n".join(out)


def show(r, width, max_chars):
    q = r["question"]
    print("=" * width)
    print("id: %s   type: %s" % (r.get("id"), q["type"]))
    st = _state(r.get("state_text") or "", width)
    if len(st) > max_chars:
        st = st[:max_chars] + "\n... [%d more chars; --max-chars 0 for all]" % (len(st) - max_chars)
    if r.get("image"):
        print("image: %s" % r["image"])
    print("-" * width + "\nSTATE\n" + st)
    print("-" * width + "\nQUESTION: " + q["instructions"])
    target = r.get("target")
    if q["type"] == "noul":
        print("  gold: %s" % ("yes" if r["label"] == 1 else "no") + ("   soft P(no), P(yes) = %s" % target if target else ""))
    else:
        for i, o in enumerate(q.get("criteria") or []):
            mark = ">>" if i == r["label"] else "  "
            soft = "  (%.3f)" % target[i] if target else ""
            print("  %s %d. %s%s" % (mark, i, str(o)[:400], soft))
    if r.get("rationale"):
        print("RATIONALE: " + r["rationale"])
    if r.get("provenance"):
        print("PROVENANCE: " + json.dumps(r["provenance"], ensure_ascii=False)[:400])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--grep", default=None, help="only rows whose JSON contains this text")
    ap.add_argument("--width", type=int, default=110)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    d = os.path.join(a.root, a.name)
    man = json.load(open(os.path.join(d, "manifest.json")))
    print("%s: counts %s | licence %s | use: %s" % (a.name, man.get("counts"), (man.get("license") or {}).get("annotations"),
                                                    man.get("use")))
    rows = [l for l in open(os.path.join(d, a.split + ".jsonl")) if (a.grep is None or a.grep in l)]
    if a.id:
        rows = [l for l in rows if json.loads(l).get("id") == a.id]
    print("%d rows in %s%s" % (len(rows), a.split, " matching" if a.grep or a.id else ""))
    for l in random.Random(a.seed).sample(rows, min(a.n, len(rows))):
        show(json.loads(l), a.width, a.max_chars if a.max_chars > 0 else 10 ** 9)


if __name__ == "__main__":
    main()
