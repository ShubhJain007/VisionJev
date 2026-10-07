"""Autonomous keep / discard verdict for new checkpoints against a baseline, from results/experiments.jsonl
(autoresearch rule: decide from fixed metrics, write it down, do not wait for a human).

For each candidate: JevBench public accuracy and ECE (`jevbench-public:<name>-len4096`, and the `-think70` variant when
present), zero-shot / never-seen / held-out-wording calibrated accuracy ("all"), robustness macro accuracy. A candidate
is KEPT when its JevBench accuracy beats the baseline's by >= --min-gain points and no generalisation metric drops by
more than --max-regression points. The best kept candidate becomes `checkpoints/best` (a symlink) unless --no-link.
Writes <out>/VERDICT.md and prints it; exit 0 when something was kept, 1 otherwise.

    .venv/bin/python scripts/visionjev_verdict.py --base qwen4b-v6b2 --new qwen4b-v6b3 qwen4b-v6b3-soup75 --out ../results/pod/b3
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))


def load(log):
    return [json.loads(l) for l in open(log) if l.strip()]


def last(recs, kind, name):
    xs = [r for r in recs if r.get("kind") == kind and r.get("name") == name]
    return xs[-1] if xs else None


def metrics(recs, name):
    m = {}
    jb = last(recs, "jevbench_public", "jevbench-public:%s-len4096" % name)
    if jb:
        s = jb["metrics"]["summary"]
        m["jevbench"] = round(100 * s["accuracy"], 1)
        m["jevbench_ece"] = round(s["ece"]["ece"] if isinstance(s["ece"], dict) else s["ece"], 3)
    jt = last(recs, "jevbench_public", "jevbench-public:%s-len4096-think70" % name)
    if jt:
        s = jt["metrics"]["summary"]
        if s.get("n_valid", 231) >= 200:
            m["jevbench_think"] = round(100 * s["accuracy"], 1)
    for kind, tag, key in (("zero_shot", "zeroshot", "zero_shot"), ("never_seen", "novel", "never_seen"),
                           ("heldout_formulation", "hof", "held_out_wording")):
        r = last(recs, kind, "%s:%s" % (tag, name))
        if r:
            m[key] = round(100 * r["metrics"]["calibrated"]["all"]["acc"], 1)
    rb = last(recs, "robustness", "robust:%s" % name)
    if rb:
        m["robust"] = round(100 * rb["metrics"]["compact"]["macro"]["orig"]["acc"], 1)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--log", default=os.path.join(WORKSPACE, "results", "experiments.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gain", type=float, default=1.0)
    ap.add_argument("--max-regression", type=float, default=1.0)
    ap.add_argument("--no-link", action="store_true")
    a = ap.parse_args(argv)
    recs = load(a.log)
    base = metrics(recs, a.base)
    rows, kept = [], []
    keys = ["jevbench", "jevbench_ece", "jevbench_think", "zero_shot", "never_seen", "held_out_wording", "robust"]
    for n in a.new:
        m = metrics(recs, n)
        why = []
        if "jevbench" not in m or "jevbench" not in base:
            why.append("no JevBench result")
        else:
            gain = m["jevbench"] - base["jevbench"]
            if gain < a.min_gain:
                why.append("JevBench +%.1f < %.1f" % (gain, a.min_gain))
            for k in ("zero_shot", "never_seen", "held_out_wording", "robust"):
                if k in m and k in base and m[k] < base[k] - a.max_regression:
                    why.append("%s %.1f -> %.1f" % (k, base[k], m[k]))
        ok = not why
        rows.append((n, m, ok, why))
        if ok:
            kept.append((m["jevbench"], -m.get("jevbench_ece", 1), n))
    lines = ["# Verdict: %s vs %s" % (", ".join(a.new), a.base), "",
             "| checkpoint | " + " | ".join(keys) + " | verdict |", "|---|" + "---|" * (len(keys) + 1)]
    lines.append("| %s (base) | %s | — |" % (a.base, " | ".join(str(base.get(k, "—")) for k in keys)))
    for n, m, ok, why in rows:
        lines.append("| %s | %s | %s |" % (n, " | ".join(str(m.get(k, "—")) for k in keys), "KEEP" if ok else "discard: " + "; ".join(why)))
    best = None
    if kept:
        best = max(kept)[2]
        lines += ["", "Best kept: **%s**" % best]
        if not a.no_link:
            link = os.path.join(WORKSPACE, "checkpoints", "best")
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.join(WORKSPACE, "checkpoints", best), link)
            lines.append("`checkpoints/best` -> `%s`" % best)
    else:
        lines += ["", "Nothing kept: %s stays the reference." % a.base]
    text = "\n".join(lines) + "\n"
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "VERDICT.md"), "w").write(text)
    print(text)
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
