"""Consistency probes with exact labels (generated; eval only): does the model answer from facts stated in the state?

``probe_state_facts``: a random business record (order, support ticket, invoice, account) as ``state_text``, and one
question about a field that is literally in it:

* ``lookup``     noul    "Is the <field> <value>?" (the stated value, or another value of that field; balanced)
* ``negated``    noul    "Is the <field> not <value>?" (same facts, negated frame; balanced)
* ``which``      choice  "What is the <field>?" (the stated value + 3 other values of that field)
* ``compare``    noul    "Is the <number field> greater than <t>?" (t never equals the value; balanced)
* ``present``    noul    "Does the record include a <field>?" (a field of the schema that is present or dropped)

A model that reads the state should be near 100% on all five; the negated / lookup pair also measures
P(yes | Q) + P(yes | not Q) ~ 1 on the same facts. The probe kind ends each record id (``...-7-negated``).

    python -m laya.visionjev_probes
"""
import argparse
import json
import os
import random
import shutil
import time
from collections import Counter

from .visionjev_data import DATA_ROOT

SCHEMAS = {
    "order": {"status": ["pending", "paid", "shipped", "delivered", "cancelled", "refunded"],
              "carrier": ["UPS", "FedEx", "DHL", "USPS", "local courier"],
              "payment_method": ["card", "bank transfer", "PayPal", "gift card", "cash on delivery"],
              "country": ["US", "Canada", "Germany", "India", "Brazil", "Japan"],
              "items": ("int", 1, 12), "total_usd": ("float", 5, 900)},
    "ticket": {"priority": ["low", "medium", "high", "urgent"],
               "category": ["billing", "login", "shipping", "refund", "bug report", "feature request"],
               "channel": ["email", "chat", "phone", "web form"],
               "assignee_team": ["tier 1", "tier 2", "engineering", "finance"],
               "age_hours": ("int", 1, 200), "reopen_count": ("int", 0, 5)},
    "invoice": {"currency": ["USD", "EUR", "GBP", "INR", "JPY"],
                "vendor_type": ["supplier", "contractor", "utility", "software"],
                "approval_state": ["draft", "submitted", "approved", "rejected", "paid"],
                "terms": ["net 15", "net 30", "net 60", "due on receipt"],
                "amount": ("float", 50, 20000), "line_items": ("int", 1, 30)},
    "account": {"plan": ["free", "starter", "pro", "enterprise"],
                "region": ["us-east", "us-west", "eu-central", "ap-south"],
                "mfa": ["enabled", "disabled"], "role": ["viewer", "editor", "admin", "owner"],
                "seats": ("int", 1, 500), "failed_logins_24h": ("int", 0, 20)},
}
KINDS = ("lookup", "negated", "which", "compare", "present")


def _human(f: str) -> str:
    return f.replace("_", " ")


def _record(rng, schema):
    rec = {}
    for f, space in SCHEMAS[schema].items():
        if isinstance(space, tuple):
            kind, lo, hi = space
            rec[f] = rng.randint(lo, hi) if kind == "int" else round(rng.uniform(lo, hi), 2)
        else:
            rec[f] = rng.choice(space)
    return rec


def _row(rng, i):
    schema = rng.choice(sorted(SCHEMAS))
    rec = _record(rng, schema)
    kind = KINDS[i % len(KINDS)]
    cat = [f for f, s in SCHEMAS[schema].items() if not isinstance(s, tuple)]
    num = [f for f, s in SCHEMAS[schema].items() if isinstance(s, tuple)]
    truth = rng.random() < 0.5
    if kind in ("lookup", "negated"):
        f = rng.choice(cat)
        v = rec[f] if truth else rng.choice([x for x in SCHEMAS[schema][f] if x != rec[f]])
        if kind == "lookup":
            q, lab = "Is the %s %s?" % (_human(f), v), int(truth)
        else:
            q, lab = "Is the %s not %s?" % (_human(f), v), int(not truth)
        question = {"type": "noul", "instructions": q, "criteria": None}
    elif kind == "which":
        f = rng.choice(cat)
        opts = [rec[f]] + rng.sample([x for x in SCHEMAS[schema][f] if x != rec[f]], min(3, len(SCHEMAS[schema][f]) - 1))
        rng.shuffle(opts)
        question = {"type": "choice", "instructions": "What is the %s?" % _human(f), "criteria": opts}
        lab = opts.index(rec[f])
    elif kind == "compare":
        f = rng.choice(num)
        v = rec[f]
        _, lo, hi = SCHEMAS[schema][f]
        t = v - rng.randint(1, max(1, int((hi - lo) / 4))) if truth else v + rng.randint(0, max(1, int((hi - lo) / 4)))
        if t == v:
            t = v + 1
        question = {"type": "noul", "instructions": "Is the %s greater than %s?" % (_human(f), t), "criteria": None}
        lab = int(v > t)
    else:
        f = rng.choice(cat + num)
        if not truth:
            rec.pop(f)
        question = {"type": "noul", "instructions": "Does the record include a %s?" % _human(f), "criteria": None}
        lab = int(truth)
    state = {"record_type": schema, schema: rec}
    return {"state_text": json.dumps(state), "question": question, "label": lab, "probe": kind}


def build(root=DATA_ROOT, n=1500, seed=0):
    name = "probe_state_facts"
    rng = random.Random(seed)
    rows = [_row(rng, i) for i in range(n)]
    tmp, final = os.path.join(root, ".tmp-" + name), os.path.join(root, name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with open(os.path.join(tmp, "val.jsonl"), "w") as f:
        for i, r in enumerate(rows):
            rec = {k: v for k, v in r.items() if k != "probe"}
            f.write(json.dumps(dict(rec, id="%s-val-%d-%s" % (name, i, r["probe"]))) + "\n")  # kind in the id
    m = {"dataset": name, "source": "generated (VisionJev visionjev_probes)", "revision": "v1",
         "license": {"annotations": "generated in-house, exact labels", "images": "none (text-only)",
                     "commercial": True, "verified": "generated in-house"},
         "use": "consistency probe: facts stated in the state (eval only)", "eval_only": True, "seed": seed,
         "counts": {"val": n}, "probes": dict(Counter(r["probe"] for r in rows)),
         "label_counts": {"val": {str(k): v for k, v in Counter(r["label"] for r in rows).items()}}}
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    with open(os.path.join(final, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    print("%s: %d rows %s labels %s" % (name, n, m["probes"], m["label_counts"]["val"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    build(a.root, a.n)
