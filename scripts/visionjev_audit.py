"""Audit prepared decision data for WRONG labels and GIVE-AWAY options (the validator only checks structure).

1. Independent re-derivation (synthetic sets): every label of ``synth_{invoice,refund,access,alert,shell_risk}`` is
   recomputed from the case state by code written separately from the generator; ``*_doc`` sets are replayed from
   their seeds (the generator is deterministic): replayed labels must equal the stored ones, re-derived labels must
   agree, and the renderer must not have dropped or clipped any line.
2. Shortcut tests (every choice / noul set): can a trivial heuristic find the answer without understanding?
   longest option; option with most word overlap with the state; for noul, the per-question majority learned on
   train applied to val. Flag when a heuristic beats chance / prior by a margin.

Results go to ``results/experiments.jsonl`` (kind ``audit``).

    .venv/bin/python scripts/visionjev_audit.py
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
ROOT = os.path.join(WORKSPACE, "data", "vqa")

from laya import visionjev_synth as synth  # noqa: E402  (only for replaying the doc seeds)
from laya.visionjev_log import log_experiment  # noqa: E402


# ---- independent rules (deliberately NOT imported from the generator) -------------------------------------------

def audit_invoice(st):
    inv = st["invoice"]
    sub = sum(l["qty"] * float(l["unit_price"]) for l in inv["lines"])
    total_ok = abs(sub + float(inv["tax"]) - float(inv["total"])) < 0.011
    po = {l["item"]: l["qty"] for l in st["purchase_order"]["lines"]}
    po_ok = all(po.get(l["item"]) == l["qty"] for l in inv["lines"])
    got = {l["item"]: l["qty_received"] for l in st["delivery_receipt"]["lines"]}
    delivered = all(got.get(l["item"]) == l["qty"] for l in inv["lines"])
    dup = any(h["invoice_no"] == inv["invoice_no"] and h["status"] == "paid" for h in st["vendor_payment_history"])
    n = (not total_ok) + (not po_ok) + (not delivered) + dup
    act = "reject" if dup else "request a corrected invoice" if not total_ok else "hold for review" if not (po_ok and delivered) else "pay"
    return {"sum of the line items": int(total_ok), "delivered in the invoiced": int(delivered), "already been paid": int(dup),
            "accounts payable": act, "paying this invoice": 3 if dup else min(3, n)}


def audit_refund(st):
    ev = [e["event"] for e in st["shipment"]["carrier_events"]]
    delivered = any(e.startswith("delivered") for e in ev)
    exp, shp = float(st["order"]["expected_weight_kg"]), float(st["shipment"]["shipped_weight_kg"])
    w_ok = abs(shp / exp - 1) <= 0.05
    photo = st["shipment"]["delivery_photo_on_file"] == "yes"
    claims = st["customer"]["missing_item_claims_last_12_months"]
    value = float(st["order"]["value_usd"])
    sup, con = (not delivered) or (not w_ok), delivered and w_ok and photo
    evid = "supports the claim" if sup else "contradicts the claim" if con else "insufficient evidence"
    act = ("refund" if value <= 150 else "reship the items") if sup else ("ask the customer for a photo" if not con else
                                                                         "escalate to a human agent" if claims >= 3 else "deny the claim")
    base = 0 if claims == 0 else 1 if claims <= 2 else 2 if claims <= 4 else 3
    risk = max(0, base - 1) if sup else min(3, base + 1) if con else base
    return {"show the parcel as delivered": int(delivered), "shipped weight match": int(w_ok),
            "evidence say": evid, "support system do next": act, "fraudulent": risk}


def audit_access(st):
    r = st["request"]
    levels = ["read", "write", "admin"]
    cap = st["access_policy_for_role"].get(r["system"])
    allowed = cap is not None and levels.index(r["access_level"]) <= levels.index(cap)
    approved = r["manager_approval"] == "approved"
    dec = "deny" if not allowed else "escalate to security" if r["access_level"] == "admin" else \
        "request manager approval" if not approved else "approve"
    return {"policy allow": int(allowed), "manager approved": int(approved), "contractor": int(r["employment"] == "contractor"),
            "access system do": dec,
            "sensitive": ["public", "internal", "confidential", "restricted"].index(st["system_info"]["data_classification"])}


def audit_alert(st):
    a = st["alert"]
    allow = a["process"] in st["process_allow_list"]
    hour = int(a["time"].split(":")[0])
    intel = st["threat_intel"]["indicator_match"] == "yes"
    base = {"repeated failed logins": 0, "port scan": 1, "malware signature": 2, "large outbound data transfer": 3}[a["type"]]
    sev = 0 if allow else min(3, base + intel + (st["host_info"]["criticality"] == "high"))
    return {"allow-list": int(allow), "outside business hours": int(not 9 <= hour < 18), "threat intelligence": int(intel),
            "How severe": sev, "security team do": ["close as benign", "monitor", "investigate", "contain the host"][sev]}


SHELL_RULES = [  # regex -> (risk, irreversible, exfil); written independently of the templates
    (r"\.ssh/id_rsa.*curl|^env \| curl|scp .*credentials", (3, 0, 1)),
    (r"^rm -rf /$|dd if=/dev/zero of=/dev/|mkfs\.|s3 rm .*--recursive|delete namespace production|push --force origin main$", (3, 1, 0)),
    (r"chmod -R 777 /$|\| sudo bash", (3, 0, 0)),
    (r"rm -rf .*/build|reset --hard|push --force origin (?!main)|DROP TABLE|truncate -s 0", (2, 1, 0)),
    (r"docker rm -f|kubectl delete pod", (2, 0, 0)),
    (r"^(mkdir|pip install|git checkout -b|npm install|cp |git commit|docker build|touch )", (1, 0, 0)),
    (r"^(ls|cat |git status|grep|docker ps|kubectl get|df -h|git log)", (0, 0, 0)),
]


def audit_shell(st):
    cmd = st["proposed_command"]
    for pat, (risk, irr, exf) in SHELL_RULES:
        if re.search(pat, cmd):
            return {"risk": risk, "irreversible data loss": irr, "secrets or credentials": exf,
                    "guardrail": ["run it", "run it in a sandbox first", "ask a human to approve", "block it"][risk]}
    return None


AUDITORS = {"synth_invoice": audit_invoice, "synth_refund": audit_refund, "synth_access": audit_access,
            "synth_alert": audit_alert, "synth_shell_risk": audit_shell}


def expected(q, rules):
    """The rule answer for this question, as the label index the record should carry."""
    ins = q["instructions"]
    for key, val in rules.items():
        if key.lower() in ins.lower() or (key == "risk" and "risk" in ins.lower() and q["type"] == "score"):
            if q["type"] == "choice":
                return q["criteria"].index(val) if val in q["criteria"] else "missing-option"
            return int(val)
    return "unmatched"


def audit_text_set(name, fn):
    bad, n, unmatched = [], 0, 0
    for split in ("train", "val"):
        for line in open(os.path.join(ROOT, name, split + ".jsonl")):
            r = json.loads(line)
            rules = fn(json.loads(r["state_text"]))
            if rules is None:
                unmatched += 1
                continue
            e = expected(r["question"], rules)
            n += 1
            if e == "unmatched":
                unmatched += 1
            elif e != r["label"]:
                bad.append((r["id"], r["question"]["instructions"][:60], r["label"], e))
    return {"checked": n, "mismatches": len(bad), "unmatched": unmatched, "examples": bad[:5]}


def audit_doc_set(wf, fn, n_train=1500, n_val=200):
    """Replay the seeded generator for the rendered set; compare stored labels, re-derive, check rendering."""
    name, gen = "synth_%s_doc" % wf, synth.WORKFLOWS[wf]
    stored = {s: [json.loads(l) for l in open(os.path.join(ROOT, name, s + ".jsonl"))] for s in ("train", "val")}
    replay_bad = rule_bad = dropped = clipped = 0
    for split, n, seed in (("train", n_train, 1), ("val", n_val, 2)):
        rng = random.Random("%s-doc-%s" % (wf, seed))
        k = 0
        for i in range(n):
            state, title, qs = gen(rng)
            img = synth.render_state(state, title, rng)
            dropped += img.info["dropped_lines"] > 0
            clipped += img.info["clipped_lines"] > 0
            rules = fn(state)
            for (_, q, lab) in qs:
                rec = stored[split][k]
                k += 1
                replay_bad += rec["label"] != lab or rec["question"]["instructions"] != q["instructions"]
                rule_bad += expected(q, rules) != lab
    return {"cases": n_train + n_val, "replay_mismatches": replay_bad, "rule_mismatches": rule_bad,
            "pages_with_dropped_lines": dropped, "pages_with_clipped_lines": clipped}


# ---- shortcut tests -----------------------------------------------------------------------------------------------

def _words(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def _opts(q):
    """Option texts as the model sees them (dict criteria render as 'key: description')."""
    c = q["criteria"]
    return ["%s: %s" % (k, v) if v else str(k) for k, v in c.items()] if isinstance(c, dict) else [str(x) for x in c]


def shortcuts(name):
    out = {}
    path = lambda s: os.path.join(ROOT, name, s + ".jsonl")  # noqa: E731
    if not os.path.exists(path("val")):
        return out
    val = [json.loads(l) for l in open(path("val"))]
    ch = [r for r in val if r["question"]["type"] == "choice"]
    if ch:
        longest = overlap = 0
        for r in ch:
            o = _opts(r["question"])
            longest += max(range(len(o)), key=lambda i: len(o[i])) == r["label"]
            sw = _words(r.get("state_text") or "") | _words(r["question"]["instructions"])
            overlap += max(range(len(o)), key=lambda i: len(_words(o[i]) & sw)) == r["label"]
        chance = sum(1 / len(r["question"]["criteria"]) for r in ch) / len(ch)
        out["choice"] = {"n": len(ch), "chance": round(chance, 3), "longest_option": round(longest / len(ch), 3),
                         "state_overlap": round(overlap / len(ch), 3)}
    pairs = []
    for r in ch:
        st = r.get("state_text") or ""
        for a, b in (("function_A", "function_B"), ("implementation_A", "implementation_B"), ("patch_A", "patch_B")):
            if '"%s"' % a in st:
                s = json.loads(st)
                pairs.append((s[a], s[b], r["label"]))
    if pairs:  # "the shorter one is the answer" (vulnerability fixes add code; a mutant can change the length)
        hit = sum((0 if len(a) < len(b) else 1) == lab for a, b, lab in pairs)
        out["pair_shorter_is_gold"] = round(hit / len(pairs), 3)
    num = []
    for r in ch:
        try:
            vals = [float(o) for o in _opts(r["question"])]
        except ValueError:
            continue
        if len(vals) >= 3:
            num.append((vals, r["label"]))
    if len(num) >= 20:  # near-miss numeric distractors (x-1, x+1) would make the true value the median
        hit = sum(sorted(range(len(v)), key=lambda i: v[i])[(len(v) - 1) // 2] == lab for v, lab in num)
        out["numeric_median"] = {"n": len(num), "median_is_gold": round(hit / len(num), 3),
                                 "chance": round(sum(1 / len(v) for v, _ in num) / len(num), 3)}
    nl = [r for r in val if r["question"]["type"] == "noul"]
    if nl and os.path.exists(path("train")):
        by_q = defaultdict(Counter)
        for l in open(path("train")):
            r = json.loads(l)
            if r["question"]["type"] == "noul":
                by_q[r["question"]["instructions"]][r["label"]] += 1
        prior = Counter(r["label"] for r in nl).most_common(1)[0][1] / len(nl)
        hit = sum((by_q[r["question"]["instructions"]].most_common(1)[0][0] if by_q[r["question"]["instructions"]] else 1) == r["label"] for r in nl)
        out["noul"] = {"n": len(nl), "majority_prior": round(prior, 3), "question_only": round(hit / len(nl), 3)}
    return out


def main():
    report = {"rederive": {}, "doc_replay": {}, "shortcuts": {}, "flags": []}
    for name, fn in AUDITORS.items():
        report["rederive"][name] = r = audit_text_set(name, fn)
        print("%-18s checked %6d  mismatches %d  unmatched %d %s" % (name, r["checked"], r["mismatches"], r["unmatched"], r["examples"][:2]))
        if r["mismatches"] or r["unmatched"]:
            report["flags"].append("%s: %d label mismatches, %d unmatched" % (name, r["mismatches"], r["unmatched"]))
    for wf in ("invoice", "refund", "access", "alert"):
        report["doc_replay"][wf] = r = audit_doc_set(wf, AUDITORS["synth_" + wf])
        print("synth_%s_doc  %s" % (wf, r))
        if any(r[k] for k in ("replay_mismatches", "rule_mismatches", "pages_with_dropped_lines", "pages_with_clipped_lines")):
            report["flags"].append("synth_%s_doc: %s" % (wf, r))
    names = sorted(d for d in os.listdir(ROOT) if not d.startswith(".") and os.path.exists(os.path.join(ROOT, d, "val.jsonl")))
    for n in names:
        sc = shortcuts(n)
        if not sc:
            continue
        report["shortcuts"][n] = sc
        c, u = sc.get("choice"), sc.get("noul")
        warn = []
        if c and max(c["longest_option"], c["state_overlap"]) > c["chance"] + 0.2:
            warn.append("choice heuristic %.2f vs chance %.2f" % (max(c["longest_option"], c["state_overlap"]), c["chance"]))
        if "pair_shorter_is_gold" in sc and abs(sc["pair_shorter_is_gold"] - 0.5) > 0.1:
            warn.append("pair: length predicts the answer (shorter-is-gold %.2f)" % sc["pair_shorter_is_gold"])
        nm = sc.get("numeric_median")
        if nm and nm["median_is_gold"] > nm["chance"] + 0.2:
            warn.append("numeric options: the median is gold %.2f vs chance %.2f" % (nm["median_is_gold"], nm["chance"]))
        if u and u["question_only"] > max(u["majority_prior"], 0.5) + 0.15:
            warn.append("noul question-only %.2f vs prior %.2f" % (u["question_only"], u["majority_prior"]))
        if warn:
            report["flags"].append("%s: %s" % (n, "; ".join(warn)))
    print("\nFLAGS (%d):" % len(report["flags"]))
    for f in report["flags"]:
        print("  -", f)
    log_experiment("audit", "label-and-shortcut-audit", config={"script": "visionjev_audit.py", "root": ROOT},
                   metrics=report, datasets=list(report["shortcuts"]))


if __name__ == "__main__":
    main()
