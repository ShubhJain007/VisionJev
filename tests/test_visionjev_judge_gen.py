"""Planted-error judge sets: every chain is re-derived here FROM THE REQUEST TEXT ALONE (own parsers, own arithmetic,
own formatting), so a response's correctness is decided independently of ``laya.visionjev_judge_gen``. Corrupted
responses must differ from the correct ones in exactly one step (later steps may carry the error), rubric levels must
follow from the rubric text, adequacy requirements are re-checked one by one, and the no-shortcut properties (label
balance, equal lengths, digit counts, gold position, longer-is-gold) are measured on samples."""
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP
from decimal import Decimal as D

import pytest

from laya import visionjev_judge_gen as G

TOK = re.compile(r"\d{2} [A-Z][a-z]{2} \d{4}|\d{1,2}:\d{2}|\d+(?:,\d{3})*(?:\.\d+)?")
DATE = re.compile(r"\d{2} [A-Z][a-z]{2} \d{4}")
CLOCK = re.compile(r"\b\d{2}:\d{2}\b")
ENUM = re.compile(r"^(?:\d+\. |Step \d+: )", re.M)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "vqa")


# ---- own formatting and parsing -----------------------------------------------------------------------------------------
def fn(x):
    return format(x.quantize(D("0.001")), "f").rstrip("0").rstrip(".") or "0"


def fm(x):
    return format(x.quantize(D("0.01")), ",.2f")


def fi(x):
    return "%d" % x


def fd(x):
    return x.strftime("%d %b %Y")


def fc(x):
    return "%02d:%02d" % divmod(int(x), 60)


def cents(x):
    return x.quantize(D("0.01"), ROUND_HALF_UP)


def toks(text):
    return TOK.findall(ENUM.sub("", text))


def nums(text):
    """Plain numbers of a text in order (dates and clock times excluded), as Decimals."""
    return [D(t.replace(",", "")) for t in toks(text) if not DATE.fullmatch(t) and not CLOCK.fullmatch(t)]


def dates(text):
    return [datetime.strptime(t, "%d %b %Y").date() for t in DATE.findall(text)]


class Ch:
    """A chain with per-step overrides (an override replaces that step's value and the rest is recomputed)."""

    def __init__(self, ov=None):
        self.ov, self.v, self.f = ov or {}, [], []

    def put(self, f, x):
        x = self.ov.get(len(self.v), x)
        self.v.append(x)
        self.f.append(f)
        return x

    def strs(self):
        return [f(x) for f, x in zip(self.f, self.v)]


# ---- givens(request) and chain(givens, ov, skip) per family ---------------------------------------------------------------
def g_flow(r):
    n = nums(r)
    g = {"money": "$" in r, "cap": "capacity" in r, "pause": " first " in r, "two": bool(re.search(r" and (?:a second unit )?then ", r)),
         "out": "Finally" in r}
    i = 0
    g["S"] = n[i]; i += 1
    if g["cap"]:
        g["C"] = n[i]; i += 1
    g["r1"], g["t1"] = n[i], n[i + 1]; i += 2
    if g["two"]:
        g["r2"], g["t2"] = n[i], n[i + 1]; i += 2
    if g["pause"]:
        g["p"] = n[i]; i += 1
    g["A"] = n[i]; i += 1
    if g["out"]:
        g["B"] = n[i]; i += 1
    assert i == len(n)
    return g


def c_flow(g, ov=None, skip=False):
    c, F = Ch(ov), (fm if g["money"] else fn)
    t = g["t1"]
    if g["pause"]:
        t = c.put(fn, g["t1"] if skip else g["t1"] - g["p"])
    l = c.put(F, g["r1"] * t)
    if g["two"]:
        l2 = c.put(F, g["r2"] * g["t2"])
        l = c.put(F, l + l2)
    rem = c.put(F, g["S"] - l)
    mid = c.put(F, rem + g["A"])
    if g["out"]:
        c.put(F, mid - g["B"])
    return c


def g_invoice(r):
    pairs = [(D(q), D(p.replace(",", ""))) for q, p in re.findall(r"(\d+)[^\d$]{1,26}\$([\d,]+\.\d\d)", r)]
    g = {"n": len(pairs), "pct": D(re.search(r"(\d+(?:\.\d)?)%", r).group(1))}
    for i, (q, p) in enumerate(pairs, 1):
        g["q%d" % i], g["p%d" % i] = q, p
    m = re.search(r"does not apply to the ([a-z ]+)\.", r)
    if m:
        g["var"], g["j"] = "exempt", r[:r.index(m.group(1))].count("$")
    elif "not taxed" in r:
        g["var"], g["sh"] = "ship", D(re.search(r"delivery charge of \$([\d,]+\.\d\d)", r).group(1).replace(",", ""))
    else:
        g["var"] = "disc" if "discount" in r or "off the subtotal" in r else "tax"
    return g


def c_invoice(g, ov=None, skip=False):
    c = Ch(ov)
    ls = [c.put(fm, g["q%d" % i] * g["p%d" % i]) for i in range(1, g["n"] + 1)]
    sub = c.put(fm, sum(ls))
    if g["var"] in ("disc", "exempt"):
        base = sub
        if g["var"] == "exempt":
            base = c.put(fm, sub if skip else sub - ls[g["j"]])
        disc = c.put(fm, cents(base * g["pct"] / 100))
        c.put(fm, sub - disc)
    elif g["var"] == "ship":
        tax = c.put(fm, cents((sub + g["sh"] if skip else sub) * g["pct"] / 100))
        c.put(fm, sub + tax + g["sh"])
    else:
        tax = c.put(fm, cents(sub * g["pct"] / 100))
        c.put(fm, sub + tax)
    return c


def g_units_trip(r):
    n = nums(r)
    return {"d": n[0], "c": n[3], "pr": n[5], "rt": "each way" in r}


def c_units_trip(g, ov=None, skip=False):
    c = Ch(ov)
    d = c.put(fn, g["d"] if skip else g["d"] * 2) if g["rt"] else g["d"]
    km = c.put(fn, d * D("1.6"))
    fuel = c.put(fn, km * g["c"] / 100)
    c.put(fm, cents(fuel * g["pr"]))
    return c


def g_units_cargo(r):
    n = nums(r)
    return {"n": n[0], "w": n[1], "T": n[2], "rate": n[5]}


def c_units_cargo(g, ov=None, skip=False):
    c = Ch(ov)
    lb = c.put(fn, g["n"] * g["w"])
    kg = c.put(fn, lb * D("0.45"))
    gross = c.put(fn, kg + g["T"])
    c.put(fm, cents(gross * g["rate"]))
    return c


def g_units_paint(r):
    n = nums(r)
    return dict(zip("LHOkcp", n))


def c_units_paint(g, ov=None, skip=False):
    c = Ch(ov)
    area = c.put(fn, g["L"] * g["H"])
    net = c.put(fn, area if skip else area - g["O"])
    coat = c.put(fn, net * g["k"])
    lit = c.put(fn, coat / g["c"])
    c.put(fm, cents(lit * g["p"]))
    return c


def g_units_time(r):
    n = nums(r)
    return {"m": n[0], "n": n[1], "rate": n[2], "fee": n[3] if len(n) > 3 else None}


def c_units_time(g, ov=None, skip=False):
    c = Ch(ov)
    mins = c.put(fn, g["m"] * g["n"])
    hrs = c.put(fn, mins / 60)
    chg = c.put(fm, cents(hrs * g["rate"]))
    if g["fee"] is not None:
        c.put(fm, chg + g["fee"])
    return c


def g_units_dozen(r):
    n = nums(r)
    return {"n": n[0], "w": n[1], "pr": n[2]}


def c_units_dozen(g, ov=None, skip=False):
    c = Ch(ov)
    pw = c.put(fn, g["n"] * 12)
    al = c.put(fn, pw * g["w"])
    c.put(fm, al * g["pr"])
    return c


def g_dates_chain(r):
    return {"D0": dates(r)[0], "n": nums(r)}


def c_dates_chain(g, ov=None, skip=False):
    c = Ch(ov)
    d = g["D0"]
    for k in g["n"][:3]:
        d = c.put(fd, d + timedelta(days=int(k)))
    if len(g["n"]) == 4:
        c.put(fd, d - timedelta(days=int(g["n"][3])))
    return c


def g_dates_hire(r):
    d1, d2 = dates(r)
    n = nums(r)
    return {"D1": d1, "D2": d2, "rate": n[0], "fee": n[1], "sun": "Sundays" in r}


def c_dates_hire(g, ov=None, skip=False):
    c = Ch(ov)
    days = c.put(fi, D((g["D2"] - g["D1"]).days + 1))
    ch = days
    if g["sun"]:
        sundays = sum(1 for i in range((g["D2"] - g["D1"]).days + 1) if (g["D1"] + timedelta(days=i)).weekday() == 6)
        ch = c.put(fi, days if skip else days - sundays)
    hire = c.put(fm, ch * g["rate"])
    c.put(fm, hire + g["fee"])
    return c


def g_dates_trip(r):
    n = nums(r)
    return {"D0": dates(r)[0], "n1": n[0], "n2": n[1], "ra": n[2], "rb": n[3]}


def c_dates_trip(g, ov=None, skip=False):
    c = Ch(ov)
    mv = c.put(fd, g["D0"] + timedelta(days=int(g["n1"])))
    c.put(fd, mv + timedelta(days=int(g["n2"])))
    ca = c.put(fm, g["n1"] * g["ra"])
    cb = c.put(fm, g["n2"] * g["rb"])
    c.put(fm, ca + cb)
    return c


def g_pay_overtime(r):
    n = nums(r)
    return {"h": n[0], "w": n[1], "base": n[2], "mult": n[3], "p": n[4]}


def c_pay_overtime(g, ov=None, skip=False):
    c = Ch(ov)
    ot = c.put(fn, g["h"] - g["base"])
    reg = c.put(fm, g["base"] * g["w"])
    otp = c.put(fm, cents(ot * g["w"] * (1 if skip else g["mult"])))
    gross = c.put(fm, reg + otp)
    tax = c.put(fm, cents(gross * g["p"] / 100))
    c.put(fm, gross - tax)
    return c


def g_pay_stacked(r):
    n = nums(r)
    return {"P": n[0], "d1": n[1], "d2": n[2], "tx": n[3]}


def c_pay_stacked(g, ov=None, skip=False):
    c = Ch(ov)
    a = c.put(fm, cents(g["P"] * (100 - g["d1"]) / 100))
    b = c.put(fm, cents(a * (100 - g["d2"]) / 100))
    tax = c.put(fm, cents(b * g["tx"] / 100))
    c.put(fm, b + tax)
    return c


def g_pay_commission(r):
    n = nums(r)
    return {"B": n[0], "c1": n[1], "T": n[2], "c2": n[3], "S": n[4]}


def c_pay_commission(g, ov=None, skip=False):
    c = Ch(ov)
    ab = c.put(fm, g["S"] - g["T"])
    k1 = c.put(fm, cents(g["T"] * g["c1"] / 100))
    k2 = c.put(fm, cents(ab * g["c2"] / 100))
    c.put(fm, g["B"] + k1 + k2)
    return c


def g_pay_latefee(r):
    n = nums(r)
    return {"A": n[0], "m": n[1], "p": n[2], "f": n[3], "grace": "grace month" in r}


def c_pay_latefee(g, ov=None, skip=False):
    c = Ch(ov)
    f1 = c.put(fm, cents(g["A"] * g["p"] / 100))
    m = g["m"]
    if g["grace"]:
        m = c.put(fn, g["m"] if skip else g["m"] - 1)
    fees = c.put(fm, f1 * m)
    c.put(fm, g["A"] + fees + g["f"])
    return c


def g_avg_readings(r):
    n = nums(r)
    xs, T = n[:-1], n[-1]
    drop = None
    if re.search(r"highest|largest", r):
        drop = max(xs)
    elif re.search(r"lowest|smallest", r):
        drop = min(xs)
    kept = [x for x in xs if x != drop]
    return {"xs": xs, "kept": kept, "T": T, "money": "$" in r}


def c_avg_readings(g, ov=None, skip=False):
    c, F = Ch(ov), (fm if g["money"] else fn)
    s = c.put(F, sum(g["xs"] if skip else g["kept"]))
    mean = c.put(F, s / len(g["kept"]))
    c.put(F, mean - g["T"])
    return c


def g_avg_cost(r):
    n = nums(r)
    return {"q1": n[0], "c1": n[1], "q2": n[2], "c2": n[3], "sold": n[4]}


def c_avg_cost(g, ov=None, skip=False):
    c = Ch(ov)
    spend = c.put(fm, g["q1"] * g["c1"] + g["q2"] * g["c2"])
    units = c.put(fi, g["q1"] + g["q2"])
    avg = c.put(fm, cents(spend / units))
    left = c.put(fi, units - g["sold"])
    c.put(fm, left * avg)
    return c


def g_route(r):
    n = nums(r)
    h, m = CLOCK.search(r).group(0).split(":")
    g = {"dep": D(int(h) * 60 + int(m)), "d1": n[0], "v1": n[1], "b": n[2]}
    if len(n) > 3:
        g["d2"], g["v2"] = n[3], n[4]
    return g


def c_route(g, ov=None, skip=False):
    c = Ch(ov)
    dr = c.put(fn, g["d1"] / g["v1"])
    if "d2" in g:
        g2 = c.put(fn, g["d2"] / g["v2"])
        dr = c.put(fn, dr + g2)
    mn = c.put(fn, dr * 60)
    al = c.put(fn, mn + g["b"])
    c.put(fc, g["dep"] + al)
    return c


def g_scale(r):
    n = nums(r)
    return {"n0": n[0], "qs": n[1:-1], "n": n[-1]}


def c_scale(g, ov=None, skip=False):
    c = Ch(ov)
    f = c.put(fn, g["n"] / g["n0"])
    a = [c.put(fn, q * f) for q in g["qs"]]
    tot = c.put(fn, sum(a))
    c.put(fn, tot / 1000)
    return c


KINDS = {"flow": (g_flow, c_flow), "invoice": (g_invoice, c_invoice), "units_trip": (g_units_trip, c_units_trip),
         "units_cargo": (g_units_cargo, c_units_cargo), "units_paint": (g_units_paint, c_units_paint), "units_time": (g_units_time, c_units_time),
         "units_dozen": (g_units_dozen, c_units_dozen), "dates_chain": (g_dates_chain, c_dates_chain), "dates_hire": (g_dates_hire, c_dates_hire),
         "dates_trip": (g_dates_trip, c_dates_trip), "pay_overtime": (g_pay_overtime, c_pay_overtime), "pay_stacked": (g_pay_stacked, c_pay_stacked),
         "pay_commission": (g_pay_commission, c_pay_commission), "pay_latefee": (g_pay_latefee, c_pay_latefee),
         "avg_readings": (g_avg_readings, c_avg_readings), "avg_cost": (g_avg_cost, c_avg_cost), "route": (g_route, c_route), "scale": (g_scale, c_scale)}
FAM = {f.__name__.lstrip("_"): f for f, _ in G.FAMILIES}


def derive(kind, request, ov=None, skip=False):
    gf, cf = KINDS[kind]
    return cf(gf(request), ov, skip)


def extra_allowed(kind, request):
    """Numbers a correct response may state that are neither in the request nor chain values: the divisor of a mean."""
    return {str(len(g_avg_readings(request)["kept"]))} if kind == "avg_readings" else set()


def is_correct(kind, request, response):
    """Independent verdict: every number of the response is a request number, a chain value or a known constant, and the
    chain values appear in order."""
    chain = derive(kind, request).strs()
    allowed = set(toks(request)) | set(chain) | extra_allowed(kind, request)
    ts = toks(response)
    if any(t not in allowed for t in ts):
        return False
    pos = 0
    for c in chain:
        try:
            pos = ts.index(c, pos) + 1
        except ValueError:
            return False
    return True


def parse_state(state):
    """The request and the response(s) of a stored row (JSON or labelled prose, any of the four key namings)."""
    try:
        obj = json.loads(state)
    except ValueError:
        obj = {}
        for block in re.split(r"\n\n(?=(?:Request|Task|Question|Prompt|Response|Answer|Worked answer|Reply|Rubric|Scoring rubric|Marking rubric)(?: [AB])?: )", state):
            k, v = block.split(": ", 1)
            obj[k.lower().replace(" ", "_")] = v
    req = next(v for k, v in obj.items() if k in ("request", "task", "question", "prompt"))
    rub = next((v for k, v in obj.items() if "rubric" in k), None)
    resp = [v for k, v in obj.items() if k not in ("request", "task", "question", "prompt") and "rubric" not in k]
    return req, resp, rub


# ---- tests: chains and single planted errors -------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_correct_and_corrupted_responses_judged_from_the_request_alone(kind):
    rng = random.Random("chain-" + kind)
    n_bad_types = Counter()
    for _ in range(120):
        w = G.worked(rng, fn=FAM[kind])
        p, bad = w["p"], w["bad"]
        assert p.kind == kind
        assert is_correct(kind, p.request, w["good_text"]), (p.request, w["good_text"])
        assert not is_correct(kind, p.request, w["bad_text"]), (p.request, w["bad_text"], bad["why"])
        assert len(w["good_text"]) == len(w["bad_text"])
        n_bad_types[bad["type"]] += 1
    assert len(n_bad_types) >= 5, n_bad_types


def _results(sents):
    return [toks(s)[-1] for s in sents]


@pytest.mark.parametrize("typ", sorted(G.ERR_W))
def test_exactly_one_step_is_wrong(typ):
    """The first changed step is the planted one; later steps either continue from the right value (isolated errors) or
    from the wrong one (carried errors), which is checked by recomputing the chain with that step overridden."""
    rng = random.Random("one-" + typ)
    seen = 0
    for kind in sorted(KINDS):
        for _ in range(30):
            w = G.worked(rng, types=(typ,), fn=FAM[kind])
            if w is None:
                break
            seen += 1
            p, bad = w["p"], w["bad"]
            good_s, bad_s = w["good"][0], bad["sents"]
            chain = derive(kind, p.request)
            gold = chain.strs()
            assert _results(good_s) == gold
            assert (w["good"][1] is None) == (bad["fin"] is None)
            res = _results(bad_s)
            diff = [i for i, (a, b) in enumerate(zip(good_s, bad_s)) if a != b]
            fin_diff = bad["fin"] != w["good"][1]
            if typ == "final":
                assert not diff and fin_diff and toks(bad["fin"])[-1] != gold[-1]
                continue
            assert diff and diff[0] == bad["k"]
            k = bad["k"]
            if typ in ("slip_iso", "copy_iso"):
                assert diff == [k] and not fin_diff
                assert res[:k] == gold[:k] and res[k + 1:] == gold[k + 1:]
                if typ == "slip_iso":
                    assert res[k] != gold[k] and sum(a != b for a, b in zip(res[k], gold[k])) == 1
                else:
                    assert res[k] == gold[k]  # only an operand was miscopied
                continue
            assert res[:k] == gold[:k] and res[k] != gold[k]
            # later results follow from the wrong value
            gf, cf = KINDS[kind]
            g = gf(p.request)
            wrong = chain.v[k].__class__  # date / Decimal
            val = datetime.strptime(res[k], "%d %b %Y").date() if wrong.__name__ == "date" else (
                D(int(res[k][:2]) * 60 + int(res[k][3:])) if CLOCK.fullmatch(res[k]) else D(res[k].replace(",", "")))
            assert res == cf(g, {k: val}).strs(), (typ, kind, p.request, bad_s)
            if bad["fin"]:
                assert toks(bad["fin"])[-1] == res[-1]
            if typ == "slip":
                assert sum(a != b for a, b in zip(res[k], gold[k])) == 1
            elif typ == "exception":
                assert res[k] == cf(g, None, skip=True).strs()[k]
            elif typ == "offby1":
                a, b = chain.v[k], val
                assert abs((b - a).days if wrong.__name__ == "date" else int(b - a)) == 1
            elif typ == "operand":  # the wrong value is the step computed with one operand (a given or an earlier step)
                keys = [x for x in g if isinstance(g[x], D)]  # replaced by another number of the request or of the chain
                vals = [g[x] for x in keys] + [v for v in chain.v[:k] if isinstance(v, D)]
                hits = [cf(dict(g, **{a: b})).strs()[k] for a in keys for b in vals if g[a] != b]
                hits += [cf(g, {j: b}).strs()[k] for j in range(k) if isinstance(chain.v[j], D) for b in vals if chain.v[j] != b]
                assert res[k] in hits, (kind, bad["why"])
            elif typ == "unit":
                assert chain.v[k] != val and res[k] != gold[k]
    assert seen >= 60, (typ, seen)  # every type is admitted by at least three families


def test_rationales_are_short_and_name_the_wrong_step():
    rng = random.Random("why")
    for _ in range(300):
        w = G.worked(rng)
        why = w["bad"]["why"]
        assert 1 <= len(re.findall(r"[.!?](?:\s|$)", why)) <= 3 and len(why) < 700 and len(why.split()) <= 110
        assert " not " in why or "instead" in why or "but the stated" in why
        assert len(G._all_good(w["p"]).split()) <= 110


# ---- tests: no shortcuts -----------------------------------------------------------------------------------------------------
def test_step_check_rows_balanced_equal_length_same_digit_count():
    rows = []
    for i, row in enumerate(G.judge_step_check(None, 2400)):
        rows.append(row)
    labels = Counter(r[3] for r in rows)
    assert abs(labels[1] / len(rows) - 0.5) < 0.04, labels
    lens = {0: [], 1: []}
    for split, state, q, lab, target, extra in rows:
        req, resp, _ = parse_state(state)
        lens[lab].append(len(resp[0]))
        assert q["type"] == "noul" and extra["rationale"] and extra["kind"].split(":")[0] in KINDS
        kind = extra["kind"].split(":")[0]
        assert is_correct(kind, req, resp[0]) == bool(lab)
    m0, m1 = sum(lens[0]) / len(lens[0]), sum(lens[1]) / len(lens[1])
    assert abs(m0 - m1) / m1 < 0.01, (m0, m1)
    rng = random.Random("digits")
    for _ in range(400):  # every changed number keeps its digit count and its decimals
        w = G.worked(rng)
        for a, b in zip(w["good"][0] + [w["good"][1] or ""], w["bad"]["sents"] + [w["bad"]["fin"] or ""]):
            for ta, tb in zip(toks(a), toks(b)):
                if ta != tb:
                    assert len(ta) == len(tb) and sum(ch.isdigit() for ch in ta) == sum(ch.isdigit() for ch in tb)
                    if w["bad"]["type"] != "unit":  # a unit error may move the decimal point (3.905 kg vs 39.05 kg)
                        assert ("." in ta) == ("." in tb) and len(ta.split(".")[-1]) == len(tb.split(".")[-1])


def test_pair_rows_gold_position_and_length_at_chance():
    rows = list(G.judge_step_pair(None, 2000))
    gold = Counter(r[3] for r in rows)
    assert abs(gold[0] / len(rows) - 0.5) < 0.04, gold
    longer = 0.0
    for split, state, q, lab, target, extra in rows:
        req, resp, _ = parse_state(state)
        kind = extra["kind"].split(":")[0]
        assert [is_correct(kind, req, x) for x in resp] == [lab == 0, lab == 1]
        la, lb = len(resp[0]), len(resp[1])
        longer += 0.5 if la == lb else float((0 if la > lb else 1) == lab)
    assert abs(longer / len(rows) - 0.5) < 0.05, longer / len(rows)


# ---- tests: rubric ---------------------------------------------------------------------------------------------------------------
def rubric_level(text, num, miss, off):
    """The level the rubric text gives, parsed here: caps, deductions or a ladder."""
    if off:
        return 0
    if "cap" in text or "at most" in text:
        m = re.findall(r"caps the score at (\d)", text) or re.findall(r"at most (\d)", text)
        cf, cm, ce = map(int, m)
        caps = [4] + ([cf] if num == "final" else [cm] if num == "mid" else []) + ([ce] if miss else [])
        return max(0, min(caps) - (1 if "one further point" in text and len(caps) == 3 else 0))
    if "Take off" in text:
        x, y, z = map(int, re.findall(r"Take off (\d)", text))
    elif "Deductions" in text:
        x, y, z = map(int, re.findall(r"(\d) (?:if|per)", text))
    else:
        if "Level rules" in text:
            e1, e2, m, f = map(int, re.findall(r": (\d)\.", text)[1:5])
        else:
            f, m, e2, e1 = map(int, re.findall(r"Level (\d):", text)[1:5])
        return min([4] + ([f] if num == "final" else [m] if num == "mid" else []) + ([e1] if miss == 1 else [e2] if miss >= 2 else []))
    return max(0, 4 - (x if num == "final" else y if num == "mid" else 0) - z * miss)


def check_rubric_row(kind, request, response, rubric, label):
    core, extra = request.split("\n\n", 1)
    head, body = (response.rsplit("\n\n", 1) if "\n\n" in response else ("", response))
    chain = derive(kind, core).strs()
    ts = toks(body)
    allowed = set(toks(core)) | set(chain) | extra_allowed(kind, core)
    off = chain[0] not in ts and chain[-1] not in ts
    final_ok = ts[-1] == chain[-1]
    num = None if off else ("final" if not final_ok else ("mid" if any(t not in allowed for t in ts) else None))
    miss = 0
    m = re.search(r"(?:Address the answer to|Open by greeting) ([A-Z]\w+ [A-Z]\w+)", extra)
    if m:
        miss += m.group(1).split()[0] not in head
    m = re.search(r"reference ([A-Z]{2}-\d{5})", extra)
    if m:
        miss += m.group(1) not in head
    m = re.search(r"(?:Date the answer|Put the date) (\d{1,2} [A-Z][a-z]+ \d{4})", extra)
    if m:
        miss += m.group(1) not in head
    assert rubric_level(rubric, num, miss, off) == label, (num, miss, off, rubric, label)
    return num, miss, off


def test_rubric_levels_follow_from_rubric_text():
    rng = random.Random("rubric")
    seen = Counter()
    for i in range(400):
        it = G.rubric_item(rng, i % 5)
        kind = it["kind"].split(":")[0]
        num, miss, off = check_rubric_row(kind, it["request"], it["response"], it["rubric"], it["label"])
        seen[(num, miss, off)] += 1
        assert len(it["rationale"].split()) <= 130
    assert len(seen) >= 8, seen
    rows = list(G.judge_rubric_cap(None, 500))
    levels = Counter(r[3] for r in rows)
    assert all(abs(levels[l] / 500 - 0.2) < 0.03 for l in range(5)), levels
    for split, state, q, lab, target, extra in rows[:150]:
        req, resp, rub = parse_state(state)
        assert q["type"] == "score" and len(q["criteria"]) == 5
        check_rubric_row(extra["kind"].split(":")[0], req, resp[0], rub, lab)


# ---- tests: adequacy ---------------------------------------------------------------------------------------------------------------
def unmet_requirements(request, response):
    """Requirement types stated in the request that the response does not meet, each checked here from scratch."""
    reg = request[re.search(r"\n(?:Requirements|The message must meet)", request).start():]
    lines = response.split("\n")
    items = [l for l in lines if re.match(r"^(?:\d+\. |- |• )", l)]
    out, seen = [], []
    m = re.search(r"(?:Address the message to|Open with a greeting to) ([A-Z]\w+ [A-Z]\w+)", reg)
    if m:
        seen.append("addressee")
        greet = next(l for l in lines if re.match(r"^(?:Hi|Hello|Dear) .+,$", l))
        if m.group(1).split()[0] not in greet:
            out.append("addressee")
    m = re.search(r"(?:Quote|Cite|Include) \w+ ([A-Z]{2}-\d{5})", reg)
    if m:
        seen.append("id")
        if m.group(1) not in response:
            out.append("id")
    m = re.search(r"(?:Mention the .+? date, |State that the .+? is on )(\d{1,2} [A-Z][a-z]+ \d{4})", reg)
    if m:
        seen.append("date")
        if m.group(1) not in response:
            out.append("date")
    for pat, ok in ((r"List no more than (\d+)|Keep the list to (\d+) \w+ or fewer", lambda n, k: n <= k),
                    (r"List at least (\d+)|Give (\d+) or more", lambda n, k: n >= k),
                    (r"List exactly (\d+)|must have exactly (\d+)", lambda n, k: n == k)):
        m = re.search(pat, reg)
        if m:
            seen.append("items")
            if not ok(len(items), int(m.group(1) or m.group(2))):
                out.append("items")
    for pat, ok in ((r"at most (\d+) words|Do not go over (\d+) words", lambda n, k: n <= k),
                    (r"at least (\d+) words|run to (\d+) words or more", lambda n, k: n >= k)):
        m = re.search(pat, reg)
        if m:
            seen.append("words")
            if not ok(len(response.split()), int(m.group(1) or m.group(2))):
                out.append("words")
    if "numbered list" in reg:
        seen.append("numbered")
        if not all(re.match(r"^\d+\. ", l) for l in items):
            out.append("numbered")
    if "total of the three charges" in reg or "sum of the charges" in reg:
        seen.append("total")
        amounts = [D(a.replace(",", "")) for a in re.findall(r"\$([\d,]+\.\d\d)", request)]
        assert len(amounts) == 3
        if "$" + fm(sum(amounts)) not in response:
            out.append("total")
    m = re.search(r"(?:Sign off as|End with the sign-off) (.+?)\.", reg)
    if m:
        seen.append("signoff")
        if m.group(1) not in lines[-1]:
            out.append("signoff")
    if "'Subject:'" in reg:
        seen.append("subject")
        if not lines[0].startswith("Subject:"):
            out.append("subject")
    m = re.search(r"between (\d\d:\d\d) and (\d\d:\d\d)", reg)
    if m:
        seen.append("time")
        t1, t2 = (int(x[:2]) * 60 + int(x[3:]) for x in m.groups())
        found = [int(x[:2]) * 60 + int(x[3:]) for x in CLOCK.findall(response)]
        if len(found) != 1 or not t1 <= found[0] <= t2:
            out.append("time")
    m = re.search(r"Do not name the [\w ]+? \((.+?)\)\.|Leave out the name of the [\w ]+?, (.+?)\.", reg)
    if m:
        seen.append("forbid")
        if (m.group(1) or m.group(2)) in response:
            out.append("forbid")
    m = re.search(r'(?:word for word: |exact sentence )"(.+?)"', reg)
    if m:
        seen.append("sentence")
        if m.group(1) not in response:
            out.append("sentence")
    return seen, out


def test_adequacy_requirements_rechecked_one_by_one():
    rng = random.Random("adq")
    kinds = Counter()
    for i in range(500):
        n = 3 + i % 4
        viol = None if i % 3 == 0 else rng.randrange(n)
        it = G.adequacy_item(rng, n, viol)
        seen, out = unmet_requirements(it["request"], it["response"])
        assert sorted(seen) == sorted(it["types"]) and len(seen) == n, (seen, it["types"])
        assert out == ([it["bad"]] if it["bad"] else []), (out, it["bad"], it["request"], it["response"])
        kinds[it["bad"]] += 1
    assert len(kinds) == len(G.RTYPES) + 1, kinds
    rows = list(G.judge_adequacy(None, 1500))
    noul = [r for r in rows if r[2]["type"] == "noul"]
    choice = [r for r in rows if r[2]["type"] == "choice"]
    assert 0.4 < len(noul) / len(rows) < 0.6
    assert abs(sum(r[3] for r in noul) / len(noul) - 0.5) < 0.05
    allmet = 0
    for split, state, q, lab, target, extra in rows:
        req, resp, _ = parse_state(state)
        seen, out = unmet_requirements(req, resp[0])
        if q["type"] == "noul":
            assert lab == int(not out)
        else:
            opts = q["criteria"]
            assert len(opts) == len(seen) + 1
            gold = opts[lab]
            if gold in G.RLAB[None]:
                assert not out
                allmet += 1
            else:
                assert len(out) == 1 and gold in G.RLAB[out[0]], (gold, out)
    share = allmet / len(choice)
    chance = sum(1 / len(r[2]["criteria"]) for r in choice) / len(choice)
    assert abs(share - chance) < 0.05, (share, chance)


# ---- tests: built datasets (skipped when not built) ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["judge_step_check", "judge_step_pair", "judge_rubric_cap", "judge_adequacy"])
def test_built_dataset_labels_rederived(name):
    path = os.path.join(ROOT, name, "val.jsonl")
    if not os.path.exists(path):
        pytest.skip("dataset not built")
    rows = [json.loads(l) for l in open(path)]
    assert len(rows) >= 500
    for r in rows[:400]:
        req, resp, rub = parse_state(r["state_text"])
        kind = r["kind"].split(":")[0]
        assert r["rationale"]
        if name == "judge_step_check":
            assert is_correct(kind, req, resp[0]) == bool(r["label"])
        elif name == "judge_step_pair":
            assert [is_correct(kind, req, x) for x in resp] == [r["label"] == 0, r["label"] == 1]
        elif name == "judge_rubric_cap":
            check_rubric_row(kind, req, resp[0], rub, r["label"])
        else:
            seen, out = unmet_requirements(req, resp[0])
            q = r["question"]
            if q["type"] == "noul":
                assert r["label"] == int(not out)
            else:
                gold = q["criteria"][r["label"]]
                assert (not out) if gold in G.RLAB[None] else (len(out) == 1 and gold in G.RLAB[out[0]])


def test_no_jevbench_ngram_overlap():
    clone = os.environ.get("JEVBENCH_CLONE")
    if not clone or not os.path.isdir(clone):
        pytest.skip("set JEVBENCH_CLONE to the JevBench checkout")
    from laya.visionjev_teacher import _ngrams, jevbench_ngrams

    bad = jevbench_ngrams(clone)
    for name in ["judge_step_check", "judge_step_pair", "judge_rubric_cap", "judge_adequacy"]:
        for split in ("train", "val"):
            path = os.path.join(ROOT, name, split + ".jsonl")
            if not os.path.exists(path):
                continue
            for line in open(path):
                assert not (_ngrams(line) & bad), (name, line[:200])
