"""Round-2 generators: every answer is recomputed here independently of ``laya.visionjev_gen2`` (numpy business-day
calendar, explicit UTC offsets, re-implemented rules), and the structural guarantees (rank-uniform options, withheld
inputs) are checked on samples."""
import json
import math
import random
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from zoneinfo import ZoneInfo

import numpy as np

from laya import visionjev_gen2 as G

D = "%a %d %b %Y"


def _date(s):
    return datetime.strptime(s, D).date()


def _gold(item):
    obj, q, lab, why, _ = item
    return q["criteria"][lab] if q.get("criteria") else lab


def _items(fn, n=300, seed=0):
    rng, out = random.Random(seed), []
    for i in range(n):
        it = fn(rng, i)
        if it is not None:
            out.append(it)
    return out


def test_deemed_receipt_matches_numpy_business_days():
    for obj, q, lab, why, _ in _items(G._deemed):
        dr, n = map(int, re.findall(r"(\d+) business days", obj["rule"]))
        hol = [np.datetime64(_date(c)) for c in obj["office_closures"]]
        posted = np.datetime64(_date(obj["notice_posted"]))
        deemed = np.busday_offset(posted, dr, roll="forward", holidays=hol)
        due = np.busday_offset(deemed, n, roll="forward", holidays=hol)
        assert q["criteria"][lab] == due.astype(object).strftime(D)


def test_tz_deadline_matches_explicit_offsets():
    for obj, q, lab, why, _ in _items(G._tz_deadline):
        city = {c: z for c, z in G.ZONES}
        dday, dtime, dcity = re.match(r"(.+) at (\d\d:\d\d) local time in (.+)", obj["deadline"]).groups()
        sday, stime, scity = re.match(r"(.+) at (\d\d:\d\d) local time in (.+)", obj["submitted"]).groups()
        d_loc = datetime.strptime(dday + " " + dtime, D + " %H:%M")
        s_loc = datetime.strptime(sday + " " + stime, D + " %H:%M")
        d_utc = d_loc - ZoneInfo(city[dcity]).utcoffset(d_loc)
        s_utc = s_loc - ZoneInfo(city[scity]).utcoffset(s_loc)
        assert lab == int(s_utc <= d_utc)


def test_weight_recomputed():
    for obj, q, lab, why, _ in _items(G._weight):
        total = Decimal(0)
        for c in obj["cartons"]:
            v, unit = c["weight"].split()
            total += Decimal(v) * (Decimal("0.45359237") if unit == "lb" else 1)
        assert q["criteria"][lab] == "%d kg" % math.ceil(total)


def test_coldchain_recomputed():
    for obj, q, lab, why, _ in _items(G._coldchain):
        lo, hi = map(int, re.findall(r"(-?\d+) °C", obj["storage_rule"]))
        bad = sum(1 for f in obj["logger_readings_F"] if not lo <= (Fraction(str(f)) - 32) * 5 / 9 <= hi)
        assert q["criteria"][lab] == str(bad)


def test_fx_recomputed():
    for obj, q, lab, why, _ in _items(G._fx):
        base = sum(Decimal(re.sub(r"[^0-9.]", "", p)) for p in obj["purchases"])
        rate = Decimal(re.search(r"= ([0-9.]+) USD", obj["exchange_rate"]).group(1))
        pct = Decimal(re.search(r"then a ([0-9.]+)%", obj["card_terms"]).group(1)) / 100
        flat = Decimal(re.search(r"flat \$([0-9.]+)", obj["card_terms"]).group(1))
        want = (base * rate * (1 + pct) + flat).quantize(Decimal("0.01"))
        assert Decimal(q["criteria"][lab].replace("$", "").replace(",", "")) == want


def test_eligibility_recomputed():
    for obj, q, lab, why, _ in _items(G._eligibility):
        c = obj["case"]
        rule4 = obj["store_policy"][3]
        a, b, end = [_date(x) for x in re.findall(r"(\w{3} \d\d \w{3} \d{4})", rule4)]
        order, delivery, req = _date(c["order_date"]), _date(c["delivery_date"]), _date(c["refund_requested"])
        if c["final_sale"]:
            rule = "defective-item rule" if c["reported_defective"] else "final-sale rule"
            ok = c["reported_defective"] and (req - delivery).days <= 60
        else:
            days = 14 if (c["category"] == "electronics" and c["opened"]) else 30
            normal = delivery.toordinal() + days
            if a <= order <= b and end.toordinal() > normal:
                rule, deadline = "holiday extension", end.toordinal()
            else:
                rule, deadline = ("opened-electronics exception" if days == 14 else "general 30-day rule"), normal
            ok = req.toordinal() <= deadline
        got = q["criteria"][lab]
        assert got == (("approve" if ok else "deny") if got in ("approve", "deny") else rule)


def test_long_policy_recomputed():
    for obj, q, lab, why, _ in _items(G._long_policy, 150):
        doc, c = obj["policy"], obj["claim"]
        window = int(re.search(r"submitted within (\d+) calendar days", doc).group(1))
        intl = int(re.search(r"extended to (\d+) calendar days", doc).group(1))
        thr_old = Decimal(re.search(r"Every item above \$([0-9.]+)", doc).group(1))
        eff, thr_new = re.search(r"Amendment, effective (.+?)\. .*?threshold in the Claims section is \$([0-9.]+)", doc).groups()
        end, sub = _date(c["last_day_of_travel"]), _date(c["submitted_on"])
        w = intl if c["trip"].startswith("international") else window
        amount = Decimal(re.search(r"\$([0-9.,]+)", c["largest_item"]).group(1).replace(",", ""))
        thr = Decimal(thr_new) if sub >= _date(eff) else thr_old
        if (sub - end).days > w:
            want = "rejected: submitted late"
        elif amount > thr and "no receipt" in c["largest_item"]:
            want = "rejected: receipt missing"
        else:
            want = "approved"
        assert q["criteria"][lab] == want


def test_numeric_options_are_rank_uniform():
    for fn in (G._deemed, G._fx, G._weight):
        ranks, n = Counter(), 0
        for obj, q, lab, why, _ in _items(fn, 1200, seed=1):
            vals = [o for o in q["criteria"]]
            if fn is G._deemed:
                key = [_date(v).toordinal() for v in vals]
            else:
                key = [float(re.sub(r"[^0-9.]", "", v)) for v in vals]
            if len(key) == 4:
                ranks[sorted(key).index(key[lab])] += 1
                n += 1
        assert all(0.17 < ranks[r] / n < 0.33 for r in range(4)), (fn.__name__, ranks)


def test_withheld_inputs_make_cbd_the_answer_and_only_then():
    rows = list(G._rows([("deemed", G._deemed), ("eligibility", G._eligibility), ("longpolicy", G._long_policy)], "t", 400))
    held = with_cbd = 0
    for split, state, q, lab, target, extra in rows:
        crit = q.get("criteria") or []
        if G.CBD in crit:
            with_cbd += 1
        is_held = G.TBC in state
        assert (crit[lab] == G.CBD) == is_held if crit else not is_held
        held += is_held
        assert extra["rationale"]
    assert 0.12 < with_cbd / len(rows) < 0.28 and 0.02 < held / len(rows) < 0.09
