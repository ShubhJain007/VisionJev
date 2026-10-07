"""Exact-answer reasoning generators (Hypothesis/V6.1.md §4.3; PROBLEMS D2). No model in the loop: every answer is
computed (``datetime`` / ``zoneinfo`` / ``Fraction`` / ``Decimal``), and every wrong option is the answer under one named
mistake, so a distractor can only be rejected by not making that mistake. Entities are fictional.

* ``gen_temporal``    business-day deadlines with closures; flight / shift durations across time zones and DST;
                      "N months later" with end-of-month roll-back; minimum-rest checks across a clock change (noul).
* ``gen_money``       invoice totals (discount, tax after discount, untaxed delivery); stacked discounts; proration
                      of an annual fee by active days (leap years, inclusive dates).
* ``gen_probability`` draws without replacement; conditional probability from a 2x2 table; at least one of two
                      independent events.

    python -m laya.visionjev_gen all
"""
import argparse
import hashlib
import json
import random
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from typing import Dict, Iterator, List
from zoneinfo import ZoneInfo

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT

ORGS = ["Harbor Logistics", "Kestrel Foods", "Northwind Clinics", "Brightline Appliances", "Tamsin Legal", "Orchid Telecom",
        "Pelham Freight", "Juniper Insurance", "Calder Energy", "Maple Street Schools", "Vireo Software", "Ostrava Metals"]
PEOPLE = ["A. Moreau", "J. Okafor", "L. Brandt", "S. Iyer", "M. Kowalski", "R. Haddad", "T. Nakamura", "E. Lindqvist"]
ZONES = [("New York", "America/New_York"), ("London", "Europe/London"), ("Berlin", "Europe/Berlin"),
         ("Tokyo", "Asia/Tokyo"), ("Sydney", "Australia/Sydney"), ("Los Angeles", "America/Los_Angeles"),
         ("Dubai", "Asia/Dubai"), ("Sao Paulo", "America/Sao_Paulo"), ("Delhi", "Asia/Kolkata"),
         ("Chicago", "America/Chicago")]
N_PER_KIND = 4000


def _split(key: str) -> str:
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < 8 else "train"


def _choice(ins: str, gold: str, wrong: List[str], rng, k: int = 4):
    opts = [gold]
    for w in wrong:
        if w not in opts:
            opts.append(w)
    if len(opts) < 2:
        return None
    return ts._fixed_choice(ins, opts[:k], 0, rng)


def _num_choice(ins: str, gold: float, mistakes: List[float], fmt, rng, near: List[float], k: int = 4, lo=None, hi=None):
    """Gold + k-1 distractors with the gold's RANK among the options drawn uniformly (audit: fixed mistake directions put
    the gold always lowest / always third, a learnable shortcut). Named mistakes are used first on each side, then
    near-misses ``gold + d`` for ``d`` in ``near``; ``lo`` / ``hi`` clamp the range (probabilities)."""
    g = fmt(gold)
    below, above, seen = [], [], {g}
    pool = [(m, 0) for m in mistakes] + [(gold + d, 1) for d in near]
    rng.shuffle(pool)
    pool.sort(key=lambda t: t[1])  # mistakes first, then near-misses (each group shuffled)
    for v, _ in pool:
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            continue
        f = fmt(v)
        if f in seen:
            continue
        seen.add(f)
        (below if v < gold else above).append(f)
    n = min(k - 1, len(below) + len(above))
    feasible = [r for r in range(n + 1) if r <= len(below) and n - r <= len(above)]
    if n < 2 or not feasible:
        return None
    r = rng.choice(feasible)
    return ts._fixed_choice(ins, [g] + below[:r] + above[:n - r], 0, rng)


def _state(obj: dict, rng) -> str:
    """Half JSON, half prose-like lines (the same content)."""
    if rng.random() < 0.5:
        return json.dumps(obj, ensure_ascii=False)
    return "\n".join("%s: %s" % (k.replace("_", " "), v if not isinstance(v, (list, dict)) else json.dumps(v))
                     for k, v in obj.items())


def _d(x: date) -> str:
    return x.strftime("%a %d %b %Y")


# ---- temporal ---------------------------------------------------------------------------------------------------------


def _add_business(start: date, n: int, closed: set) -> date:
    d = start
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in closed:
            n -= 1
    return d


def _deadline(rng, i):
    start = date(2026, 1, 5) + timedelta(days=rng.randrange(0, 700))
    n = rng.randint(3, 15)
    naive_end = _add_business(start, n, set())
    span = [start + timedelta(days=j) for j in range(1, (naive_end - start).days + 1)]
    weekdays = [x for x in span if x.weekday() < 5]
    closed = set(rng.sample(weekdays, min(len(weekdays), rng.randint(1, 2))))
    gold = _add_business(start, n, closed)
    wrong = [_d(start + timedelta(days=n)),  # calendar days
             _d(naive_end),  # closures ignored
             _d(_add_business(start - timedelta(days=1), n, closed)),  # counts the notice day itself
             _d(gold + timedelta(days=1 if gold.weekday() < 4 else 3))]  # one business day too many
    org = rng.choice(ORGS)
    obj = {"company": org, "policy": "A written response is due %d business days after the notice is received. Business "
           "days are Monday to Friday, excluding the office closure days listed. The day of receipt does not count." % n,
           "notice_received": _d(start), "office_closures": [_d(c) for c in sorted(closed)]}
    return _state(obj, rng), _choice("On which date is the response due?", _d(gold), wrong, rng), (
        "Counting %d business days after %s (receipt day excluded), skipping weekends and the closures %s → %s." % (
            n, _d(start), ", ".join(_d(c) for c in sorted(closed)), _d(gold)))


def _flight(rng, i):
    (c1, z1), (c2, z2) = rng.sample(ZONES, 2)
    base = datetime(2026, rng.choice([3, 3, 10, 10, 11, 4, 6, 1]), rng.randint(1, 28), rng.randint(0, 23), rng.choice([0, 15, 30, 45]))
    dep = base.replace(tzinfo=ZoneInfo(z1))
    dur = timedelta(minutes=rng.randint(90, 16 * 60) // 5 * 5)
    arr = (dep + dur).astimezone(ZoneInfo(z2))
    true_min = int(dur.total_seconds() // 60)
    naive = int((arr.replace(tzinfo=None) - dep.replace(tzinfo=None)).total_seconds() // 60)  # clocks subtracted
    std = lambda z, t: (z.utcoffset(t) - (z.dst(t) or timedelta(0)))  # noqa: E731
    no_dst = true_min + int(((ZoneInfo(z1).dst(dep) or timedelta(0)) - (ZoneInfo(z2).dst(arr) or timedelta(0))).total_seconds() // 60)
    flipped = naive + 2 * (true_min - naive)  # offset applied with the wrong sign
    fmt = lambda m: "%d h %02d min" % (int(m) // 60, int(m) % 60)  # noqa: E731
    mistakes = [naive, flipped, no_dst, true_min - 60, true_min + 60]
    obj = {"flight": "%s%d" % (rng.choice(["KX", "VA", "LH", "NZ", "QR"]), rng.randint(100, 999)),
           "departs": "%s local time, %s" % (dep.strftime("%d %b %Y %H:%M"), c1),
           "arrives": "%s local time, %s" % (arr.strftime("%d %b %Y %H:%M"), c2),
           "note": "Both times are the local clock time at each airport, including any daylight-saving time in force."}
    _ = std
    return _state(obj, rng), _num_choice("What is the actual elapsed flight time?", true_min, mistakes, fmt, rng,
                                         [-120, -90, -30, 30, 90, 120, 180], lo=30), (
        "Departure %s UTC, arrival %s UTC (each local time minus its UTC offset that day) → %s." % (
            dep.astimezone(ZoneInfo("UTC")).strftime("%d %b %H:%M"), arr.astimezone(ZoneInfo("UTC")).strftime("%d %b %H:%M"),
            fmt(true_min)))


def _months_later(rng, i):
    y, m = rng.choice([2026, 2027, 2028]), rng.randint(1, 12)
    day = rng.choice([28, 29, 30, 31, 31, 30, 15])
    import calendar
    day = min(day, calendar.monthrange(y, m)[1])
    start = date(y, m, day)
    n = rng.choice([1, 3, 6, 12, 18, 24, 13, 7])
    ym = (y * 12 + m - 1) + n
    ty, tm = ym // 12, ym % 12 + 1
    last = calendar.monthrange(ty, tm)[1]
    gold = date(ty, tm, min(day, last))
    def shifted(k):  # the same rule applied with k months instead of n (off-by-one month counts)
        yk = (y * 12 + m - 1) + k
        return date(yk // 12, yk % 12 + 1, min(day, calendar.monthrange(yk // 12, yk % 12 + 1)[1]))
    wrong = [_d(shifted(n + 1)), _d(shifted(n - 1)), _d(start + timedelta(days=30 * n))]
    if day > last:  # the end-of-month rule is the point: rolling over into the next month is the typical mistake
        wrong.insert(0, _d(date(ty, tm, 1) + timedelta(days=day - 1)))
    obj = {"agreement": "%s service term" % rng.choice(ORGS), "start_date": _d(start),
           "term": "The term ends %d calendar month%s after the start date, on the day with the same number. If that month "
           "has no such day, the term ends on the last day of that month." % (n, "" if n == 1 else "s")}
    return _state(obj, rng), _choice("On which date does the term end?", _d(gold), wrong, rng), (
        "%d months after %s lands in %s; day %d %s → %s." % (
            n, _d(start), gold.strftime("%b %Y"), day, "exists" if gold.day == day else "does not exist, so the last day",
            _d(gold)))


def _rest(rng, i):
    city, z = rng.choice([c for c in ZONES if c[0] in ("London", "Berlin", "New York", "Chicago", "Sydney", "Los Angeles")])
    tz = ZoneInfo(z)
    # nights around the 2026 clock changes, and ordinary nights
    base = rng.choice([date(2026, 3, 28), date(2026, 3, 7), date(2026, 10, 24), date(2026, 10, 31), date(2026, 4, 4),
                       date(2026, 6, 12), date(2026, 9, 18)]) + timedelta(days=rng.choice([0, 0, 1]))
    end = datetime.combine(base, datetime.min.time()).replace(hour=rng.randint(19, 23), minute=rng.choice([0, 30]), tzinfo=tz)
    start = datetime.combine(base + timedelta(days=1), datetime.min.time()).replace(
        hour=rng.randint(5, 11), minute=rng.choice([0, 30]), tzinfo=tz)
    real = (start.astimezone(ZoneInfo("UTC")) - end.astimezone(ZoneInfo("UTC"))).total_seconds() / 3600
    minimum = rng.choice([10, 11, 11, 12])
    label = int(real >= minimum)
    obj = {"rule": "Between the end of one shift and the start of the next there must be at least %d consecutive hours of "
           "rest, measured in actual elapsed time (clock changes count)." % minimum, "location": city,
           "shift_ends": end.strftime("%a %d %b %Y %H:%M local time"),
           "next_shift_starts": start.strftime("%a %d %b %Y %H:%M local time")}
    return _state(obj, rng), (ts._noul("Does this roster meet the minimum rest rule?"), label), (
        "Real elapsed rest from %s to %s is %.1f h (clock changes included) %s the %d h minimum." % (
            end.strftime("%a %H:%M"), start.strftime("%a %H:%M"), real, "meets" if label else "is below", minimum))


# ---- money ------------------------------------------------------------------------------------------------------------


def _m(x) -> str:
    return "$" + format(Decimal(x).quantize(Decimal("0.01"), ROUND_HALF_UP), ",")


def _invoice(rng, i):
    lines = [(rng.choice(["pallet wrap", "label roll", "sensor kit", "filter pack", "cable set", "toner", "valve"]),
              rng.randint(1, 40), Decimal(rng.randint(150, 90000)) / 100) for _ in range(rng.randint(2, 5))]
    sub = sum(q * p for _, q, p in lines)
    disc, tax = Decimal(rng.choice([5, 8, 10, 12, 15, 20])) / 100, Decimal(rng.choice([5, 7, 8.25, 10, 13, 19, 20])) / 100
    deliv = Decimal(rng.randint(0, 12000)) / 100
    gold = sub * (1 - disc) * (1 + tax) + deliv
    mistakes = [sub * (1 - disc) + sub * tax + deliv,  # tax on the undiscounted subtotal
                (sub * (1 - disc) + deliv) * (1 + tax),  # delivery taxed
                sub * (1 + tax) + deliv,  # discount forgotten
                sub * (1 - disc) + deliv,  # tax forgotten
                sub * (1 - disc) * (1 + tax)]  # delivery forgotten
    obj = {"vendor": rng.choice(ORGS), "invoice_no": "INV-%05d" % rng.randint(0, 99999),
           "lines": [{"item": n, "qty": q, "unit_price": _m(p)} for n, q, p in lines],
           "discount": "%s%% off the line subtotal" % disc.normalize().__mul__(100).normalize(),
           "sales_tax": "%s%% on the discounted subtotal" % (tax * 100).normalize(),
           "delivery": "%s (not taxed)" % _m(deliv)}
    return _state(obj, rng), _num_choice("What is the invoice total?", gold, mistakes, _m, rng,
                                         [gold * Decimal(x) for x in ("-0.06", "-0.03", "0.03", "0.06")]), (
        "Subtotal %s; minus %s%% discount = %s; plus %s%% tax on that = %s; plus untaxed delivery %s → %s." % (
            _m(sub), (disc * 100).normalize(), _m(sub * (1 - disc)), (tax * 100).normalize(),
            _m(sub * (1 - disc) * (1 + tax)), _m(deliv), _m(gold)))


def _stacked(rng, i):
    price = Decimal(rng.randint(2000, 250000)) / 100
    d1, d2 = rng.choice([10, 15, 20, 25, 30]), rng.choice([5, 10, 15, 20])
    gold = price * (1 - Decimal(d1) / 100) * (1 - Decimal(d2) / 100)
    mistakes = [price * (1 - Decimal(d1 + d2) / 100), price * (1 - Decimal(d1) / 100), price * (1 - Decimal(d2) / 100),
                price * (1 - Decimal(d1) / 100) * (1 + Decimal(d2) / 100)]
    obj = {"item_list_price": _m(price), "promotion": "%d%% off, then an extra %d%% off the reduced price at checkout" % (d1, d2)}
    return _state(obj, rng), _num_choice("What does the customer pay?", gold, mistakes, _m, rng,
                                         [gold * Decimal(x) for x in ("-0.08", "-0.04", "0.04", "0.08")]), (
        "%s × (1 − %d%%) = %s, then × (1 − %d%%) on the reduced price = %s (discounts multiply, they do not add)." % (
            _m(price), d1, _m(price * (1 - Decimal(d1) / 100)), d2, _m(gold)))


def _prorate(rng, i):
    y = rng.choice([2026, 2027, 2028])
    ndays = 366 if y % 4 == 0 else 365
    a = date(y, 1, 1) + timedelta(days=rng.randrange(0, ndays - 20))
    b = a + timedelta(days=rng.randint(5, min(200, (date(y, 12, 31) - a).days)))
    fee = Decimal(rng.choice([365, 366, 730, 1098, 1200, 2400, 3650, 5000]))
    days = (b - a).days + 1
    gold = fee / ndays * days
    months = (b.year - a.year) * 12 + b.month - a.month + 1
    mistakes = [fee / ndays * (days - 1), fee / (365 if ndays == 366 else 366) * days, fee / 12 * months,
                fee / ndays * (days + 1)]
    obj = {"plan": "annual plan %s" % rng.choice(ORGS), "annual_fee": _m(fee), "year": y,
           "rule": "The fee is prorated by active calendar days in that year, counting both the start and the end date.",
           "active_from": _d(a), "active_to": _d(b)}
    return _state(obj, rng), _num_choice("What is the prorated fee?", gold, mistakes, _m, rng,
                                         [fee / ndays * d for d in (-3, -2, 2, 3)]), (
        "%d active days (%s to %s inclusive) of %d in %d: %s × %d / %d = %s." % (
            days, _d(a), _d(b), ndays, y, _m(fee), days, ndays, _m(gold)))


# ---- probability ------------------------------------------------------------------------------------------------------


def _f(x: Fraction) -> str:
    # decimals only: a fraction's size gave the gold away (audit: longest option 0.355 vs chance 0.25)
    return "%.3f" % float(x)


def _draws(rng, i):
    r, b = rng.randint(2, 9), rng.randint(2, 12)
    n = r + b
    gold = Fraction(r, n) * Fraction(r - 1, n - 1)
    mistakes = [Fraction(r, n) ** 2, Fraction(r, n), 1 - Fraction(b, n) * Fraction(b - 1, n - 1),
                Fraction(r, n) * Fraction(r - 1, n), Fraction(r - 1, n - 1), Fraction(r, n) * Fraction(r - 1, n - 1) * 2]
    item = rng.choice([("defective", "working", "units"), ("red", "blue", "tokens"), ("late", "on-time", "shipments")])
    obj = {"batch": "%d %s %s and %d %s %s, well mixed" % (r, item[0], item[2], b, item[1], item[2]),
           "procedure": "An inspector picks 2 %s at random, one after the other, without putting the first back." % item[2]}
    return _state(obj, rng), _num_choice("What is the probability that both picked %s are %s?" % (item[2], item[0]),
                                         gold, mistakes, _f, rng, [Fraction(d, 100) for d in (-6, -3, 3, 6)], lo=0, hi=1), (
        "Without replacement: %d/%d × %d/%d = %s." % (r, n, r - 1, n - 1, _f(gold)))


def _conditional(rng, i):
    a_b, a_nb, na_b, na_nb = [rng.randint(3, 90) for _ in range(4)]
    A, B = rng.choice([("flagged by the filter", "actually fraudulent"), ("tested positive", "has the condition"),
                       ("opened the email", "made a purchase")])
    tot = a_b + a_nb + na_b + na_nb
    gold = Fraction(a_b, a_b + na_b)  # P(A | B)
    mistakes = [Fraction(a_b, a_b + a_nb), Fraction(a_b, tot), Fraction(a_b + a_nb, tot), Fraction(a_b + na_b, tot),
                Fraction(na_b, a_b + na_b)]
    obj = {"counts": {"%s and %s" % (A, B): a_b, "%s, not %s" % (A, B): a_nb, "not %s, %s" % (A, B): na_b,
                      "neither": na_nb}, "total_cases": tot}
    return _state(obj, rng), _num_choice("Among the cases that were %s, what fraction were %s?" % (B, A), gold, mistakes,
                                         _f, rng, [Fraction(d, 100) for d in (-6, -3, 3, 6)], lo=0, hi=1), (
        "Condition on %s: %d cases, of which %d were also %s → %d / %d = %s." % (
            B, a_b + na_b, a_b, A, a_b, a_b + na_b, _f(gold)))


def _at_least_one(rng, i):
    p1, p2 = Fraction(rng.randint(5, 60), 100), Fraction(rng.randint(5, 60), 100)
    gold = 1 - (1 - p1) * (1 - p2)
    mistakes = [p1 + p2, p1 * p2, max(p1, p2), (1 - p1) * (1 - p2), p1 + p2 - p1 * p2 * 2]
    obj = {"component_A_failure_rate_per_year": float(p1), "component_B_failure_rate_per_year": float(p2),
           "note": "The two components fail independently of each other."}
    return _state(obj, rng), _num_choice("What is the probability that at least one of the two fails in a year?", gold,
                                         mistakes, _f, rng, [Fraction(d, 100) for d in (-6, -3, 3, 6)], lo=0, hi=1), (
        "1 − P(neither) = 1 − (1 − %s)(1 − %s) = %s." % (_f(p1), _f(p2), _f(gold)))


def _rows(kinds, seed_tag) -> Iterator[ts.Row]:
    for kname, fn in kinds:
        rng = random.Random(seed_tag + kname)
        for i in range(N_PER_KIND):
            state, qa, why = fn(rng, i)
            if qa is None:
                continue
            q, lab = qa
            yield _split("%s-%s-%d" % (seed_tag, kname, i)), state, q, lab, None, {"rationale": why, "kind": kname}


def gen_temporal(rng) -> Iterator[ts.Row]:
    return _rows([("deadline", _deadline), ("flight", _flight), ("months", _months_later), ("rest", _rest)], "t")


def gen_money(rng) -> Iterator[ts.Row]:
    return _rows([("invoice", _invoice), ("stacked", _stacked), ("prorate", _prorate)], "m")


def gen_probability(rng) -> Iterator[ts.Row]:
    return _rows([("draws", _draws), ("conditional", _conditional), ("atleastone", _at_least_one)], "p")


LIC = "generated (VisionJev, Apache-2.0); labels computed"
SOURCES: Dict[str, dict] = {
    "gen_temporal": {"fn": gen_temporal, "repo": "generated", "license": LIC, "use": "dates, time zones, DST, deadlines",
                     "train_cap": 20000, "val_cap": 1500},
    "gen_money": {"fn": gen_money, "repo": "generated", "license": LIC, "use": "invoice / discount / proration arithmetic",
                  "train_cap": 15000, "val_cap": 1000},
    "gen_probability": {"fn": gen_probability, "repo": "generated", "license": LIC, "use": "exact probability",
                        "train_cap": 15000, "val_cap": 1000},
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (list(SOURCES) if a.names == ["all"] else a.names):
        ts.build(n, a.root, spec=SOURCES[n], log=lambda m: print(m, flush=True))
