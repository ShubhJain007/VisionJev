"""Round-2 exact-answer reasoning generators (Hypothesis/V6.2.md §2; PROBLEMS D2, E8). Same contract as
``visionjev_gen``: every answer is computed (``datetime`` / ``zoneinfo`` / ``Fraction`` / ``Decimal``), numeric and date
options are rank-uniform with named-mistake distractors first, entities are fictional. New here:

* every item carries a short worked **rationale** (record field ``rationale``) for the rationale loss;
* **withheld input**: ~20% of choice rows get the option "cannot be determined from the information given"; in a
  quarter of those one required input is replaced by "to be confirmed" and that option is the answer (the option also
  appears on answerable rows, so its presence is not a tell).

Families
* ``gen_units``       chargeable weight (lb → kg, round the TOTAL up), cold-chain log (°F readings vs a °C range,
                      inclusive bounds), card payment in a foreign currency (rate, % fee on the converted amount, flat fee).
* ``gen_rules``       refund eligibility under a general rule, an opened-electronics exception, a final-sale rule with a
                      defect carve-out and a holiday extension, with stated precedence: outcome, and which rule decided.
* ``gen_tables``      two small tables (customers, orders): filtered join + aggregate, and arg-max over a filtered join.
* ``gen_temporal2``   two-stage deadlines (deemed receipt of post, then N business days, closures); an on-time check
                      against a deadline in another time zone around the US / EU daylight-saving mismatch weeks.
* ``gen_longpolicy``  1.5–3.5k-token expense policies with the deciding exception buried mid-document and an amendment at
                      the end; outcome with the reason.

    python -m laya.visionjev_gen2 all
"""
import argparse
import calendar
import json
import math
import random
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from typing import Dict, Iterator, List, Optional
from zoneinfo import ZoneInfo

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT
from .visionjev_gen import ORGS, ZONES, _add_business, _d, _m, _num_choice, _split, _state

CBD = "cannot be determined from the information given"
TBC = "to be confirmed"
P_CBD, P_WITHHOLD = 0.20, 0.25


def _with_cbd(obj: dict, q: dict, lab: int, withhold: List[str], rng):
    """Maybe add the CBD option; maybe withhold one required input (a key, or ``outer.inner`` for a nested field) and
    make CBD the answer. Returns (obj, q, label, withheld key or None)."""
    if q.get("type") != "choice" or not withhold or rng.random() >= P_CBD:
        return obj, q, lab, None
    crit = list(q["criteria"])
    gold = crit[lab]
    held = None
    if rng.random() < P_WITHHOLD:
        held = rng.choice(withhold)
        obj = json.loads(json.dumps(obj, default=str))
        outer, _, inner = held.partition(".")
        if inner:
            obj[outer][inner] = TBC
        else:
            obj[outer] = TBC
        gold = CBD
    crit = crit + [CBD]
    rng.shuffle(crit)
    return obj, dict(q, criteria=crit), crit.index(gold), held


def _rows(kinds, tag: str, n_per_kind: int) -> Iterator[ts.Row]:
    for kname, fn in kinds:
        rng = random.Random("gen2-%s-%s" % (tag, kname))
        for i in range(n_per_kind):
            item = fn(rng, i)
            if item is None:
                continue
            obj, q, lab, why, withhold = item
            obj, q, lab, held = _with_cbd(obj, q, lab, withhold, rng)
            if held:
                why = "A required input (%s) is still to be confirmed, so the answer cannot be determined." % \
                      held.split(".")[-1].replace("_", " ")
            yield (_split("gen2-%s-%s-%d" % (tag, kname, i)), _state(obj, rng), q, lab, None,
                   {"rationale": why, "kind": kname})


def _date_choice(ins: str, gold: date, mistakes: List[date], rng, weekdays_only: bool = True):
    """Rank-uniform date options; near-misses are +-1..3 BUSINESS days so no option is an obvious weekend."""
    near = [(_add_business(gold, k, set()) if k > 0 else _sub_business(gold, -k)) for k in (1, 2, 3, -1, -2, -3)]
    ms = [m for m in mistakes if not weekdays_only or m.weekday() < 5]
    return _num_choice(ins, gold.toordinal(), [m.toordinal() for m in ms] + [n.toordinal() for n in near],
                       lambda o: _d(date.fromordinal(int(o))), rng, [])


def _sub_business(d: date, n: int) -> date:
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


# ---- units ------------------------------------------------------------------------------------------------------------
LB = Fraction("0.45359237")


def _weight(rng, i):
    items = []
    for j in range(rng.randint(3, 6)):
        if rng.random() < 0.7:
            items.append(("carton %d" % (j + 1), Fraction(rng.randint(20, 900), 10), "lb"))
        else:
            items.append(("carton %d" % (j + 1), Fraction(rng.randint(10, 400), 10), "kg"))
    kg = lambda it: it[1] * LB if it[2] == "lb" else it[1]  # noqa: E731
    total = sum(kg(it) for it in items)
    gold = math.ceil(total)
    mistakes = [sum(math.ceil(kg(it)) for it in items),  # rounded each carton up
                math.ceil(sum((it[1] * Fraction(45, 100) if it[2] == "lb" else it[1]) for it in items)),  # 0.45
                int((total + Fraction(1, 2)) // 1),  # nearest, not up
                math.ceil(sum((it[1] if it[2] == "lb" else it[1]) for it in items)),  # kg entries treated as lb... no conversion
                math.ceil(sum((it[1] / LB if it[2] == "lb" else it[1]) for it in items))]  # divided instead of multiplied
    q = _num_choice("What is the chargeable weight of this shipment?", gold, mistakes, lambda v: "%d kg" % int(v), rng,
                    [-3, -2, -1, 1, 2, 3], lo=1)
    if q is None:
        return None
    obj = {"carrier_rule": "Convert every weight to kilograms (1 lb = 0.45359237 kg), add them up, then round the TOTAL up "
           "to the next whole kilogram. That is the chargeable weight.",
           "cartons": [{"id": n, "weight": "%s %s" % (float(w), u)} for n, w, u in items]}
    why = "Sum in kg = %.3f kg (lb × 0.45359237, kg as given); rounded up once at the end → %d kg." % (float(total), gold)
    return obj, q[0], q[1], why, ["cartons"]


def _coldchain(rng, i):
    lo_c, hi_c = rng.choice([(2, 8), (0, 4), (15, 25), (-25, -15)])
    to_f = lambda c: Fraction(c) * 9 / 5 + 32  # noqa: E731
    readings = []
    for _ in range(rng.randint(8, 14)):
        r = rng.random()
        if r < 0.12:
            f = to_f(rng.choice([lo_c, hi_c]))  # exactly on a bound (inclusive: compliant)
        elif r < 0.35:
            f = to_f(rng.choice([lo_c - Fraction(rng.randint(1, 30), 10), hi_c + Fraction(rng.randint(1, 30), 10)]))
        else:
            f = to_f(lo_c + (hi_c - lo_c) * Fraction(rng.randint(5, 95), 100))
        readings.append(Fraction(round(float(f), 1)).limit_denominator(10))
    c = lambda f: (f - 32) * 5 / 9  # noqa: E731
    out = lambda conv, incl=True: sum(1 for f in readings if not (  # noqa: E731
        (lo_c <= conv(f) <= hi_c) if incl else (lo_c < conv(f) < hi_c)))
    gold = out(c)
    mistakes = [out(c, incl=False), out(lambda f: f - 32), out(lambda f: f * 5 / 9), len(readings) - gold]
    q = _num_choice("How many readings violated the storage range?", gold, mistakes, lambda v: "%d" % int(v), rng,
                    [-2, -1, 1, 2], lo=0, hi=len(readings))
    if q is None:
        return None
    obj = {"storage_rule": "Keep between %d °C and %d °C inclusive; a reading exactly on a limit is compliant." % (lo_c, hi_c),
           "logger_readings_F": [float(f) for f in readings]}
    why = "Convert each reading with C = (F − 32) × 5/9 and count those outside [%d, %d] °C (bounds inclusive): %d." % (
        lo_c, hi_c, gold)
    return obj, q[0], q[1], why, ["storage_rule"]


def _fx(rng, i):
    cur, sym = rng.choice([("EUR", "€"), ("GBP", "£"), ("JPY", "¥"), ("CHF", "CHF ")])
    prices = [Decimal(rng.randint(500, 45000)) / (1 if cur == "JPY" else 100) for _ in range(rng.randint(1, 4))]
    rate = Decimal(rng.randint(60, 1400)) / (100000 if cur == "JPY" else 1000)  # USD per 1 unit
    pct = Decimal(rng.choice([1, 1.5, 2, 2.5, 3])) / 100
    flat = Decimal(rng.choice([0, 30, 50, 100])) / 100
    base = sum(prices)
    gold = base * rate * (1 + pct) + flat
    mistakes = [base / rate * (1 + pct) + flat, base * rate + flat, base * rate * (1 + pct),
                (base + flat) * rate * (1 + pct), base * rate * (1 - pct) + flat]
    q = _num_choice("How much is charged to the card in US dollars?", gold, mistakes, _m, rng,
                    [gold * Decimal(x) for x in ("-0.03", "-0.015", "0.015", "0.03")])
    if q is None:
        return None
    obj = {"purchases": ["%s%s" % (sym, p) for p in prices], "exchange_rate": "1 %s = %s USD" % (cur, rate),
           "card_terms": "Converted at the rate shown, then a %s%% foreign transaction fee on the converted amount, then a "
           "flat %s per transaction." % ((pct * 100).normalize(), _m(flat))}
    why = "%s%s × %s = %s; + %s%% fee; + %s flat → %s." % (sym, base, rate, _m(base * rate), (pct * 100).normalize(),
                                                          _m(flat), _m(gold))
    return obj, q[0], q[1], why, ["exchange_rate"]


# ---- rules ------------------------------------------------------------------------------------------------------------
RULES = ["general 30-day rule", "opened-electronics exception", "holiday extension", "final-sale rule",
         "defective-item rule"]


def _eligibility(rng, i):
    year = rng.choice([2025, 2026])
    hol_a, hol_b = date(year, 11, 15), date(year, 12, 31)
    hol_end = date(year + 1, 1, 31)
    cat = rng.choice(["electronics", "electronics", "clothing", "home goods", "toys"])
    opened = rng.random() < 0.5
    final = rng.random() < 0.2
    defective = rng.random() < 0.35
    order = hol_a + timedelta(days=rng.randint(-40, 60))
    delivery = order + timedelta(days=rng.randint(2, 9))
    # decide which rule applies and the deadline
    if final:
        rule = "defective-item rule" if defective else "final-sale rule"
        deadline = delivery + timedelta(days=60) if defective else None
    else:
        normal_days = 14 if (cat == "electronics" and opened) else 30
        normal = delivery + timedelta(days=normal_days)
        if hol_a <= order <= hol_b and hol_end > normal:
            rule, deadline = "holiday extension", hol_end
        else:
            rule = "opened-electronics exception" if normal_days == 14 else "general 30-day rule"
            deadline = normal
    if deadline is None:
        request = delivery + timedelta(days=rng.randint(1, 40))
        approved = False
    else:
        request = deadline + timedelta(days=rng.choice([-9, -4, -2, -1, 0, 0, 1, 1, 2, 5, 12]))
        approved = request <= deadline
    obj = {"store_policy": [
        "1. General rule: refunds are accepted within 30 days of delivery.",
        "2. Opened electronics may be refunded within 14 days of delivery; unopened electronics follow rule 1.",
        "3. Items marked final sale are not refundable, except defective items, which may be refunded within 60 days of "
        "delivery.",
        "4. Holiday extension: for orders placed from %s to %s, the refund window ends on %s or at the deadline under rules "
        "1-2, whichever is later." % (_d(hol_a), _d(hol_b), _d(hol_end)),
        "5. Precedence: rule 3 overrides rules 1, 2 and 4; rule 4 overrides rules 1 and 2."],
        "case": {"category": cat, "opened": opened, "final_sale": final, "reported_defective": defective,
                 "order_date": _d(order), "delivery_date": _d(delivery), "refund_requested": _d(request)}}
    if rng.random() < 0.55:
        q, lab = ts._fixed_choice("Should this refund be approved under the store policy?", ["approve", "deny"],
                                  0 if approved else 1, rng)
    else:
        opts = [rule] + rng.sample([r for r in RULES if r != rule], 3)
        q, lab = ts._fixed_choice("Which rule of the policy decides this refund?", opts, 0, rng)
    why = "%s applies%s; request %s → %s." % (rule.capitalize(), (", deadline %s" % _d(deadline)) if deadline else
                                             " (no refund)", _d(request), "approve" if approved else "deny")
    return obj, q, lab, why, ["case.refund_requested", "case.delivery_date", "case.order_date"]


# ---- tables -----------------------------------------------------------------------------------------------------------
REGIONS = ["North", "South", "East", "West"]
NAMES = ["Aldous Co", "Birchfield", "Corvid Ltd", "Dunmore", "Elstree", "Farrow & Co", "Gantry", "Holloway", "Ivel Group",
         "Jarrow", "Kestle", "Lindqvist AB"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]


def _tables(rng, i):
    cust = [{"customer_id": 100 + k, "name": n, "region": rng.choice(REGIONS)}
            for k, n in enumerate(rng.sample(NAMES, rng.randint(6, 9)))]
    orders = []
    for k in range(rng.randint(14, 24)):
        c = rng.choice(cust)
        orders.append({"order_id": 5000 + k, "customer_id": c["customer_id"], "month": rng.choice(MONTHS[:4]),
                       "status": rng.choices(["shipped", "cancelled", "returned"], [0.7, 0.15, 0.15])[0],
                       "amount": Decimal(rng.randint(2000, 90000)) / 100})
    region = {c["customer_id"]: c["region"] for c in cust}
    name = {c["customer_id"]: c["name"] for c in cust}
    R, M = rng.choice(REGIONS), rng.choice(MONTHS[:4])
    sel = lambda st=True, mo=True, rg=True: [o for o in orders if (not st or o["status"] == "shipped") and  # noqa: E731
                                             (not mo or o["month"] == M) and (not rg or region[o["customer_id"]] == R)]
    obj = {"customers": cust, "orders": [dict(o, amount=str(o["amount"])) for o in orders]}
    if rng.random() < 0.6:
        tot = lambda os: sum((o["amount"] for o in os), Decimal(0))  # noqa: E731
        gold = tot(sel())
        if len(sel()) < 2:  # one matching order makes the gold a verbatim amount from the table (audit: overlap 0.45)
            return None
        mistakes = [tot(sel(st=False)), tot(sel(mo=False)), tot(sel(rg=False)), tot(sel(st=False, mo=False))]
        q = _num_choice("What is the total amount of shipped %s orders from customers in the %s region?" % (M, R), gold,
                        mistakes, _m, rng, [gold * Decimal(x) for x in ("-0.12", "-0.06", "0.06", "0.12")], lo=Decimal(0))
        if q is None:
            return None
        why = "Join orders to customers, keep status = shipped, month = %s, region = %s: %d orders, total %s." % (
            M, R, len(sel()), _m(gold))
        return obj, q[0], q[1], why, ["orders"]
    by = lambda os: {cid: sum((o["amount"] for o in os if o["customer_id"] == cid), Decimal(0))  # noqa: E731
                     for cid in {o["customer_id"] for o in os}}
    best = lambda os: max(by(os).items(), key=lambda kv: (kv[1], -kv[0]))[0] if os else None  # noqa: E731
    g = best(sel(mo=False))
    if g is None:
        return None
    cands = [g] + [c for c in (best(sel(st=False, mo=False)), best(sel(mo=False, rg=False)), best(orders)) if c]
    opts = []
    for c in cands + rng.sample([c["customer_id"] for c in cust], len(cust)):
        if name[c] not in opts:
            opts.append(name[c])
    q, lab = ts._fixed_choice("Which customer in the %s region has the highest total of shipped orders?" % R, opts[:4], 0, rng)
    why = "Shipped totals by customer in %s: top is %s (%s)." % (R, name[g], _m(by(sel(mo=False))[g]))
    return obj, q, lab, why, ["customers"]


# ---- temporal compositions --------------------------------------------------------------------------------------------


def _deemed(rng, i):
    posted = date(2026, 1, 5) + timedelta(days=rng.randrange(0, 700))
    if posted.weekday() >= 5:
        posted -= timedelta(days=posted.weekday() - 4)
    dr, n = rng.choice([2, 2, 3]), rng.randint(5, 15)
    horizon = _add_business(posted, dr + n + 6, set())
    wk = [posted + timedelta(days=j) for j in range(1, (horizon - posted).days) if (posted + timedelta(days=j)).weekday() < 5]
    closed = set(rng.sample(wk, min(len(wk), rng.randint(1, 3))))
    deemed = _add_business(posted, dr, closed)
    gold = _add_business(deemed, n, closed)
    mistakes = [_add_business(posted, n, closed),  # skipped the deemed-receipt step
                _add_business(posted + timedelta(days=dr), n, closed),  # deemed receipt in calendar days
                _add_business(_add_business(posted, dr, set()), n, set()),  # closures ignored
                posted + timedelta(days=dr + n)]  # everything in calendar days
    q = _date_choice("On which date is the response due?", gold, mistakes, rng)
    if q is None:
        return None
    obj = {"rule": "A notice sent by post is deemed received %d business days after posting. The response is due %d "
           "business days after deemed receipt. Business days are Monday to Friday, excluding the closure days listed." % (
               dr, n), "notice_posted": _d(posted), "office_closures": [_d(c) for c in sorted(closed)]}
    why = "Deemed received %s (%d business days after posting, skipping closures); + %d business days → %s." % (
        _d(deemed), dr, n, _d(gold))
    return obj, q[0], q[1], why, ["notice_posted", "office_closures"]


def _tz_deadline(rng, i):
    (c1, z1), (c2, z2) = rng.sample([z for z in ZONES if z[0] in ("New York", "London", "Berlin", "Chicago", "Los Angeles",
                                                                  "Sydney", "Tokyo")], 2)
    day = rng.choice([date(2026, 3, rng.randint(6, 30)), date(2026, 10, rng.randint(23, 31)), date(2026, 11, rng.randint(1, 6)),
                      date(2026, 6, rng.randint(1, 28)), date(2026, 1, rng.randint(5, 28))])
    hh = rng.choice([12, 17, 18, 23])
    dl = datetime.combine(day, time(hh, 0 if hh != 23 else 59), tzinfo=ZoneInfo(z1))
    delta = timedelta(minutes=rng.choice([-95, -61, -60, -30, -5, -1, 1, 5, 30, 59, 60, 61, 95]))
    sub = (dl + delta).astimezone(ZoneInfo(z2))
    on_time = (dl + delta) <= dl
    obj = {"deadline": "%s at %s local time in %s" % (_d(day), dl.strftime("%H:%M"), c1),
           "submitted": "%s at %s local time in %s" % (_d(sub.date()), sub.strftime("%H:%M"), c2),
           "note": "Both times are local wall-clock times, including any daylight-saving time in force that day."}
    q = ts._noul("Was the submission on time?")
    diff = int((dl - (dl + delta)).total_seconds() // 60)
    why = "Converting both to UTC, the submission is %d minutes %s the deadline → %s." % (
        abs(diff), "before" if diff >= 0 else "after", "on time" if on_time else "late")
    return obj, q, int(on_time), why, []


# ---- long policies ----------------------------------------------------------------------------------------------------
FILLER = [
    "Scope. This policy applies to all employees, contractors engaged for more than three months, and board members "
    "travelling on {org} business. It does not apply to relocation, which is covered by the Relocation Policy.",
    "Approval. Trips must be approved in advance by the traveller's line manager. Trips above the regional budget also "
    "require approval from the budget holder, recorded in the travel system before booking.",
    "Booking channels. Flights, rail and hotels should be booked through the approved travel agency so that duty-of-care "
    "tracking works. Bookings made elsewhere are reimbursed only if the fare is no higher than the agency quote.",
    "Class of travel. Economy class is the default for flights under six hours. Premium economy may be booked for longer "
    "flights. Business class requires approval from a director and a stated reason.",
    "Hotels. Hotel rates should not exceed the city cap published by Finance. Where no cap is published, a mid-range "
    "business hotel is expected. Upgrades are at the traveller's own expense.",
    "Meals. Reasonable meal costs are reimbursed while travelling. Alcohol is reimbursed only when hosting external guests "
    "and must be itemised separately on the receipt.",
    "Ground transport. Public transport is preferred where practical. Taxis and ride-hailing are reimbursed for early or "
    "late journeys, when carrying equipment, or where public transport is unsafe.",
    "Mileage. Use of a private car is reimbursed at the published mileage rate for the shortest practical route. Parking "
    "and tolls are reimbursed with receipts; fines are never reimbursed.",
    "Currency. Expenses incurred in a foreign currency are converted at the rate on the card statement, or, for cash, at "
    "the rate on the day of the expense as published by Finance.",
    "Advances. Cash advances may be requested for trips of more than five days. Unspent advances must be returned within "
    "ten business days of the end of the trip.",
    "Corporate cards. Holders must not use the corporate card for personal expenses. Accidental personal charges must be "
    "reported and repaid within fourteen days.",
    "Insurance. Business travel insurance covers employees on approved trips. Personal belongings are covered only up to "
    "the limits in the insurance summary on the intranet.",
    "Safety. Travellers to destinations flagged by the security team must complete the pre-travel briefing and keep their "
    "itinerary updated in the travel system.",
    "Sustainability. Rail is encouraged for journeys under four hours. Travellers are asked to combine trips where "
    "possible and to consider video meetings first.",
    "Gifts and hospitality. Gifts received while travelling must be declared under the Gifts and Hospitality Policy and "
    "are not expense items.",
    "Loyalty schemes. Travellers may keep personal loyalty points, but loyalty membership must not influence the choice "
    "of airline or hotel when a cheaper option meets the policy.",
    "Records. Receipts and approvals must be kept for seven years. The travel system stores them automatically when they "
    "are uploaded with the claim.",
    "Audit. Finance samples claims every month. Claims found to breach this policy may be recovered from the claimant's "
    "next salary payment after discussion with the claimant.",
    "Data protection. Personal data collected for travel is processed under the Employee Privacy Notice and shared with "
    "travel suppliers only as needed to deliver the booking.",
    "Queries. Questions about this policy should be sent to the Finance helpdesk. Answers given by the helpdesk do not "
    "override the text of this policy.",
    "Conflicts. Where this policy conflicts with a local law or collective agreement, the local law or agreement "
    "prevails for employees in that country.",
    "Exceptions process. Exceptions not described in this policy require written approval from the Chief Financial "
    "Officer before the expense is incurred.",
    "Training. New employees receive a summary of this policy during induction. Managers are responsible for making sure "
    "their team follows it.",
    "Visas. Visa fees for business trips are reimbursed. Expedited processing is reimbursed only if the trip was approved "
    "at short notice.",
    "Communication costs. Roaming charges for business calls and data are reimbursed. Travellers should enable the "
    "company roaming plan before departure.",
    "Laundry. Laundry is reimbursed for trips longer than five consecutive nights.",
    "Personal travel. Personal days added to a business trip are at the traveller's own cost, including any extra hotel "
    "nights and fare differences.",
    "Cancellations. Cancellation fees are reimbursed when the trip is cancelled for business reasons or for reasons "
    "outside the traveller's control.",
    "Health. Vaccinations and medication required for a business destination are reimbursed on production of a receipt.",
    "Review. This policy is reviewed every year by Finance and approved by the Executive Committee.",
]


SUB = [
    "Line managers may ask for supporting documents before approving.", "Finance may update the published rates without notice.",
    "Where a supplier invoice is issued, the invoice replaces the receipt.", "Shared costs must be claimed by one person only.",
    "Tips are reimbursed up to the customary local amount.", "Costs for companions or family members are never reimbursed.",
    "Travellers are responsible for keeping their contact details current in the travel system.",
    "Claims must be made in the travel system; paper claims are no longer accepted.",
    "Where a booking is changed, the traveller must record the reason in the booking notes.",
    "Items bought for personal comfort, such as neck pillows or headphones, are personal expenses.",
    "Costs incurred on days that are not travel days are personal unless approved in advance.",
    "Finance may reject any claim that does not follow this section and will say why.",
    "Questions about how this section applies to a particular trip go to the Finance helpdesk.",
    "Employees working part-time follow the same rules as full-time employees.",
    "Where local practice differs, the stricter rule applies unless Finance agrees otherwise.",
    "Duplicate claims for the same expense are treated as a breach of this policy.",
    "Nothing in this section limits the rights of employees under local law.",
    "Travellers should keep booking confirmations until the claim has been paid.",
    "The travel agency's out-of-hours line may be used for urgent changes while travelling.",
    "Approvals given by email must be attached to the claim.",
    "Where the traveller is unsure whether a cost is covered, they should ask before incurring it.",
    "Contractors follow this section only where their contract says so.",
    "The Chief Financial Officer may issue guidance notes that explain this section.",
    "Costs that are refunded by a supplier must not be claimed.",
]


def _long_policy(rng, i):
    org = rng.choice(ORGS)
    window, intl_window = rng.choice([30, 45, 60]), rng.choice([60, 75, 90])
    thr_old, thr_new = rng.choice([(75, 50), (100, 60), (60, 40)])
    eff = date(2026, rng.randint(2, 11), rng.randint(1, 28))
    international = rng.random() < 0.5
    travel_end = eff + timedelta(days=rng.randint(-80, 40))
    w = intl_window if international else window
    deadline = travel_end + timedelta(days=w)
    submitted = deadline + timedelta(days=rng.choice([-20, -6, -1, 0, 1, 3, 10]))
    thr = thr_new if submitted >= eff else thr_old
    item_amount = Decimal(rng.choice([thr_new - 5, thr_new + 5, (thr_new + thr_old) // 2, thr_old + 5, thr_old - 5]))
    has_receipt = rng.random() < 0.35 or submitted > deadline  # a late claim has its receipt: one reason, not two
    if submitted > deadline:
        outcome, reason = "rejected: submitted late", "submitted after the %d-day window (%s)" % (w, _d(deadline))
    elif item_amount > thr and not has_receipt:
        outcome, reason = "rejected: receipt missing", "the %s item has no receipt and the threshold on the submission date " \
                                                       "is %s" % (_m(item_amount), _m(thr))
    else:
        outcome, reason = "approved", "within the %d-day window and the receipt rule (threshold %s) is met" % (w, _m(thr))
    sections = [s + " " + " ".join(rng.sample(SUB, rng.randint(2, 4))) for s in rng.sample(FILLER, rng.randint(24, 30))]
    rule = ("Claims. Expense claims must be submitted within %d calendar days of the last day of travel. Every item above "
            "%s must have an itemised receipt; items at or below that amount may be claimed without a receipt." % (
                window, _m(thr_old)))
    exception = ("International travel. For trips that include at least one night outside the traveller's home country, "
                 "the claim window in the Claims section is extended to %d calendar days." % intl_window)
    amendment = ("Amendment, effective %s. For claims submitted on or after this date, the receipt threshold in the Claims "
                 "section is %s instead of the amount stated there. Claims submitted earlier follow the original "
                 "threshold." % (_d(eff), _m(thr_new)))
    body = sections[:3] + [rule] + sections[3:3 + len(sections) // 2] + [exception] + sections[3 + len(sections) // 2:]
    doc = "%s — Travel and Expenses Policy\n\n" % org + "\n\n".join(
        "%d. %s" % (k + 1, s.format(org=org)) for k, s in enumerate(body)) + "\n\n" + amendment
    obj = {"policy": doc, "claim": {"trip": "international (3 nights abroad)" if international else "domestic",
                                    "last_day_of_travel": _d(travel_end), "submitted_on": _d(submitted),
                                    "largest_item": "%s, %s" % (_m(item_amount), "receipt attached" if has_receipt else
                                                                "no receipt")}}
    opts = ["approved", "rejected: submitted late", "rejected: receipt missing"]
    q, lab = ts._fixed_choice("What is the outcome of this expense claim under the policy?", opts, opts.index(outcome), rng)
    return obj, q, lab, "%s: %s." % (outcome.capitalize(), reason), ["claim.submitted_on", "claim.last_day_of_travel"]


def _gen(kinds, tag, n):
    return lambda rng: _rows(kinds, tag, n)


LIC = "generated (VisionJev, Apache-2.0); labels computed"
SOURCES: Dict[str, dict] = {
    "gen_units": {"fn": _gen([("weight", _weight), ("coldchain", _coldchain), ("fx", _fx)], "units", 5000),
                  "repo": "generated", "license": LIC, "use": "unit conversion, rounding, fees", "train_cap": 14000,
                  "val_cap": 1000},
    "gen_rules": {"fn": _gen([("eligibility", _eligibility)], "rules", 12000), "repo": "generated", "license": LIC,
                  "use": "rule precedence and exceptions", "train_cap": 11000, "val_cap": 1000},
    "gen_tables": {"fn": _gen([("tables", _tables)], "tables", 24000), "repo": "generated", "license": LIC,
                   "use": "multi-hop joins and aggregates", "train_cap": 11000, "val_cap": 1000},
    "gen_temporal2": {"fn": _gen([("deemed", _deemed), ("tz_deadline", _tz_deadline)], "temporal2", 12000),
                      "repo": "generated", "license": LIC, "use": "two-stage deadlines, time zones / DST",
                      "train_cap": 22000, "val_cap": 1500},
    "gen_longpolicy": {"fn": _gen([("longpolicy", _long_policy)], "longpolicy", 9000), "repo": "generated",
                       "license": LIC, "use": "long-document policy reasoning (1.5-3.5k tokens)", "train_cap": 8000,
                       "val_cap": 800},
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (list(SOURCES) if a.names == ["all"] else a.names):
        ts.build(n, a.root, spec=SOURCES[n], log=lambda m: print(m, flush=True))
