"""Round-3 compound exact-answer generators (Hypothesis/V6.3.md §3.1). Same contract as ``visionjev_gen2``: every
answer is computed (``datetime`` / ``zoneinfo`` / ``Fraction`` / ``Decimal``), numeric and date options are rank-uniform,
every wrong option is the result of ONE named wrong step, entities are fictional. New here:

* each item chains 3-4 steps, and the **rationale carries the intermediate values** of every step (for the rationale
  loss and the think-if-unsure notes), ending with the answer;
* train and val are generated from different seeds AND different entity pools, and a scenario seen in one split is
  dropped from the other;
* the option "cannot be determined from the information given" is a wrong option on ~20% of rows and the answer on ~5%
  (one required input replaced by "to be confirmed").

Families
* ``gen_deadline_chain``   N months later with the month-end rule x leap years x business-day roll x "by end of day" /
                           "within N hours" x two time zones (DST included): in time? / last day / deadline elsewhere.
* ``gen_period_count``     business-day counts: inclusive / exclusive ends x closures x half days x "received after the
                           cut-off counts as the next business day"; storage days after a free period.
* ``gen_fx_aggregate``     alias -> vendor id, invoices in two currencies with dated rates, a rolling window, a
                           look-alike vendor that must not be added, an approval-tier table: tier / aggregated amount.
* ``gen_lookup_chain``     employee -> cost centre -> policy class -> limit (-> regional uplift) with a superseded row.
* ``gen_cond_probability`` two-stage events, Bayes, "at least one", expected value with a fee; prose, mixed formats.
* ``gen_constraint_rank``  eliminate on hard constraints (one fact hidden in a handling note that overrides the
                           summary), then rank on a priority order with a tie-break.
* ``gen_amended_policy``   2-3.5k-token policies: base clause, amendment with an effective date, an exception, and a
                           prominent summary that contradicts the authoritative clause.

    python -m laya.visionjev_gen3 all
"""
import argparse
import calendar
import copy
import hashlib
import json
import os
import random
import re
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from typing import Dict, Iterator, List
from zoneinfo import ZoneInfo

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT
from .visionjev_gen import _d, _num_choice
from .visionjev_gen2 import CBD, TBC, _date_choice

UTC = ZoneInfo("UTC")
P_CBD_WRONG, P_CBD_GOLD = 0.20, 0.05  # shares of ALL rows of a family (choice rows carry them)
DFMTS = ["%a %d %b %Y", "%d %B %Y", "%Y-%m-%d"]  # date formats used in states (options always use ``_d``)
DT = "%a %d %b %Y %H:%M"
DTS = "%d %b %H:%M"  # compact form for the UTC steps of a rationale

# entity pools: train and val never share an organisation, a person or a vendor stem
POOLS = {
    "train": {
        "orgs": ["Aldgate Marine", "Bexmoor Dairy", "Corrin Optics", "Dovecote Rail", "Eskdale Pharma", "Fennick Tools",
                 "Garrow Textiles", "Hollin Aerospace", "Inchmarlo Foods", "Jessop Water", "Kirkby Robotics",
                 "Lomond Paper", "Marram Hotels", "Netherby Mining", "Oakhanger Media", "Pendle Biotech",
                 "Quarrendon Glass", "Rushmere Power"],
        "people": ["Abara", "Bergstrom", "Castellano", "Dlamini", "Eriksen", "Fofanah", "Grigoryan", "Halvorsen",
                   "Ibekwe", "Jovanovic", "Kaplan", "Lemaire", "Mbeki", "Novak", "Oyelaran", "Petrenko", "Quintero",
                   "Rasmussen", "Sorensen", "Takahashi", "Umarov", "Valdivia", "Wojcik", "Yilmaz"],
        "stems": ["Norvane", "Halberd", "Syncora", "Tessel", "Brackwell", "Ormeau", "Lindholt", "Caspar", "Wexcombe",
                  "Durnford", "Pryce", "Astrela", "Mondrake", "Velten", "Kestrow", "Rivelin"],
    },
    "val": {
        "orgs": ["Thornaby Cement", "Ullesby Freight", "Varley Instruments", "Winterbourne Seeds", "Yarrow Shipping",
                 "Zennor Analytics", "Ashcombe Dental", "Barrowby Timber"],
        "people": ["Adeyemi", "Blomqvist", "Carvalho", "Demirci", "Esposito", "Fitzwilliam", "Gutierrez", "Horvath",
                   "Iwasaki", "Jablonski", "Kowalczuk", "Lindahl"],
        "stems": ["Garvane", "Steenrod", "Polwarth", "Minterne", "Oxenhope", "Trevarno", "Bellancourt", "Hadleigh"],
    },
}
INITIALS = "ABCDEFGHJKLMNPRSTVW"
SUFFIXES = ["AG", "GmbH", "Ltd", "SA", "BV", "Inc", "LLC", "AB", "Pty", "SpA"]


def _m2(x, cur="$") -> str:
    x = Decimal(x.numerator) / Decimal(x.denominator) if isinstance(x, Fraction) else Decimal(x)
    s = format(x.quantize(Decimal("0.01"), ROUND_HALF_UP), ",")
    return cur + s if cur in "$€£" else "%s %s" % (cur, s)


def _cbd(obj: dict, q: dict, lab: int, withhold: List[str], rng, p_add: float, p_hold: float):
    """Maybe add the CBD option; maybe replace one required input by ``TBC`` (key or ``outer.inner``) and make CBD the
    answer. Returns (obj, q, label, withheld key or None)."""
    if q.get("type") != "choice" or not withhold or rng.random() >= p_add:
        return obj, q, lab, None
    crit = list(q["criteria"])
    gold, held = crit[lab], None
    if rng.random() < p_hold:
        held = rng.choice(withhold)
        obj = copy.deepcopy(obj)
        outer, _, inner = held.partition(".")
        if inner:
            obj[outer][inner] = TBC
        else:
            obj[outer] = TBC
        gold = CBD
    crit.append(CBD)
    rng.shuffle(crit)
    return obj, dict(q, criteria=crit), crit.index(gold), held


_BANNED = None


def _ngrams(text: str, n: int = 8) -> set:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _banned() -> set:
    """Lint: with ``VISIONJEV_JEVBENCH`` pointing at a JevBench clone, the 8-grams of its public files; a generated row
    that shares one (a chance collision, e.g. two dates in a row) is dropped. Empty when the variable is not set."""
    global _BANNED
    if _BANNED is None:
        clone = os.environ.get("VISIONJEV_JEVBENCH")
        if clone:
            from .visionjev_teacher import jevbench_ngrams
            _BANNED = jevbench_ngrams(clone)
        else:
            _BANNED = set()
    return _BANNED


def _cell(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _block(v) -> str:
    """A value as text lines: a list of records becomes a pipe table, a list of strings bullets, a dict sub-lines."""
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        keys = list(v[0])
        return "\n".join([" | ".join(k.replace("_", " ") for k in keys)] +
                         [" | ".join(_cell(x.get(k, "")) for k in keys) for x in v])
    if isinstance(v, list):
        return "\n".join("- " + _cell(x) for x in v) if v else "none"
    if isinstance(v, dict):
        return "\n".join("  %s: %s" % (k.replace("_", " "), _cell(x)) for k, x in v.items())
    return _cell(v)


def _render(obj: dict, rng) -> str:
    """The same content as JSON, as ``key: value`` lines with tables, or as headed sections. ``_style`` = "prose" joins
    the values as sentences (a withheld value becomes "<Key>: to be confirmed."); "doc" never uses JSON."""
    obj = dict(obj)
    style = obj.pop("_style", None)
    if style == "prose":
        parts = [("%s: %s." % (k.replace("_", " ").capitalize(), TBC)) if v == TBC else v for k, v in obj.items()]
        return rng.choice([" ", "\n", "\n\n"]).join(parts)
    r = rng.random()
    if r < 0.4 and style != "doc":
        return json.dumps(obj, ensure_ascii=False)
    if r < 0.7:
        out = []
        for k, v in obj.items():
            b = _block(v)
            out.append("%s:%s%s" % (k.replace("_", " "), "\n" if "\n" in b else " ", b))
        return "\n".join(out)
    return "\n\n".join("%s\n%s" % (k.replace("_", " ").upper(), _block(v)) for k, v in obj.items())


def _rows(kinds, tag: str, n_train: int, n_val: int, noul_share: float = 0.0) -> Iterator[ts.Row]:
    """``kinds``: (name, fn, share of the family). ``fn(rng, i, pool)`` returns None or (state object, question, label,
    rationale, withhold keys). Splits use different seeds and pools; a repeated scenario is dropped."""
    p_add = min(0.9, (P_CBD_WRONG + P_CBD_GOLD) / max(1e-6, 1 - noul_share))
    p_hold = P_CBD_GOLD / (P_CBD_WRONG + P_CBD_GOLD)
    seen, bad = set(), _banned()
    for split, n in (("train", n_train), ("val", n_val)):
        for kname, fn, share in kinds:
            rng = random.Random("gen3-%s-%s-%s" % (tag, kname, split))
            for i in range(int(n * share)):
                item = fn(rng, i, POOLS[split])
                if item is None:
                    continue
                obj, q, lab, why, withhold = item
                obj, q, lab, held = _cbd(obj, q, lab, withhold, rng, p_add, p_hold)
                if held:
                    why = "A required input (%s) is still to be confirmed, so the answer %s." % (
                        held.split(".")[-1].replace("_", " "), CBD)
                state = _render(obj, rng)
                sig = hashlib.md5(state.encode()).hexdigest()
                if sig in seen:
                    continue
                seen.add(sig)
                if bad and _ngrams(json.dumps([state, q, why], ensure_ascii=False)) & bad:
                    continue
                yield split, state, q, lab, None, {"rationale": why, "kind": kname}


def _day_choice(ins: str, gold: date, mistakes: List[date], rng, business: bool):
    """Rank-uniform date options. ``business``: near-misses are business days (when the answer must be one), otherwise
    calendar days, so a weekend option is neither always wrong nor a hint."""
    if business:
        return _date_choice(ins, gold, mistakes, rng, weekdays_only=False)
    return _num_choice(ins, gold.toordinal(), [m.toordinal() for m in mistakes], lambda o: _d(date.fromordinal(int(o))), rng,
                       [-7, -3, -2, -1, 1, 2, 3, 7])


def _pick(rng, *xs):
    return rng.choice(xs)


def _fold_ok(local: datetime, tz) -> bool:
    """False for a wall-clock time that occurs twice or never (clock change)."""
    a, b = local.replace(tzinfo=tz, fold=0), local.replace(tzinfo=tz, fold=1)
    return a.utcoffset() == b.utcoffset() and a.astimezone(UTC).astimezone(tz).replace(tzinfo=None) == local


def _off(dt: datetime) -> str:
    m = int(dt.utcoffset().total_seconds() // 60)
    return "UTC%s%d%s" % ("+" if m >= 0 else "-", abs(m) // 60, (":%02d" % (abs(m) % 60)) if abs(m) % 60 else "")


# ---- deadline chain ---------------------------------------------------------------------------------------------------
TZ = [("Rotterdam", "Europe/Amsterdam"), ("London", "Europe/London"), ("New York", "America/New_York"),
      ("Halifax", "America/Halifax"), ("Chicago", "America/Chicago"), ("Denver", "America/Denver"),
      ("Vancouver", "America/Vancouver"), ("Tokyo", "Asia/Tokyo"), ("Singapore", "Asia/Singapore"),
      ("Sydney", "Australia/Sydney"), ("Mumbai", "Asia/Kolkata"), ("Helsinki", "Europe/Helsinki"),
      ("Auckland", "Pacific/Auckland"), ("Dubai", "Asia/Dubai"), ("Lisbon", "Europe/Lisbon"),
      ("Johannesburg", "Africa/Johannesburg")]
INSTRUMENTS = [("extended warranty", "claim"), ("purchase option", "exercise notice"), ("lease break right", "break notice"),
               ("licence renewal right", "renewal request"), ("price-protection period", "adjustment request"),
               ("appeal period", "notice of appeal"), ("trial subscription", "cancellation request")]
TERM_TXT = [
    "The {inst} runs for {n} calendar month{s} from the start date and ends on the day with the same number. Where the "
    "final month has no day with that number, it ends on that month's last day.",
    "Duration: {n} month{s}, counted from the start date to the same day-number in the final month; if that month is too "
    "short to have that day, the last day of the month is used instead.",
    "This {inst} expires {n} calendar month{s} after the start date, on the correspondingly numbered day (or, if there is "
    "no such day in that month, on that month's final day).",
]
ROLL_TXT = {
    "next": ["If that day is a Saturday, a Sunday or a listed closure day, the period ends on the next business day instead.",
             "A final day that is not a business day (weekend or listed closure) moves forward to the following business "
             "day."],
    "previous": ["If that day is a Saturday, a Sunday or a listed closure day, the period ends on the preceding business "
                 "day instead.",
                 "A final day that is not a business day (weekend or listed closure) moves back to the business day "
                 "before it."],
    "none": ["The period ends on that day even if it is a weekend or a closure day; nothing is moved.",
             "No adjustment is made for weekends or closure days."],
}
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _add_months(d: date, n: int, clamp: bool = True) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    last = calendar.monthrange(y, m + 1)[1]
    if d.day <= last:
        return date(y, m + 1, d.day)
    return date(y, m + 1, last) if clamp else date(y, m + 1, 1) + timedelta(days=d.day - 1)


def _chain(sc: dict, flag: str = "") -> datetime:
    """The deadline instant (aware, UTC) of a scenario; ``flag`` names ONE wrong step."""
    start, n, closed, z1 = sc["start"], sc["n"], sc["closed"], sc["z1"]
    end = _add_months(start, n + (flag == "plus1") - (flag == "minus1"), clamp=flag != "overflow")
    if flag == "leap" and end.month == 2:  # believes February has 28 days in a leap year / 29 in a common year
        wrong_last = 28 if calendar.isleap(end.year) else 29
        end = date(end.year, 2, 1) + timedelta(days=min(start.day, wrong_last) - 1)
    roll = sc["roll"] if flag != "noroll" else "none"
    if flag == "rollback" and roll != "none":
        roll = "previous" if roll == "next" else "next"
    step = timedelta(days=1 if roll == "next" else -1)
    while roll != "none" and (end.weekday() >= 5 or (end in closed and flag != "noclosure")):
        end += step
    if sc["mode"] == "eod":
        return datetime.combine(end, time(23, 59), tzinfo=z1).astimezone(UTC)
    if flag == "wallclock":  # added the hours on the wall clock, ignoring a clock change inside the window
        return (datetime.combine(end, time(0, 0)) + timedelta(hours=sc["H"])).replace(tzinfo=z1).astimezone(UTC)
    return datetime.combine(end, time(0, 0), tzinfo=z1).astimezone(UTC) + timedelta(hours=sc["H"])


CHAIN_FLAGS = ["overflow", "leap", "noroll", "rollback", "noclosure", "plus1", "minus1", "wallclock"]


def _tz_note(city: str, tz, a: datetime, b: datetime) -> str:
    oa, ob = a.astimezone(tz), b.astimezone(tz)
    if oa.utcoffset() == ob.utcoffset():
        return "%s is %s on the dates involved" % (city, _off(oa))
    t = a
    while t.astimezone(tz).utcoffset() == oa.utcoffset():
        t += timedelta(minutes=30)
    wall = (t + oa.utcoffset()).replace(tzinfo=None)
    return "%s is %s until its clocks change at %s local time, then %s" % (city, _off(oa), wall.strftime(DT), _off(ob))


def _dc_scenario(rng, P):
    """Sample the END month first (February, and the weeks where clock changes differ between regions, are
    over-represented), then the term length and a late-in-the-month start day."""
    ey, em = rng.choice([2027, 2028, 2028, 2029, 2030, 2031, 2032, 2032]), rng.choice([2, 2, 2, 3, 3, 4, 10, 10, 11, 6, 9, 12, 1])
    n = rng.choice([1, 2, 3, 6, 7, 9, 12, 13, 18, 24, 30])
    sy, sm = divmod(ey * 12 + em - 1 - n, 12)
    sm += 1
    day = min(rng.choice([31, 31, 30, 29, 29, 28, rng.randint(1, 27)]), calendar.monthrange(sy, sm)[1])
    start = date(sy, sm, day)
    raw = _add_months(start, n)
    (c1, zn1), (c2, zn2) = rng.sample(TZ, 2)
    z1, z2 = ZoneInfo(zn1), ZoneInfo(zn2)
    roll = rng.choices(["next", "previous", "none"], [0.65, 0.22, 0.13])[0]
    closed = set()
    if rng.random() < 0.55:  # a closure on the final day (when a weekday) or next to it, so the roll can chain
        for off in rng.sample([0, 0, 1, -1, 3, -3], rng.randint(1, 2)):
            c = raw + timedelta(days=off)
            if c.weekday() < 5:
                closed.add(c)
    mode = "eod" if rng.random() < 0.55 else "hours"
    sc = {"start": start, "n": n, "closed": closed, "z1": z1, "z2": z2, "c1": c1, "c2": c2, "roll": roll, "mode": mode,
          "H": rng.choice([12, 18, 36, 48, 60, 72]), "raw": raw}
    sc["dl"] = dl = _chain(sc)
    sc["end"] = end = _chain(dict(sc, mode="eod")).astimezone(z1).date()  # the final day, before any hours are added
    org, (inst, req) = rng.choice(P["orgs"]), rng.choice(INSTRUMENTS)
    df = rng.choice(DFMTS)
    sc.update(org=org, inst=inst, req=req, df=df)
    if mode == "eod":
        cut = _pick(rng, "A {req} counts if it is received by the end of the final day: 23:59 on that day, {c} time.",
                    "The {req} must arrive no later than 23:59 on the final day, measured on the clock in {c}.")
    else:
        cut = _pick(rng, "A {req} counts if it is received within {H} hours of 00:00 on the final day, {c} time. The hours "
                         "are actual elapsed hours.",
                    "From 00:00 on the final day ({c} clock) the holder has {H} elapsed hours in which the {req} must "
                    "arrive.")
    first = date(raw.year, raw.month, 1)
    obj = {"document": "%s: %s %s-%05d" % (org, inst, "".join(w[0] for w in inst.split()).upper(), rng.randint(0, 99999)),
           "start_date": start.strftime(df),
           "term": rng.choice(TERM_TXT).format(inst=inst, n=n, s="" if n == 1 else "s"),
           "non_business_days": rng.choice(ROLL_TXT[roll]),
           "office_closures": [c.strftime(df) for c in sorted(closed)],
           "cut_off": cut.format(req=req, c=c1, H=sc["H"]),
           "office": c1,
           "calendar_note": "%s falls on a %s." % (first.strftime("1 %B %Y"), WD[first.weekday()])}
    last = calendar.monthrange(raw.year, raw.month)[1]
    why = "%d month%s after %s is %s (%d days%s), so day %d %s → %s. " % (
        n, "" if n == 1 else "s", _d(start), raw.strftime("%B %Y"), last,
        ", leap year" if raw.month == 2 and last == 29 else "", start.day,
        "exists" if raw.day == start.day else "does not exist, last day", _d(raw))
    if end != raw:
        why += "That is %s, so it moves to the %s business day, %s. " % (
            "a closure day" if raw in closed else "a " + WD[raw.weekday()], roll, _d(end))
    sc["why"] = why
    return sc, obj


def _dc_zone_note(sc, obj, rng, t0: datetime, t1: datetime):
    if rng.random() < 0.6:
        a, b = min(t0, t1) - timedelta(days=1), max(t0, t1) + timedelta(days=1)
        obj["time_zone_note"] = "%s; %s." % (_tz_note(sc["c1"], sc["z1"], a, b), _tz_note(sc["c2"], sc["z2"], a, b))
    else:
        obj["time_zone_note"] = "Both cities keep their usual local time, including daylight-saving time where it is observed."


def _dc_cut_why(sc) -> str:
    dl, z1 = sc["dl"], sc["z1"]
    loc = dl.astimezone(z1)
    if sc["mode"] == "eod":
        return "Cut-off 23:59 %s (%s) = %s UTC. " % (sc["c1"], _off(loc), dl.strftime(DTS))
    return "00:00 %s + %d elapsed hours = %s %s (%s) = %s UTC. " % (
        _d(sc["end"]), sc["H"], loc.strftime(DTS), sc["c1"], _off(loc), dl.strftime(DTS))


def _dc_in_time(rng, i, P):
    sc, obj = _dc_scenario(rng, P)
    dl, z1, z2 = sc["dl"], sc["z1"], sc["z2"]
    off1, off2 = dl.astimezone(z1).utcoffset(), dl.astimezone(z2).utcoffset()
    alts = {_chain(sc, f) for f in CHAIN_FLAGS} | {dl + (off1 - off2), dl - (off1 - off2)}  # last two: zone slips
    want = rng.random() < 0.5
    side = sorted(a for a in alts if (a < dl if want else a > dl))
    if side and rng.random() < 0.75:  # land between the true deadline and a mistaken one: that mistake flips the answer
        a = rng.choice(side)
        gap = max(1, min(int(abs((dl - a).total_seconds()) // 60), 2880))
        delta = rng.randint(0 if want else 1, gap - 1 if want and gap > 1 else gap)
    else:
        delta = rng.choice([1, 5, 20, 45, 90, 180, 600])
    T = dl - timedelta(minutes=delta) if want else dl + timedelta(minutes=delta)
    loc = T.astimezone(z2)
    if not _fold_ok(loc.replace(tzinfo=None), z2):
        return None
    obj["submission"] = {"sent_from": sc["c2"], "local_time_at_sender": loc.strftime(sc["df"] + " %H:%M"),
                         "received": "on sending (electronic)"}
    _dc_zone_note(sc, obj, rng, dl, T)
    ok = T <= dl
    mins = int(abs((dl - T).total_seconds()) // 60)
    why = sc["why"] + _dc_cut_why(sc) + "Sent %s %s (%s) = %s UTC, %d min %s → %s." % (
        loc.strftime(DTS), sc["c2"], _off(loc), T.strftime(DTS), mins, "before" if ok and mins else
        "after" if not ok else "before (exactly at)", "in time: yes" if ok else "too late: no")
    q = ts._noul(_pick(rng, "Was the %s received in time?", "Did the %s arrive before the period ended?",
                       "Is the %s within the period set by the document?") % sc["req"])
    return obj, q, int(ok), why, []


def _dc_last_day(rng, i, P):
    sc, obj = _dc_scenario(rng, P)
    end = sc["end"]
    mist = [_chain(dict(sc, mode="eod"), f).astimezone(sc["z1"]).date() for f in CHAIN_FLAGS[:7]]
    q = _day_choice(_pick(rng, "What is the final day of the %s (office calendar)?", "On which date does the %s end?",
                           "Which day is the last day of the %s?") % sc["inst"], end, mist, rng, sc["roll"] != "none")
    if q is None:
        return None
    return obj, q[0], q[1], sc["why"] + "Final day → %s." % _d(end), ["start_date"]


def _dc_local(rng, i, P):
    sc, obj = _dc_scenario(rng, P)
    dl, z1, z2 = sc["dl"], sc["z1"], sc["z2"]
    naive = lambda t, z: t.astimezone(z).replace(tzinfo=None)  # noqa: E731
    gold = naive(dl, z2)
    std = lambda t, z: t.astimezone(z).utcoffset() - (t.astimezone(z).dst() or timedelta(0))  # noqa: E731
    mist = [naive(_chain(sc, f), z2) for f in CHAIN_FLAGS]
    mist += [naive(dl, z1),  # not converted at all
             naive(dl, z1) + (naive(dl, z1) - gold),  # offset difference applied in the wrong direction
             naive(dl, z1) - std(dl, z1) + std(dl, z2)]  # standard-time offsets used (daylight saving ignored)
    base = datetime(2026, 1, 1)
    mins = lambda t: int((t - base).total_seconds() // 60)  # noqa: E731
    q = _num_choice(_pick(rng, "What is the latest moment, on the clock in %s, at which the %s can still arrive?",
                          "Expressed in %s local time, when does the window for the %s close?") % (sc["c2"], sc["req"]),
                    mins(gold), [mins(t) for t in mist], lambda v: (base + timedelta(minutes=int(v))).strftime(DT), rng,
                    [-2880, -1440, -120, -60, 60, 120, 1440, 2880])
    if q is None:
        return None
    obj["requester_location"] = sc["c2"]
    _dc_zone_note(sc, obj, rng, dl, dl)
    why = sc["why"] + _dc_cut_why(sc) + "In %s (%s) → %s." % (sc["c2"], _off(dl.astimezone(z2)), gold.strftime(DT))
    return obj, q[0], q[1], why, ["start_date", "cut_off"]


# ---- period count -----------------------------------------------------------------------------------------------------
HALF = Fraction(1, 2)
COUNT_TXT = {
    "both": ["Count the deemed receipt day and the completion day: both ends are included.",
             "Both the deemed day of receipt and the day of completion are counted."],
    "end": ["The deemed receipt day is not counted; counting starts on the following day and includes the completion day.",
            "Day one is the day after deemed receipt, and the completion day is counted."],
    "neither": ["Neither the deemed receipt day nor the completion day is counted: only the days strictly between them.",
                "Only days strictly between deemed receipt and completion are counted; both end days are left out."],
}
CUT_TXT = [
    "A {thing} received after {c} on a business day, or at any time on a day that is not a business day, is treated as "
    "received on the next business day. Receipt at {c} exactly still counts for that day.",
    "Cut-off {c}: anything arriving later than {c}, or on a weekend or full closure day, is deemed received on the next "
    "business day; arrival at or before {c} on a business day is deemed received that day.",
]
DESKS = ["facilities desk", "claims team", "IT service desk", "records office", "customs brokerage", "payroll helpdesk"]


def _isbiz(d: date, full) -> bool:
    return d.weekday() < 5 and d not in full


def _next_biz(d: date, full) -> date:
    d += timedelta(days=1)
    while not _isbiz(d, full):
        d += timedelta(days=1)
    return d


def _deemed_day(rdate: date, rtime: time, cut: time, full, flag: str = "") -> date:
    late = rtime > cut and flag != "nocut"
    if _isbiz(rdate, full) and not late:
        return rdate
    if flag == "calroll" and _isbiz(rdate, full):
        return rdate + timedelta(days=1)  # "next day" taken as the next calendar day
    return _next_biz(rdate, full)


def _sla_count(A: date, E: date, conv: str, full, half, flag: str = "") -> Fraction:
    a = A if conv == "both" else A + timedelta(days=1)
    e = E - timedelta(days=1) if conv == "neither" else E
    tot, d = Fraction(0), a
    while d <= e:
        if flag == "calendar":
            tot += 1
        elif d.weekday() < 5:
            if d in full:
                tot += 1 if flag == "closures" else 0
            elif d in half:
                tot += {"halffull": Fraction(1), "halfzero": Fraction(0)}.get(flag, HALF)
            else:
                tot += 1
        d += timedelta(days=1)
    return tot


def _pc_base(rng, P, thing: str):
    rdate = date(2026, 1, 5) + timedelta(days=rng.randrange(0, 720))
    cut = rng.choice([time(14, 0), time(15, 0), time(16, 0), time(16, 30), time(17, 0)])
    r = rng.random()
    if r < 0.08:
        rtime = cut
    elif r < 0.5:
        rtime = (datetime.combine(rdate, cut) - timedelta(minutes=rng.randint(1, 300))).time()
    else:
        rtime = (datetime.combine(rdate, cut) + timedelta(minutes=rng.randint(1, 240))).time()
    wk = [rdate + timedelta(days=j) for j in range(0, 24) if (rdate + timedelta(days=j)).weekday() < 5]
    rng.shuffle(wk)
    nf, nh = rng.randint(1, 3), rng.randint(0, 2)
    full, half = set(wk[:nf]), set(wk[nf:nf + nh])
    df = rng.choice(DFMTS)
    obj = {"team": "%s %s" % (rng.choice(P["orgs"]), rng.choice(DESKS)),
           "receipt_rule": rng.choice(CUT_TXT).format(thing=thing, c=cut.strftime("%H:%M")),
           "business_days": "Monday to Friday, except the full closure days listed.",
           "full_closures": [c.strftime(df) for c in sorted(full)]}
    return rdate, rtime, cut, full, half, df, obj


def _pc_deemed_why(rdate, rtime, cut, full, A) -> str:
    when = "%s %s" % (_d(rdate), rtime.strftime("%H:%M"))
    if A == rdate:
        return "Received %s, not after the %s cut-off on a business day → deemed received that day. " % (
            when, cut.strftime("%H:%M"))
    why = "after the %s cut-off" % cut.strftime("%H:%M") if _isbiz(rdate, full) else "not a business day"
    return "Received %s (%s) → deemed received %s, the next business day. " % (when, why, _d(A))


def _pc_sla(rng, i, P):
    rdate, rtime, cut, full, half, df, obj = _pc_base(rng, P, "request")
    conv = rng.choice(["both", "end", "neither"])
    A = _deemed_day(rdate, rtime, cut, full)
    E = A
    for _ in range(rng.randint(3, 12)):
        E = _next_biz(E, full)
    gold = _sla_count(A, E, conv, full, half)
    alt = lambda flag="", cv=conv: _sla_count(_deemed_day(rdate, rtime, cut, full, flag), E, cv, full, half, flag)  # noqa: E731
    mist = [alt("nocut"), alt("calroll"), alt("closures"), alt("halffull"), alt("halfzero"), alt("calendar")] + \
        [alt(cv=c) for c in COUNT_TXT if c != conv]
    obj.update(half_days=[c.strftime(df) for c in sorted(half)],
               half_day_rule="A listed half day counts as 0.5 of a business day.",
               counting_rule=rng.choice(COUNT_TXT[conv]),
               request={"id": "RQ-%05d" % rng.randint(0, 99999), "received": "%s %s" % (rdate.strftime(df), rtime.strftime("%H:%M")),
                        "completed": E.strftime(df)})
    a = A if conv == "both" else A + timedelta(days=1)
    e = E - timedelta(days=1) if conv == "neither" else E
    span = [a + timedelta(days=j) for j in range((e - a).days + 1)]
    wd, nf, nh = sum(d.weekday() < 5 for d in span), sum(d in full for d in span), sum(d in half for d in span)
    fmt = lambda v: "%.1f business days" % float(v)  # noqa: E731
    why = _pc_deemed_why(rdate, rtime, cut, full, A) + "Counted span %s to %s: %d weekdays − %d full closure%s − 0.5 × %d " \
        "half day%s = %s" % (_d(a), _d(e), wd, nf, "" if nf == 1 else "s", nh, "" if nh == 1 else "s", fmt(gold))
    if rng.random() < 0.3:
        met = rng.random() < 0.5
        up = -(-gold // 1)
        target = int(up + rng.choice([0, 0, 1])) if met else int(up - 1 - rng.choice([0, 0, 1]))
        if target < 1:
            return None
        obj["service_target"] = "%d business days: met when the count is no more than %d." % (target, target)
        q = ts._noul(_pick(rng, "Was the service target met for this request?", "Did the team complete the request within "
                           "its target?", "Does the business-day count stay within the service target?"))
        return obj, q, int(gold <= target), why + ", target %d → %s." % (target, "met: yes" if gold <= target else
                                                                         "missed: no"), []
    q = _num_choice(_pick(rng, "How many business days did the request take under the counting rule?",
                          "What is the business-day count for this request?",
                          "How many business days are counted from receipt to completion?"), gold, mist, fmt, rng,
                    [Fraction(x, 2) for x in (-4, -3, -2, -1, 1, 2, 3, 4)], lo=0)
    if q is None:
        return None
    return obj, q[0], q[1], why + ".", ["request.received", "request.completed", "counting_rule"]


def _pc_storage(rng, i, P):
    rdate, rtime, cut, full, half, df, obj = _pc_base(rng, P, "consignment")
    F, first, incl = rng.randint(2, 7), rng.choice(["arrival", "after"]), rng.random() < 0.6

    def charged(flag=""):
        A = _deemed_day(rdate, rtime, cut, full, flag)
        cl = set() if flag == "closures" else full
        start_first = (first == "arrival") != (flag == "first")
        if flag == "calfree":
            L = A + timedelta(days=F - 1 if start_first else F)
        else:
            L = A if start_first else _next_biz(A, cl)
            for _ in range(F - 1):
                L = _next_biz(L, cl)
        return L, A

    L, A = charged()
    # mostly >= 4 days after the free period (room for distractors on both sides); 4%: collected inside it (no charge)
    coll = L + timedelta(days=rng.randint(4, 16) if rng.random() > 0.04 else rng.choice([-1, 0]))
    if coll < A:
        return None

    def days(flag=""):
        l, _ = charged(flag)
        last = coll if incl != (flag == "incl") else coll - timedelta(days=1)
        span = [l + timedelta(days=j) for j in range(1, (last - l).days + 1)]
        return sum(1 for d in span if flag != "bizonly" or _isbiz(d, full))

    gold = days()
    mist = [days(f) for f in ("nocut", "calroll", "closures", "first", "calfree", "incl", "bizonly")]
    rate = Decimal(rng.choice([18, 24, 35, 42.5, 60, 75]))
    obj.update(free_period="%d business days, the first free day being the %s." % (
        F, "deemed arrival day itself" if first == "arrival" else "business day after deemed arrival"),
        charging_rule="Storage is charged for every calendar day after the last free day (weekends and closure days "
        "included), up to %s the day of collection." % ("and including" if incl else "but not including"),
        daily_rate=_m2(rate),
        consignment={"id": "CN-%05d" % rng.randint(0, 99999), "arrived": "%s %s" % (rdate.strftime(df), rtime.strftime("%H:%M")),
                     "collected": coll.strftime(df)})
    why = _pc_deemed_why(rdate, rtime, cut, full, A) + "%d free business days from %s → last free day %s. Collected %s (%s)" \
        " → %d charged day%s" % (F, "that day" if first == "arrival" else "the next business day", _d(L), _d(coll),
                                "counted" if incl else "not counted", gold, "" if gold == 1 else "s")
    hold = ["consignment.arrived", "consignment.collected", "free_period"]
    if rng.random() < 0.4:
        q = _num_choice(_pick(rng, "What is the storage charge for this consignment?", "How much storage is payable?"),
                        gold * rate, [m * rate for m in mist], _m2, rng, [rate * x for x in (-3, -2, -1, 1, 2, 3)], lo=0)
        why += " × %s = %s." % (_m2(rate), _m2(gold * rate))
    else:
        q = _num_choice(_pick(rng, "For how many days is storage charged?", "How many chargeable storage days are there?"),
                        gold, mist, lambda v: "%d" % int(v), rng, [-3, -2, -1, 1, 2, 3], lo=0)
        why += " → %d." % gold
    if q is None:
        return None
    return obj, q[0], q[1], why, hold


# ---- fx aggregate -----------------------------------------------------------------------------------------------------
CUR_PAIRS = [("EUR", "USD"), ("EUR", "CHF"), ("GBP", "EUR"), ("USD", "CAD"), ("CHF", "EUR"), ("SGD", "USD"), ("AUD", "USD"),
             ("USD", "GBP")]
VWORDS = ["Services", "Logistics", "Systems", "Trading", "Supply", "Engineering", "Consulting", "Packaging"]
TIER_TITLES = [["team leader", "division head", "finance director", "board finance committee"],
               ["cost-centre owner", "controller", "chief financial officer", "executive committee"],
               ["supervisor", "plant manager", "regional director", "group treasurer"],
               ["buyer", "category manager", "head of procurement", "managing director"]]
LADDER = [1000, 2000, 2500, 4000, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 75000,
          100000, 125000, 150000, 200000, 250000, 400000]
FX_FLAGS = ["lookalike", "exactname", "nowindow", "single", "nofx", "inverse", "latest", "cancelled"]


def _fx_total(sc: dict, flag: str = ""):
    """Aggregated amount in the home currency (and the included conversions); ``flag`` names ONE wrong step."""
    tgt, tot, used = sc["target"], Decimal(0), []
    for inv in [tgt] + sc["ledger"]:
        same = inv["vid"] == tgt["vid"] or (flag == "lookalike" and inv["vid"] == sc["look_id"])
        if flag == "exactname":
            same = inv["name"] == tgt["name"]
        if not same:
            continue
        if inv is not tgt:
            age = (tgt["date"] - inv["date"]).days
            if flag == "single" or (age > sc["W"] and flag != "nowindow") or (inv["status"] == "cancelled" and flag != "cancelled"):
                continue
        amt = inv["amount"]
        if inv["cur"] != sc["H"] and flag != "nofx":
            when = tgt["date"] if flag == "latest" else inv["date"]
            rate = [r for d, r in sc["rates"] if d <= when][-1]
            div = sc["f_per_h"] != (flag == "inverse")
            amt = (amt / rate if div else amt * rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
            used.append((inv, rate, "÷" if div else "×", amt))
        else:
            used.append((inv, None, "", amt))
        tot += amt
    return tot, used


def _fx_scenario(rng, P):
    H, F = rng.choice(CUR_PAIRS)
    stem, word = rng.choice(P["stems"]), rng.choice(VWORDS)
    sfx = rng.sample(SUFFIXES, 3)
    short = "".join(c for c in stem.upper() if c not in "AEIOU")[:3]
    a1, a2, look = "%s %s %s" % (stem, word, sfx[0]), "%s %s %s" % (short, word, sfx[0]), "%s %s %s" % (stem, word, sfx[1])
    ids = rng.sample(range(100, 9999), 8)
    vid, look_id = "V-%04d" % ids[0], "V-%04d" % ids[1]
    others = [("%s %s %s" % (s, rng.choice(VWORDS), rng.choice(SUFFIXES)), "V-%04d" % k)
              for s, k in zip(rng.sample([s for s in P["stems"] if s != stem], rng.randint(2, 4)), ids[2:])]
    alias = [(a1, vid), (a2, vid), (look, look_id)] + others
    rng.shuffle(alias)
    W = rng.choice([30, 45, 60, 90])
    dv = date(2026, 4, 1) + timedelta(days=rng.randrange(0, 600))
    f_per_h = rng.random() < 0.5
    base = Decimal(rng.choice([rng.randint(5500, 9500), rng.randint(10600, 16500)])) / 10000
    rates, d0 = [], dv - timedelta(days=W + 75)
    for k in range(4):
        rates.append((d0, (base * (1 + Decimal(rng.randint(-40, 40)) / 1000)).quantize(Decimal("0.0001"))))
        d0 += timedelta(days=rng.randint(28, 45))
    rates = [(d, r) for d, r in rates if d <= dv]
    seq = iter(rng.sample(range(100, 999), 20))
    pfx = "".join(rng.sample("ABCDEFGHKLMNPRST", 2))

    def inv(name, v, age, status="posted"):
        cur = F if rng.random() < 0.6 else H
        return {"no": "%s-%d-%d" % (pfx, dv.year, next(seq)), "name": name, "vid": v, "date": dv - timedelta(days=age),
                "amount": Decimal(rng.randint(1200, 48000)), "cur": cur, "status": status}

    target = inv(rng.choice([a1, a2]), vid, 0)
    other_alias = a2 if target["name"] == a1 else a1
    ledger = []
    for _ in range(rng.choice([0, 1, 1, 2, 2, 3])):  # same vendor, inside the window (either alias; boundary day included)
        ledger.append(inv(rng.choice([other_alias, other_alias, target["name"]]), vid, rng.choice([W, rng.randint(1, W)])))
    for _ in range(rng.choice([0, 1, 1, 2])):  # same vendor, too old (first day outside included)
        ledger.append(inv(rng.choice([a1, a2]), vid, rng.choice([W + 1, rng.randint(W + 2, W + 40)])))
    for _ in range(rng.choice([1, 1, 2])):  # the look-alike vendor, inside the window
        ledger.append(inv(look, look_id, rng.randint(1, W)))
    if rng.random() < 0.35:
        ledger.append(inv(rng.choice([a1, a2]), vid, rng.randint(1, W), "cancelled"))
    for name, v in rng.sample(others, rng.randint(1, len(others))):
        ledger.append(inv(name, v, rng.randint(1, W + 30)))
    ledger.sort(key=lambda x: (x["date"], x["no"]), reverse=rng.random() < 0.5)
    sc = {"H": H, "F": F, "W": W, "rates": rates, "f_per_h": f_per_h, "target": target, "ledger": ledger, "look_id": look_id}
    gold, used = _fx_total(sc)
    if len(used) > 4 or not any(u[1] for u in used):
        return None
    sc.update(gold=gold, used=used, mist=[_fx_total(sc, f)[0] for f in FX_FLAGS])
    df = rng.choice(DFMTS)
    money = lambda x: "%s %s" % (x["cur"], format(x["amount"], ",.2f"))  # noqa: E731
    org = rng.choice(P["orgs"])
    obj = {"manual": [
        "%s, invoice approval routing." % org,
        "Vendor: match the trading name printed on the invoice EXACTLY, including the legal-form suffix, against the alias "
        "list; the vendor id found there is used for every later step. Different ids are different vendors, however "
        "similar the names.",
        "Aggregation: add to the invoice under review every other posted invoice of the same vendor id dated no more than "
        "%d days before it. Cancelled invoices are left out." % W,
        "Currency: amounts are compared in %s. Convert each %s invoice at the rate in force on its own invoice date (the "
        "latest rate row dated on or before it), round to the cent, then add." % (H, F),
        "The aggregated amount decides the approval tier."],
        "vendor_aliases": [{"trading_name": n, "vendor_id": v} for n, v in alias],
        "exchange_rates": [{"effective_from": d.strftime(df), "rate": ("1 %s = %s %s" % (H, r, F)) if f_per_h else
                            ("1 %s = %s %s" % (F, r, H))} for d, r in rates],
        "invoice_under_review": {"invoice_no": target["no"], "vendor_name": target["name"], "date": dv.strftime(df),
                                 "amount": money(target)},
        "invoice_ledger": [{"invoice_no": x["no"], "vendor_name": x["name"], "date": x["date"].strftime(df),
                            "amount": money(x), "status": x["status"]} for x in ledger]}
    parts = []
    for x, rate, op, amt in used:
        tag = "current" if x is target else "#" + x["no"].rsplit("-", 1)[1]  # the ledger number's last block
        short = "%s %s" % (x["cur"], format(x["amount"], ","))
        parts.append("%s %s %s %s = %s" % (tag, short, op, rate, format(amt, ",")) if rate else "%s %s" % (tag, short))
    sc["why"] = "'%s' → %s. Window from %s: %s; total %s" % (
        target["name"], vid, (dv - timedelta(days=W)).strftime("%d %b %Y"), "; ".join(parts), _m2(gold, H))
    sc["vid"] = vid
    return sc, obj


def _fx_tier(rng, i, P):
    got = _fx_scenario(rng, P)
    if got is None:
        return None
    sc, obj = got
    H, gold = sc["H"], sc["gold"]
    tier = lambda v, th: sum(v > t for t in th)  # noqa: E731
    want, th = rng.randrange(4), None
    for attempt in range(300):
        cand = sorted(rng.sample(LADDER, 3))
        moved = {tier(m, cand) for m in sc["mist"]} - {want}
        if tier(gold, cand) == want and len(moved) >= (2 if attempt < 150 else 1):
            th = cand
            break
    if th is None:
        return None
    titles = rng.choice(TIER_TITLES)
    names = ["Tier %d (%s)" % (k + 1, t) for k, t in enumerate(titles)]
    bands = ["up to %s" % _m2(th[0], H)] + ["above %s, up to %s" % (_m2(th[k - 1], H), _m2(th[k], H)) for k in (1, 2)] + \
        ["above %s" % _m2(th[2], H)]
    obj["approval_tiers"] = [{"tier": n, "aggregated_amount": b} for n, b in zip(names, bands)]
    q, lab = ts._fixed_choice(_pick(rng, "Which approval tier must invoice %s be routed to?",
                                    "Who has to approve invoice %s under the routing rules?",
                                    "Which tier applies to invoice %s once aggregation is done?") % sc["target"]["no"],
                              names, want, rng)
    band = " (%s)" % bands[want].replace(".00", "") if len(sc["used"]) < 4 else ""  # four conversions: keep it short
    why = sc["why"] + "%s → %s." % (band, names[want])
    return obj, q, lab, why, ["invoice_under_review.vendor_name"]


def _fx_amount(rng, i, P):
    got = _fx_scenario(rng, P)
    if got is None:
        return None
    sc, obj = got
    H, gold = sc["H"], sc["gold"]
    q = _num_choice(_pick(rng, "What is the aggregated amount for invoice %s, in " + H + "?",
                          "Which amount decides the approval routing of invoice %s?",
                          "After aggregation and conversion, what total applies to invoice %s?") % sc["target"]["no"],
                    gold, sc["mist"], lambda v: _m2(v, H), rng,
                    [gold * Decimal(x) for x in ("-0.15", "-0.09", "-0.04", "0.04", "0.09", "0.15")], lo=Decimal(0))
    if q is None:
        return None
    return obj, q[0], q[1], sc["why"] + " → %s." % _m2(gold, H), ["invoice_under_review.vendor_name",
                                                                "invoice_under_review.amount"]


# ---- lookup chain -----------------------------------------------------------------------------------------------------
CATS = ["hotel, per night", "meals, per day", "taxi, per trip", "client gift, per item", "training, per course",
        "car hire, per day"]
CLASS_SETS = [["Standard", "Field", "Executive", "Research"], ["Class A", "Class B", "Class C", "Class D"],
              ["Band 1", "Band 2", "Band 3", "Band 4"], ["Office", "Site", "Sales", "Leadership"]]
REGIONS = ["North", "Coastal", "Metro", "Overseas"]
CC_NAMES = ["Plant maintenance", "Key accounts", "Quality lab", "Warehouse operations", "Product design", "Field service",
            "Internal audit", "Customer care", "Tooling", "Regulatory affairs"]
LC_FLAGS = ["superseded", "lookalike", "category", "nouplift", "otheruplift"]


def _lc_resolve(sc: dict, flag: str = ""):
    """Employee → cost centre → policy class → limit (→ regional uplift). ``flag`` names ONE wrong step."""
    when = sc["when"]

    def latest(rows, loc):
        rows = sorted(rows, key=lambda r: r["from"])
        right = [r for r in rows if r["from"] <= when][-1]
        if flag == "superseded" and loc == sc["sup_loc"] and len(rows) > 1:
            return rows[0] if right is rows[-1] else rows[-1]
        return right

    e = latest([r for r in sc["emps"] if r["id"] == (sc["look"] if flag == "lookalike" else sc["emp_id"])], "employee")
    c = latest([r for r in sc["ccs"] if r["code"] == e["cc"]], "centre")
    cat = sc["other_cat"] if flag == "category" else sc["cat"]
    lim = latest([r for r in sc["limits"] if r["cls"] == c["cls"] and r["cat"] == cat], "limit")
    val = lim["limit"]
    if sc["uplift"] and flag != "nouplift":
        val = val * (100 + sc["uplift"][sc["other_region"] if flag == "otheruplift" else c["region"]]) / 100
    return val, e, c, lim


def _lc_scenario(rng, P):
    classes, cats = rng.choice(CLASS_SETS), rng.sample(CATS, 3)
    old = lambda: date(2023, 1, 1) + timedelta(days=rng.randrange(0, 700))  # noqa: E731
    vals = rng.sample(range(40, 900, 5), len(classes) * len(cats) + 1)
    limits = [{"cls": c, "cat": k, "limit": Decimal(vals.pop()), "from": old()} for c in classes for k in cats]
    codes = ["CC-%03d" % x for x in rng.sample(range(100, 999), rng.randint(5, 7))]
    ccs = [{"code": c, "name": n, "cls": rng.choice(classes), "region": rng.choice(REGIONS), "from": old()}
           for c, n in zip(codes, rng.sample(CC_NAMES, len(codes)))]
    sur = rng.sample(P["people"], rng.randint(7, 10))
    ids = ["E-%04d" % x for x in rng.sample(range(1000, 9999), len(sur) + 1)]
    emps = [{"id": i, "name": "%s. %s" % (rng.choice(INITIALS), s), "cc": rng.choice(codes), "from": old()}
            for i, s in zip(ids, sur)]
    tgt = emps[0]
    cc_of = {c["code"]: c for c in ccs}
    diff = [c for c in codes if cc_of[c]["cls"] != cc_of[tgt["cc"]]["cls"]]
    if not diff:
        return None
    ini = rng.choice([x for x in INITIALS if x != tgt["name"][0]])
    look = {"id": ids[-1], "name": "%s. %s" % (ini, sur[0]), "cc": rng.choice(diff), "from": old()}
    emps.append(look)
    eff = date(2026, 1, 10) + timedelta(days=rng.randrange(0, 500))
    sup_loc = rng.choice(["employee", "centre", "limit"])
    cat = rng.choice(cats)
    if sup_loc == "employee":
        emps.append(dict(tgt, cc=rng.choice(diff), **{"from": eff}))
    elif sup_loc == "centre":
        ccs.append(dict(cc_of[tgt["cc"]], cls=rng.choice([c for c in classes if c != cc_of[tgt["cc"]]["cls"]]),
                        **{"from": eff}))
    else:
        limits.append({"cls": cc_of[tgt["cc"]]["cls"], "cat": cat, "limit": Decimal(vals.pop()), "from": eff})
    if rng.random() < 0.5:  # an unrelated superseded row, so "two rows for one key" does not point at the target
        emps.append(dict(emps[1], cc=rng.choice(codes), **{"from": eff - timedelta(days=rng.randint(20, 200))}))
    when = eff + timedelta(days=rng.choice([1, -1]) * rng.randint(0, 120))
    if when < eff and rng.random() < 0.5:
        when = eff - timedelta(days=rng.randint(1, 10))
    uplift = dict(zip(REGIONS, rng.sample([5, 10, 15, 20, 25, 30], 4))) if rng.random() < 0.55 else None
    sc = {"emps": emps, "ccs": ccs, "limits": limits, "emp_id": tgt["id"], "look": look["id"], "cat": cat, "when": when,
          "other_cat": rng.choice([c for c in cats if c != cat]), "uplift": uplift, "sup_loc": sup_loc}
    gold, e, c, lim = _lc_resolve(sc)
    sc["other_region"] = rng.choice([r for r in REGIONS if r != c["region"]])
    sc["gold"], sc["mist"] = gold, [_lc_resolve(sc, f)[0] for f in LC_FLAGS]
    if sc["mist"][0] == gold:
        return None
    df = rng.choice(DFMTS)
    for t in (emps, ccs, limits):
        rng.shuffle(t)
    obj = {"rule": "%s expense limits. Follow the tables in order: employee → cost centre → policy class → limit. Where a "
           "table has more than one row for the same key, use the row with the latest 'from' date that is on or before the "
           "expense date; rows it replaces are superseded from that date. A claim is within the limit if it does not "
           "exceed it." % rng.choice(P["orgs"]),
           "employees": [{"employee_id": r["id"], "name": r["name"], "cost_centre": r["cc"], "from": r["from"].strftime(df)}
                         for r in emps],
           "cost_centres": [{"cost_centre": r["code"], "name": r["name"], "policy_class": r["cls"], "region": r["region"],
                             "from": r["from"].strftime(df)} for r in ccs],
           "policy_limits": [{"policy_class": r["cls"], "category": r["cat"], "limit": _m2(r["limit"]),
                              "from": r["from"].strftime(df)} for r in limits]}
    if uplift:
        obj["regional_uplift"] = [{"region": r, "uplift_on_limit": "+%d%%" % p} for r, p in uplift.items()]
        obj["rule"] += " The limit from the table is increased by the uplift of the cost centre's region."
    obj["claim"] = {"employee": "%s, id %s" % (tgt["name"], tgt["id"]), "category": cat, "expense_date": when.strftime(df)}
    note = " (assignment row from %s)" % _d(e["from"]) if sup_loc == "employee" else ""
    why = "Expense date %s. %s → %s%s → class %s%s → %s limit %s%s" % (
        _d(when), tgt["id"], c["code"], note, c["cls"],
        " (row from %s)" % _d(c["from"]) if sup_loc == "centre" else "", cat, _m2(lim["limit"]),
        " (row from %s)" % _d(lim["from"]) if sup_loc == "limit" else "")
    if uplift:
        why += "; region %s +%d%% = %s" % (c["region"], uplift[c["region"]], _m2(gold))
    table = [r["limit"] * gold / lim["limit"] - gold for r in limits if r["limit"] != lim["limit"]]
    sc.update(why=why, cls=c["cls"], classes=classes, near=table + [gold * Decimal(x) for x in ("-0.1", "-0.05", "0.05", "0.1")])
    return sc, obj


def _lc_limit(rng, i, P):
    got = _lc_scenario(rng, P)
    if got is None:
        return None
    sc, obj = got
    g = sc["gold"]
    q = _num_choice(_pick(rng, "Which limit applies to this claim?", "What is the maximum the employee may claim here?",
                          "What limit governs the claim in the record?"), g, sc["mist"], _m2, rng, sc["near"], lo=Decimal(0))
    if q is None:
        return None
    return obj, q[0], q[1], sc["why"] + " → %s." % _m2(g), ["claim.employee", "claim.expense_date"]


def _lc_within(rng, i, P):
    got = _lc_scenario(rng, P)
    if got is None:
        return None
    sc, obj = got
    g, want = sc["gold"], rng.random() < 0.5
    side = [m for m in sc["mist"] if (m < g if want else m > g)]
    if side and rng.random() < 0.75:  # between the true limit and a mistaken one
        a = rng.choice(side)
        amt = g if (want and rng.random() < 0.15) else a + (g - a) * Decimal(rng.randint(15, 85)) / 100
    else:
        amt = g * Decimal(rng.randint(80, 99) if want else rng.randint(101, 125)) / 100
    amt = amt.quantize(Decimal("0.01"), ROUND_HALF_UP)
    if (amt <= g) != want:
        return None
    obj["claim"]["amount_claimed"] = _m2(amt)
    q = ts._noul(_pick(rng, "Is the claimed amount within the limit that applies?", "Does this claim stay within its limit?",
                       "Can the claim be paid in full under the applicable limit?"))
    why = sc["why"] + "; claimed %s %s %s → %s." % (_m2(amt), "≤" if want else ">", _m2(g), "within: yes" if want else
                                                     "over the limit: no")
    return obj, q, int(want), why, []


def _lc_class(rng, i, P):
    got = _lc_scenario(rng, P)
    if got is None:
        return None
    sc, obj = got
    q, lab = ts._fixed_choice(_pick(rng, "Which policy class governs this claim?",
                                    "Under which policy class does the claim fall on the expense date?"),
                              sc["classes"], sc["classes"].index(sc["cls"]), rng)
    why = sc["why"].split(" → %s limit" % sc["cat"])[0] + " → %s." % sc["cls"]
    return obj, q, lab, why, ["claim.employee"] + (["claim.expense_date"] if sc["sup_loc"] != "limit" else [])


# ---- conditional probability (prose; fractions, percents, "k in n" and decimals mixed) ---------------------------------
NICE = [Fraction(n, d) for d in (4, 5, 8, 10, 20, 25, 40, 50) for n in range(1, d) if Fraction(n, d).denominator == d]
THIRDS = [Fraction(n, d) for d in (3, 6, 12) for n in range(1, d) if Fraction(n, d).denominator == d]
TWO_SKINS = [("valve", "supplier", "defective", "buys one type of valve from two suppliers"),
             ("parcel", "depot", "delivered late", "ships parcels through two depots"),
             ("claim form", "processing team", "returned with an error", "has claim forms handled by two processing teams"),
             ("loan file", "branch", "flagged at audit", "receives loan files from two branches"),
             ("circuit board", "production line", "rejected at test", "makes circuit boards on two production lines")]
STEPS = ["intake", "screening", "assembly", "calibration", "packing", "dispatch", "billing", "archiving"]


def _pfmt(p: Fraction, rng) -> str:
    """One probability in a randomly chosen exact format."""
    forms = ["%d/%d" % (p.numerator, p.denominator), "%d in %d" % (p.numerator, p.denominator)]
    if (p * 1000).denominator == 1:
        forms += [("%g%%" % float(p * 100)), ("%g%%" % float(p * 100)), "%g" % float(p)]
    return rng.choice(forms)


def _rp(rng, lo=Fraction(1, 50), hi=Fraction(49, 50), thirds=0.15) -> Fraction:
    pool = [p for p in (THIRDS if rng.random() < thirds else NICE) if lo <= p <= hi]
    return rng.choice(pool)


def _g(x) -> str:
    return "%.4g" % float(x)


def _prob_q(ins_choice: str, event: str, gold: Fraction, mist, rng, p_noul: float = 0.25):
    """A rank-uniform choice over probabilities (all decimals, or all percents), or a threshold yes / no placed between
    the true value and a mistaken one."""
    if rng.random() < p_noul:
        want = rng.random() < 0.5
        side = [m for m in mist if 0 <= m <= 1 and (m < gold if want else m > gold)]
        mid = (gold + rng.choice(side)) / 2 if side else gold + Fraction(rng.choice([3, 6, 10]) * (-1 if want else 1), 100)
        if abs(mid - gold) > Fraction(8, 100):  # a far threshold would make its level alone predict the answer
            mid = gold + Fraction(rng.randint(2, 16) * (-1 if want else 1), 200)
        T = Fraction(round(mid * 200), 2)  # percent, in steps of 0.5
        if not 0 < T < 100 or (gold * 100 > T) != want:
            return None
        q = ts._noul(_pick(rng, "Is the probability that %s greater than %g%%?", "Does the chance that %s exceed %g%%?")
                     % (event, float(T)))
        return q, int(want), "%s %g%% → %s" % (">" if want else "≤", float(T), "yes" if want else "no")
    fmt = (lambda v: "%.3f" % float(v)) if rng.random() < 0.5 else (lambda v: "%.1f%%" % float(v * 100))
    q = _num_choice(ins_choice, gold, mist, fmt, rng, [Fraction(d, 100) for d in (-9, -6, -3, 3, 6, 9)], lo=0, hi=1)
    if q is None:
        return None
    return q[0], q[1], "→ %s" % fmt(gold)


def _cp_two(rng, P, bayes: bool):
    unit, kind, bad, act = rng.choice(TWO_SKINS)
    A, B = rng.sample(P["stems"], 2)
    a, x, y = _rp(rng, Fraction(1, 5), Fraction(4, 5)), _rp(rng, hi=Fraction(1, 2)), _rp(rng, hi=Fraction(1, 2))
    if x == y:
        return None
    obj = {"_style": "prose",
           "setup": "%s %s, %s and %s." % (rng.choice(P["orgs"]), act, A, B),
           "split": _pick(rng, "{a} of the {u}s come from {A}; the rest come from {B}.",
                          "{A} accounts for {a} of all {u}s and {B} for the remainder.").format(a=_pfmt(a, rng), u=unit, A=A, B=B),
           "first_rate": _pick(rng, "A {u} from {A} is {bad} with probability {x}.",
                               "The chance that a {u} handled by {A} is {bad} is {x}.").format(u=unit, A=A, bad=bad, x=_pfmt(x, rng)),
           "second_rate": _pick(rng, "For {B} the corresponding figure is {y}.",
                                "A {u} from {B} is {bad} with probability {y}.").format(u=unit, B=B, bad=bad, y=_pfmt(y, rng))}
    pd = a * x + (1 - a) * y
    work = "P(%s) = %s × %s + %s × %s = %s + %s = %s" % (bad, _g(a), _g(x), _g(1 - a), _g(y), _g(a * x), _g((1 - a) * y), _g(pd))
    if not bayes:
        gold, mist = pd, [(x + y) / 2, a * x, x + y, a * x + a * y, 1 - pd, (1 - a) * x + a * y]
        event = "a randomly chosen %s is %s" % (unit, bad)
        ins = _pick(rng, "What is the probability that %s?", "How likely is it that %s?") % event
    else:
        first = rng.random() < 0.6
        src, num = (A, a * x) if first else (B, (1 - a) * y)
        gold = num / pd
        mist = [x if first else y, a if first else 1 - a, num, 1 - gold, (x if first else y) / (x + y), pd]
        event = "a %s that is %s came from %s" % (unit, bad, src)
        ins = _pick(rng, "A %s picked at random is %s. What is the probability that it came from %s?",
                    "Given that a %s is %s, how likely is it that %s was its source?") % (unit, bad, src)
        work += "; P(%s | %s) = %s / %s = %s" % (src, bad, _g(num), _g(pd), _g(gold))
    got = _prob_q(ins, event, gold, mist, rng)
    if got is None:
        return None
    return obj, got[0], got[1], "%s %s." % (work, got[2]), ["split", "second_rate"]


def _cp_total(rng, i, P):
    return _cp_two(rng, P, False)


def _cp_bayes(rng, i, P):
    return _cp_two(rng, P, True)


def _cp_atleast(rng, i, P):
    steps = rng.sample(STEPS, rng.randint(2, 4))
    ps = [_rp(rng, hi=Fraction(2, 5)) for _ in steps]
    obj = {"_style": "prose", "setup": "An order at %s goes through %d independent steps: %s." % (
        rng.choice(P["orgs"]), len(steps), ", ".join(steps))}
    for s, p in zip(steps, ps):
        obj["%s_rate" % s] = _pick(rng, "The %s step fails with probability %s.", "At %s, the chance of a failure is %s.") % (
            s, _pfmt(p, rng))
    none = Fraction(1)
    for p in ps:
        none *= 1 - p
    prod = Fraction(1)
    for p in ps:
        prod *= p
    any_fail = rng.random() < 0.65
    gold = 1 - none if any_fail else none
    mist = [sum(ps), prod, none if any_fail else 1 - none, max(ps), 1 - prod, 1 - sum(ps)]
    event = "at least one step fails" if any_fail else "every step succeeds"
    work = "P(no failure) = %s = %s" % (" × ".join(_g(1 - p) for p in ps), _g(none))
    if any_fail:
        work += "; P(at least one) = 1 − %s = %s" % (_g(none), _g(gold))
    got = _prob_q(_pick(rng, "What is the probability that %s?", "How likely is it that %s for a given order?") % event,
                  event, gold, mist, rng)
    if got is None:
        return None
    return obj, got[0], got[1], "%s %s." % (work, got[2]), ["%s_rate" % rng.choice(steps)]


def _cp_draws(rng, i, P):
    N, k = rng.randint(8, 30), rng.randint(2, 4)
    K = rng.randint(1, max(1, N // 3))
    unit = rng.choice(["power supplies", "sensors", "sealed vials", "battery packs", "gearboxes"])
    obj = {"_style": "prose",
           "lot": "A lot received by %s holds %d %s, of which exactly %d are faulty." % (rng.choice(P["orgs"]), N, unit, K),
           "sampling_plan": "An inspector draws %d of them at random, one after another, without putting any back." % k}
    none, terms, with_rep = Fraction(1), [], Fraction(N - K, N) ** k
    for j in range(k):
        none *= Fraction(N - K - j, N - j)
        terms.append("%d/%d" % (N - K - j, N - j))
    any_bad = rng.random() < 0.65
    gold = 1 - none if any_bad else none
    mist = [1 - with_rep if any_bad else with_rep, Fraction(K, N) * k, Fraction(K, N), none if any_bad else 1 - none,
            Fraction(K, N) ** k]
    event = "at least one drawn unit is faulty" if any_bad else "none of the drawn units is faulty"
    work = "P(none faulty) = %s = %s" % (" × ".join(terms), _g(none))
    if any_bad:
        work += "; P(at least one) = 1 − %s = %s" % (_g(none), _g(gold))
    got = _prob_q(_pick(rng, "What is the probability that %s?", "How likely is it that %s?") % event, event, gold, mist, rng)
    if got is None:
        return None
    return obj, got[0], got[1], "%s %s." % (work, got[2]), ["lot", "sampling_plan"]


def _gain(v) -> str:
    return "%s of %s" % ("gain" if v >= 0 else "loss", _m2(abs(v)))


def _cp_ev(rng, i, P):
    p1, p2 = _rp(rng, Fraction(1, 5), Fraction(9, 10), thirds=0), _rp(rng, Fraction(1, 10), Fraction(4, 5), thirds=0)
    X, Y = rng.randint(20, 400) * 100, rng.randint(2, 30) * 50
    gross = p1 * (p2 * X + (1 - p2) * Y)
    F = max(50, int(gross * Fraction(rng.randint(55, 150), 100)) // 10 * 10)
    gold = gross - F
    if gold == 0:
        return None
    obj = {"_style": "prose",
           "fee": _pick(rng, "Submitting a bid in the %s tender costs $%s, paid up front and never returned.",
                        "%s charges an entry fee of $%s per bid, which is not refunded in any outcome.") % (
               rng.choice(P["orgs"]), format(F, ",")),
           "stage_one": _pick(rng, "A bid is shortlisted with probability %s.", "The chance of reaching the shortlist is %s.")
           % _pfmt(p1, rng),
           "stage_two": "A shortlisted bid wins the contract with probability %s; the contract is worth $%s to the bidder." % (
               _pfmt(p2, rng), format(X, ",")),
           "consolation": "A shortlisted bid that does not win receives a $%s contribution towards its costs; a bid that is "
           "not shortlisted receives nothing." % format(Y, ",")}
    mist = [gross, p2 * X + (1 - p2) * Y - F, p1 * p2 * X - F, gross + F, p1 * ((1 - p2) * X + p2 * Y) - F,
            p1 * p2 * X + (1 - p2) * Y - F]
    work = "If shortlisted: %s × %s + %s × %s = %s; × %s shortlist = %s; minus the %s fee = %s" % (
        _g(p2), _m2(X), _g(1 - p2), _m2(Y), _m2(p2 * X + (1 - p2) * Y), _g(p1), _m2(gross), _m2(F), _gain(gold))
    if rng.random() < 0.25:
        q = ts._noul(_pick(rng, "Is the expected net result of submitting a bid positive?",
                           "In expectation, does a bidder come out ahead after the fee?"))
        return obj, q, int(gold > 0), "%s → %s." % (work, "positive: yes" if gold > 0 else "negative: no"), []
    d = max(Fraction(10), gross / 20)
    q = _num_choice(_pick(rng, "What is the expected net result of submitting one bid (expected receipts minus the fee)?",
                          "Per bid, what does a bidder gain or lose on average once the fee is counted?"), gold, mist, _gain,
                    rng, [d * x for x in (-3, -2, -1, 1, 2, 3)])
    if q is None:
        return None
    return obj, q[0], q[1], "%s → %s." % (work, _gain(gold)), ["fee", "stage_one", "consolation"]


# ---- constraint rank --------------------------------------------------------------------------------------------------
CR_SKINS = [
    {"what": "a carrier for a chilled shipment", "cost": "quote", "days": "transit_days", "score": "on_time_score",
     "cert": "cold_chain_certified", "cap": "pallet_spaces", "days_req": "delivery within {v} days",
     "cert_req": "cold-chain certification", "cap_req": "space for at least {v} pallets",
     "add_days": "Add {k} days for port congestion on this lane; the summary figure leaves it out.",
     "cap_less": "{k} of the listed pallet spaces are already committed to another shipper.",
     "cost_add": "A fuel surcharge of ${k} applies on top of the quote shown.",
     "cert_lapsed": "The cold-chain certificate lapsed last month and has not been renewed.",
     "cert_new": "Cold-chain certification was awarded this week; the summary line has not caught up."},
    {"what": "a venue for a staff conference", "cost": "hire_fee", "days": "setup_days", "score": "guest_rating",
     "cert": "step_free_access", "cap": "seats", "days_req": "set-up completed within {v} days",
     "cert_req": "step-free access", "cap_req": "at least {v} seats",
     "add_days": "The main hall needs {k} extra days of rigging that the summary does not count.",
     "cap_less": "{k} of the listed seats are behind pillars and cannot be used.",
     "cost_add": "A mandatory cleaning charge of ${k} is not in the hire fee shown.",
     "cert_lapsed": "The lift is out of service until further notice, so the step-free route in the summary is not available.",
     "cert_new": "A permanent ramp was fitted last week, so access is now step-free; the summary shows the old position."},
    {"what": "a contractor for a machine overhaul", "cost": "fixed_price", "days": "lead_time_days", "score": "quality_score",
     "cert": "pressure_vessel_licence", "cap": "technicians_available", "days_req": "a start within {v} days",
     "cert_req": "a valid pressure-vessel licence", "cap_req": "at least {v} technicians",
     "add_days": "Parts are on back-order: add {k} days to the lead time in the summary.",
     "cap_less": "{k} of the listed technicians are on leave for the whole period.",
     "cost_add": "Travel of ${k} is billed separately and is not in the fixed price shown.",
     "cert_lapsed": "The licence shown in the summary was suspended last month and has not been reinstated.",
     "cert_new": "The licence was granted this week; the summary line has not been updated."},
]
NEUTRAL = ["Collection before 16:00 only.", "Invoices monthly in arrears.", "Account manager changes next quarter.",
           "Prefers requests by email.", "Nothing to report from the last review.", "Offer valid for 30 days.",
           "A reference customer is available on request.", "Used once before, two years ago."]
PRI_TXT = {"cost": "lowest {cost}", "days": "fewest {days}", "score": "highest {score}"}


def _cr_eff(o: dict, notes: bool = True) -> dict:
    e = dict(o)
    if notes and o.get("adj"):
        kind, k = o["adj"]
        if kind == "cert":
            e["cert"] = k
        else:
            e[{"add_days": "days", "cap_less": "cap", "cost_add": "cost"}[kind]] += k if kind != "cap_less" else -k
    return e


def _cr_fail(e: dict, sc: dict, drop=None):
    """The first hard requirement this option fails (or None)."""
    for c in sc["cons"]:
        if c == drop:
            continue
        if (c == "days" and e["days"] > sc["dmax"]) or (c == "cert" and not e["cert"]) or (c == "cap" and e["cap"] < sc["cmin"]):
            return c
    return None


def _cr_key(e: dict, pri):
    return tuple(e[p] if p != "score" else -e[p] for p in pri)


def _cr_best(sc: dict, flag: str = ""):
    effs = [_cr_eff(o, notes=flag != "summary") for o in sc["opts"]]
    drop = flag[5:] if flag.startswith("drop:") else None
    ok = [e for e in effs if flag == "noconstraints" or _cr_fail(e, sc, drop) is None]
    if not ok:
        return None
    pri = sc["pri"]
    if flag == "swap":
        pri = [pri[1], pri[0], pri[2]]
    ok.sort(key=lambda e: _cr_key(e, pri))
    if flag == "notie":  # stopped at the first criterion: took the other option of the tie
        tied = [e for e in ok if e[pri[0]] == ok[0][pri[0]]]
        return tied[-1]["name"]
    return ok[0]["name"]


def _cr_scenario(rng, P):
    skin = rng.choice(CR_SKINS)
    need_trap = rng.random() < 0.8
    for _ in range(40):
        names = rng.sample(P["stems"], rng.randint(4, 5))
        opts = [{"name": n, "cost": rng.randint(40, 180) * 50, "days": rng.randint(2, 14), "score": rng.randint(60, 99),
                 "cert": rng.random() < 0.7, "cap": rng.randint(10, 80), "note": rng.choice(NEUTRAL), "adj": None}
                for n in names]
        sc = {"opts": opts, "cons": rng.sample(["days", "cert", "cap"], rng.choice([2, 2, 3])),
              "pri": rng.sample(["cost", "days", "score"], 3),
              "dmax": sorted(o["days"] for o in opts)[-2] + rng.choice([0, 0, 1]),
              "cmin": sorted(o["cap"] for o in opts)[1] - rng.choice([0, 0, 3])}
        first = _cr_best(sc)
        if first is None:
            continue
        kind = rng.choice(["cert_lapsed", "cert_lapsed", "add_days", "cap_less", "cost_add", "cert_new"])
        tgt = next(o for o in opts if o["name"] == first)
        if kind == "cert_lapsed" and "cert" in sc["cons"] and tgt["cert"]:
            tgt["adj"], tgt["note"] = ("cert", False), skin["cert_lapsed"]
        elif kind == "cert_new" and "cert" in sc["cons"]:
            cand = [o for o in opts if not o["cert"]]
            if not cand:
                continue
            o = rng.choice(cand)
            o["adj"], o["note"] = ("cert", True), skin["cert_new"]
        elif kind == "add_days":
            k = rng.randint(2, 6)
            tgt["adj"], tgt["note"] = ("add_days", k), skin["add_days"].format(k=k)
        elif kind == "cap_less":
            k = rng.randint(3, 25)
            tgt["adj"], tgt["note"] = ("cap_less", k), skin["cap_less"].format(k=k)
        elif kind == "cost_add":
            k = rng.randint(2, 30) * 50
            tgt["adj"], tgt["note"] = ("cost_add", k), skin["cost_add"].format(k=k)
        else:
            continue
        if rng.random() < 0.45:  # make the two best survivors tie on the first criterion, so the tie-break decides
            ok = sorted((o for o in opts if _cr_fail(_cr_eff(o), sc) is None), key=lambda o: _cr_key(_cr_eff(o), sc["pri"]))
            p = sc["pri"][0]
            if len(ok) >= 2 and not ok[0]["adj"] and not ok[1]["adj"]:
                ok[1][p] = ok[0][p]
        effs = [_cr_eff(o) for o in opts]
        ok = [e for e in effs if _cr_fail(e, sc) is None]
        keys = [_cr_key(e, sc["pri"]) for e in ok]
        if not ok or len(set(keys)) != len(keys) or any(e["cap"] < 1 for e in effs):
            continue
        gold = _cr_best(sc)
        if need_trap and _cr_best(sc, "summary") == gold:
            continue
        sc.update(gold=gold, skin=skin, effs=effs, ok=ok)
        return sc
    return None


def _cr_obj(sc, rng, P):
    s = sc["skin"]
    req = {"days": s["days_req"].format(v=sc["dmax"]), "cert": s["cert_req"], "cap": s["cap_req"].format(v=sc["cmin"])}
    names = {k: s[k].replace("_", " ") for k in ("cost", "days", "score")}
    p = [PRI_TXT[x].format(**names) for x in sc["pri"]]
    return {"task": "%s is choosing %s." % (rng.choice(P["orgs"]), s["what"]),
            "hard_requirements": [req[c] for c in sc["cons"]],
            "priority": "Among the options that meet every hard requirement, take the %s; if that is equal, the %s; if "
            "still equal, the %s." % tuple(p),
            "note_rule": "Each option has a summary line and a handling note. Where the note conflicts with or adjusts the "
            "summary, the note is authoritative.",
            "options": [{"option": o["name"], s["cost"]: _m2(o["cost"]), s["days"]: o["days"], s["score"]: o["score"],
                         s["cert"]: "yes" if o["cert"] else "no", s["cap"]: o["cap"], "handling_note": o["note"]}
                        for o in sc["opts"]]}


def _cr_why(sc) -> str:
    s, pri = sc["skin"], sc["pri"]
    adj = [o for o in sc["opts"] if o["adj"]]
    e_of = {e["name"]: e for e in sc["effs"]}
    show = lambda e, p: _m2(e[p]) if p == "cost" else str(e[p])  # noqa: E731
    why = ""
    for o in adj:
        kind, e = o["adj"][0], e_of[o["name"]]
        why += "Note on %s: %s. " % (o["name"], ("%s = %s" % (s["cert"].replace("_", " "), "yes" if e["cert"] else "no")) if kind == "cert" else
                                     "%s = %s" % (s[{"add_days": "days", "cap_less": "cap", "cost_add": "cost"}[kind]].replace("_", " "),
                                                  show(e, {"add_days": "days", "cap_less": "cap", "cost_add": "cost"}[kind])))
    fails = ["%s (%s)" % (e["name"], {"days": "%d > %d days" % (e["days"], sc["dmax"]), "cert": "%s: no" % s["cert"].replace("_", " "),
                                     "cap": "%d < %d" % (e["cap"], sc["cmin"])}[_cr_fail(e, sc)])
             for e in sc["effs"] if _cr_fail(e, sc)]
    ok = sorted(sc["ok"], key=lambda e: _cr_key(e, pri))
    why += "Fail: %s. Remaining: %s. " % (", ".join(fails) or "none", ", ".join(
        "%s (%s)" % (e["name"], show(e, pri[0])) for e in ok))
    if len(ok) > 1 and ok[0][pri[0]] == ok[1][pri[0]]:
        why += "Tie on %s, so %s decides: %s vs %s. " % (s[pri[0]].replace("_", " "), s[pri[1]].replace("_", " "),
                                                        show(ok[0], pri[1]), show(ok[1], pri[1]))
    return why


def _cr_pick(rng, i, P):
    sc = _cr_scenario(rng, P)
    if sc is None:
        return None
    obj = _cr_obj(sc, rng, P)
    names = [o["name"] for o in sc["opts"]]
    q, lab = ts._fixed_choice(_pick(rng, "Which option should be selected?", "Which option wins under the stated rules?",
                                    "Applying the requirements and then the priority, which option is chosen?"),
                              names, names.index(sc["gold"]), rng)
    hold = ["hard_requirements"] + (["priority"] if len(sc["ok"]) > 1 else [])
    return obj, q, lab, _cr_why(sc) + "Best → %s." % sc["gold"], hold


def _cr_eligible(rng, i, P):
    sc = _cr_scenario(rng, P)
    if sc is None:
        return None
    obj = _cr_obj(sc, rng, P)
    want = rng.random() < 0.5
    pool = [e for e in sc["effs"] if (_cr_fail(e, sc) is None) == want]
    noted = [e for e in pool if e["adj"]]
    if not pool:
        return None
    e = rng.choice(noted) if noted and rng.random() < 0.7 else rng.choice(pool)
    q = ts._noul(_pick(rng, "Does %s meet every hard requirement?", "Is %s still eligible once the hard requirements are "
                       "applied?", "Does %s pass all of the hard requirements?") % e["name"])
    s, f = sc["skin"], _cr_fail(e, sc)
    fact = {"days": "%s %d (limit %d)" % (s["days"].replace("_", " "), e["days"], sc["dmax"]),
            "cert": "%s %s" % (s["cert"].replace("_", " "), "yes" if e["cert"] else "no"),
            "cap": "%s %d (minimum %d)" % (s["cap"].replace("_", " "), e["cap"], sc["cmin"])}
    facts = "%s: %s" % (e["name"], ", ".join(fact[c] for c in sc["cons"]))
    why = "%s%s → %s." % ("The handling note overrides the summary. " if e["adj"] else "", facts,
                          "passes: yes" if f is None else "fails on %s: no" % {"days": s["days"].replace("_", " "),
                                                                             "cert": s["cert"].replace("_", " "),
                                                                             "cap": s["cap"].replace("_", " ")}[f])
    return obj, q, int(f is None), why, []


# ---- amended policy (long) --------------------------------------------------------------------------------------------
AP_SKINS = [
    {"title": "Returns and Refunds Policy", "req": "return request", "anchor": "delivery date", "a_key": "delivered_on",
     "r_key": "return_requested_on", "exc": "made-to-measure items", "reg": ["stock items", "spare parts", "accessories"],
     "c_key": "item_type", "ref": "RMA"},
    {"title": "Billing Disputes Policy", "req": "dispute notice", "anchor": "invoice date", "a_key": "invoice_dated",
     "r_key": "dispute_received_on", "exc": "metered pass-through charges",
     "reg": ["service fees", "subscription charges", "installation charges"], "c_key": "charge_type", "ref": "DSP"},
    {"title": "Claims Notification Rules", "req": "claim notification", "anchor": "date of loss", "a_key": "loss_occurred_on",
     "r_key": "notified_on", "exc": "glass-only losses", "reg": ["water damage losses", "theft losses", "storm losses"],
     "c_key": "loss_type", "ref": "CLM"},
    {"title": "Warranty Registration Terms", "req": "registration", "anchor": "installation date", "a_key": "installed_on",
     "r_key": "registered_on", "exc": "refurbished units", "reg": ["new units", "ex-display units", "replacement units"],
     "c_key": "unit_type", "ref": "REG"},
]
AP_FILL = [
    "Purpose. This document sets out how {org} handles each {req} so that customers and staff can predict the result. It "
    "replaces earlier circulars on the same subject and is published on the customer portal and the staff intranet.",
    "Scope. The rules apply to every customer account held with {org}, whether opened online, by telephone or through an "
    "agent. They do not apply to transactions between {org} and its own group companies, which follow the group manual.",
    "Definitions. Words in ordinary use keep their ordinary meaning. 'Customer' includes a person acting for the customer "
    "with written authority. 'Written' includes email and messages sent through the portal, but not voicemail.",
    "Channels. A {req} may be made through the portal, by email to the address on the account statement, or by post. "
    "Requests made by telephone are logged by the agent and read back to the caller before the call ends.",
    "Identification. Staff confirm the account number and one further detail before discussing a case. Where the caller "
    "cannot be identified, the agent explains how to submit the {req} in writing and makes no change to the account.",
    "Supporting material. Photographs, statements and receipts help the case handler and should be attached where they "
    "exist. {org} may ask for more material, and will say exactly what is needed and why it is relevant.",
    "Acknowledgement. Every {req} is acknowledged with a case reference. The acknowledgement confirms what was received "
    "and does not decide anything; a decision is sent separately once the case handler has reviewed the file.",
    "Case handling. One case handler owns each file from start to finish. If the handler is absent for more than a week "
    "the team leader reassigns the file and tells the customer who the new contact is.",
    "Decisions. Decisions are given in writing with short reasons. A decision letter names the clause relied on, so that "
    "the customer can check it against the text of this document.",
    "Remedies. Where a {req} succeeds, {org} puts the customer in the position the relevant clause provides for, and no "
    "further. Goodwill gestures are possible but are recorded as such and create no precedent.",
    "Payments. Money owed to a customer is paid to the original payment method where that is still open. Otherwise a "
    "bank transfer is made after the account holder's details have been verified by a second member of staff.",
    "Partial outcomes. A {req} may succeed in part. The decision letter then lists each element separately, stating "
    "which elements succeeded, which did not, and the clause that applies to each of them.",
    "Misuse. {org} may refuse a {req} that is knowingly false, and may close the account concerned. Suspected misuse is "
    "referred to the compliance team, who decide whether to report it to the authorities.",
    "Records. Case files are kept for six years after the case is closed and are then deleted, unless a dispute about "
    "the case is still open. Call recordings are kept for eighteen months.",
    "Privacy. Personal data in a case file is used only to handle the case, to meet legal duties and to improve the "
    "service in anonymised form. Customers can ask for a copy of their file at any time.",
    "Accessibility. Customers who need documents in large print, in an easy-read version or read aloud can ask any "
    "agent. Using an accessible format never changes the rules that apply to the case.",
    "Language. The authoritative text of this document is the English text. Translations are provided as a courtesy, "
    "and if a translation differs from the English text the English text is followed.",
    "Complaints. A customer who is unhappy with how a case was handled, as distinct from its outcome, can complain to "
    "the customer relations team, who reply independently of the original case handler.",
    "Escalation. A customer who disagrees with a decision can request a second look from a senior handler who took no "
    "part in it. The reviewer can confirm, change or replace the decision, and the review is free of charge.",
    "Training. Every agent completes a module on this document before handling cases alone, and a refresher each year. "
    "Team leaders sample closed cases each month and feed the findings back to their teams.",
    "Quality checks. Internal audit reviews a sample of files every quarter. The review looks at whether the clause "
    "cited in the decision was the right one, not only at whether the customer was satisfied.",
    "Agents and resellers. Where a customer dealt with a reseller, the customer may bring the {req} to the reseller or "
    "directly to {org}. The reseller passes it on unchanged and has no power to alter these rules.",
    "Events outside control. If {org}'s systems are unavailable, staff record each {req} by hand with the time it was "
    "made, and enter it once systems return. The customer is not disadvantaged by an outage on {org}'s side.",
    "Notices from {org}. Letters are sent to the last address on the account. Customers are asked to keep their contact "
    "details up to date, since {org} cannot be responsible for mail sent to an old address.",
    "Charges. {org} makes no charge for handling a {req}. Costs the customer chooses to incur, such as an independent "
    "report, are the customer's own unless the decision letter says otherwise.",
    "Relationship to law. Nothing here reduces rights a customer has under consumer law. Where the law gives more than "
    "this document does, the law is applied and the decision letter says so.",
    "Staff discretion. Agents have no authority to vary the periods or amounts stated in the numbered clauses. Only the "
    "clauses themselves and formally issued amendments determine what applies to a case.",
    "Counting days. Periods expressed in days are calendar days. The day of the triggering event is not counted, and a "
    "step taken on the last day of a period is taken in time.",
    "Review of this document. The document owner reviews the text every year. Proposals for change are consulted on "
    "with the customer panel and only take effect when issued as a numbered amendment with an effective date.",
    "Contact. Questions about this document go to the customer relations team. Opening hours and contact details are "
    "printed on every statement and shown at the foot of each portal page.",
    "Vulnerable customers. Where an agent learns that a customer is in difficulty, the agent notes it on the file with "
    "the customer's consent, so that later contacts are handled with suitable care and patience.",
    "Third-party reports. Where an engineer, surveyor or other specialist is asked to report, {org} shares the report "
    "with the customer on request, removing only personal data about other people.",
    "Representatives. A customer may appoint a relative, adviser or solicitor to deal with a case. {org} needs the "
    "customer's signed authority first, and continues to copy the customer in unless asked not to.",
    "Deceased customers. Where an account holder has died, the personal representative may make or continue a {req}. "
    "Staff ask for the grant of representation or, for small estates, a signed declaration.",
    "Business accounts. A business customer acts through the contacts named on the account. A {req} from anyone else "
    "at the business is accepted once a named contact has confirmed that person's authority.",
    "Duplicate submissions. Where the same {req} arrives through two channels, the earlier one is treated as the {req} "
    "and the later one is added to the same file as supporting material.",
    "Corrections. If {org} discovers an error in a decision it has sent, it writes to the customer with the corrected "
    "decision and an explanation, whether the correction helps the customer or not.",
    "Publication. The current text, every numbered amendment and the date each took effect are kept on the portal. "
    "Superseded versions remain available in the archive so that older cases can be checked against them.",
    "Customer panel. A panel of customers meets twice a year to comment on how this document works in practice. Its "
    "minutes are published, with personal details removed, within a month of each meeting.",
    "Reporting. The document owner reports each quarter to the board on the number of cases, the share decided for the "
    "customer, and the most frequent reasons for decisions going the other way.",
]
AP_SUB = [
    "Templates for the usual letters are held on the intranet.", "The case reference should be quoted in every later contact.",
    "Agents note the time and channel of each contact on the file.", "Nothing in this clause limits any other clause.",
    "Team leaders may issue working notes that explain this clause but cannot change it.",
    "Where several accounts are involved, each account is handled as its own case.",
    "Attachments larger than the portal limit can be sent by post.", "Copies are accepted; originals are not required.",
    "A customer may withdraw a case at any point before the decision is sent.",
    "Joint account holders may act separately unless the account says otherwise.",
    "The portal shows the status of each open case.", "Out-of-hours messages are read on the next working morning.",
    "Staff must not give an opinion on the likely outcome before the file is complete.",
    "Any figure quoted by an agent in conversation is for guidance and is confirmed in writing.",
    "This clause is reviewed together with the rest of the document.", "Interpreters are arranged on request at no charge.",
    "Decisions are not delayed merely because a related case is still open.",
    "A file that has been closed can be reopened if new material arrives.",
    "Managers record any departure from normal practice and the reason for it.",
    "Customers are told of any delay of more than a week and the reason for it.",
    "Examples in training material illustrate the clauses and do not add to them.",
    "Forms are available in paper form from any branch.", "The customer panel receives an annual summary of case statistics.",
    "Where a clause refers to a document, the version in force on the date of the event is meant.",
]


AP_BASES = ["original", "amended", "special", "summary"]


def _ap_scenario(rng, P):
    skin, org = rng.choice(AP_SKINS), rng.choice(P["orgs"])
    w = dict(zip(AP_BASES, [rng.choice([30, 45, 60]), rng.choice([21, 28, 40, 50, 75, 90]), rng.choice([7, 10, 14, 20]),
                            rng.choice([15, 25, 35, 55, 70, 100, 120])]))
    w_draft = rng.choice([x for x in (18, 24, 36, 42, 65, 80) if x not in w.values()])
    eff = date(2026, 2, 1) + timedelta(days=rng.randrange(0, 600))
    special = rng.random() < 0.35
    A = eff + timedelta(days=rng.choice([0, 0, 1, -1, -2] + [rng.randint(-70, 70) for _ in range(5)]))
    basis = "special" if special else ("amended" if A >= eff else "original")
    W = w[basis]
    want = rng.random() < 0.5
    side = [x for b, x in w.items() if b != basis and (x < W if want else x > W)]
    if side and rng.random() < 0.75:  # between the true last day and a mistaken one
        m = rng.choice(side)
        off = rng.randint(m + 1, W) if want else rng.randint(W + 1, m)
    else:
        off = W - rng.choice([0, 0, 1, 3, 8]) if want else W + rng.choice([1, 1, 2, 5, 12])
    R = A + timedelta(days=off)
    df = rng.choice(DFMTS)
    req, anchor = skin["req"], skin["anchor"]
    body = [("fill", s + " " + " ".join(rng.sample(AP_SUB, rng.randint(2, 4)))) for s in rng.sample(AP_FILL, rng.randint(23, 35))]
    body.insert(rng.randint(2, 5), ("base", "Time limit. A {req} must be made within {w_old} days of the {anchor}. The "
                                    "{anchor} itself is not counted, and a {req} made on the last day of the period is in time."))
    body.insert(rng.randint(len(body) // 2, len(body) - 2), ("exc", "Special categories. For {exc}, the period in clause "
                "{kb} is {w_exc} days instead. This special period is fixed: it is not changed by any amendment to clause {kb}."))
    body.insert(rng.randint(1, len(body) - 1), ("status", "Status of summaries. Summary cards, cover sheets and helpdesk "
                "scripts are issued for convenience only and have no force. Where one of them differs from a numbered clause "
                "or from a dated amendment, the clause or amendment prevails."))
    kb = [t for t, _ in body].index("base") + 1
    ke = [t for t, _ in body].index("exc") + 1
    no = rng.randint(1, 6)
    fm = dict(org=org, req=req, anchor=anchor, exc=skin["exc"], kb=kb, w_old=w["original"], w_exc=w["special"])
    clauses = "\n\n".join("%d. %s" % (k + 1, s.format(**fm)) for k, (_, s) in enumerate(body))
    card = _pick(rng, "AT A GLANCE (desk card): every {req} can be made up to {w} days after the {anchor}. This covers "
                 "everything we handle, with no special cases.",
                 "QUICK SUMMARY (desk card) FOR STAFF AND CUSTOMERS: you have {w} days from the {anchor} to make a {req}, whatever the "
                 "category. No other period applies.").format(req=req, w=w["summary"], anchor=anchor)
    amend = "AMENDMENT No. %d, effective %s. In clause %d, '%d days' is replaced by '%d days' wherever the %s is on or after " \
        "that effective date. Where the %s is earlier, clause %d applies as originally written." % (
            no, eff.strftime(df), kb, w["original"], w["amended"], anchor, anchor, kb)
    if rng.random() < 0.4:
        amend += " A draft circulated earlier that proposed %d days was withdrawn and never took effect." % w_draft
    head = "%s: %s" % (org, skin["title"])
    parts = [head, card, clauses, amend] if rng.random() < 0.7 else [head, card, amend, clauses] if rng.random() < 0.5 \
        else [head, clauses, amend, card]
    cat = skin["exc"] if special else rng.choice(skin["reg"])
    obj = {"_style": "doc", "policy": "\n\n".join(parts),
           "case": {"reference": "%s-%05d" % (skin["ref"], rng.randint(0, 99999)), skin["c_key"]: cat,
                    skin["a_key"]: A.strftime(df), skin["r_key"]: R.strftime(df)}}
    last = A + timedelta(days=W)
    refs = {"original": "clause %d" % kb, "amended": "amendment %d" % no, "special": "clause %d" % ke, "summary": "desk card"}
    # options name the governing text by one word of even length; the clause numbers stay in the rationale
    opt = lambda b, ok: "%s the %s %d-day period" % ("accept: within" if ok else "refuse: beyond", b, w[b])  # noqa: E731
    if special:
        why = "Category %s → special period of %d days (clause %d), which no amendment changes. " % (cat, W, ke)
    else:
        why = "Category %s is not special. %s %s is %s the amendment date %s → %s %d days (the card's %d days has no force). " % (
            cat, anchor.capitalize(), _d(A), "on or after" if A >= eff else "before", _d(eff), basis, W, w["summary"])
    ok = R <= last
    why += "Last day %s + %d days = %s; made %s, %s" % (_d(A), W, _d(last), _d(R), "%d day%s %s" % (
        abs((last - R).days), "" if abs((last - R).days) == 1 else "s", "to spare" if ok else "late") if R != last else
        "on the last day")
    sc = {"w": w, "basis": basis, "A": A, "R": R, "last": last, "ok": ok, "opt": opt, "skin": skin, "why": why,
          "ref": refs[basis]}
    return sc, obj


def _ap_outcome(rng, i, P):
    sc, obj = _ap_scenario(rng, P)
    w, b0, A, R = sc["w"], sc["basis"], sc["A"], sc["R"]
    dec = {b: R <= A + timedelta(days=w[b]) for b in AP_BASES}
    if rng.random() < 0.5:  # one option per governing text, each with the decision that text would give
        opts = [sc["opt"](b, dec[b]) for b in AP_BASES]
    else:  # two governing texts x both decisions (a slip in the day count on the right text is also on offer)
        other = [b for b in AP_BASES if b != b0]
        flips = [b for b in other if dec[b] != dec[b0]]
        b1 = rng.choice(flips if flips and rng.random() < 0.7 else other)
        opts = [sc["opt"](b, d) for b in (b0, b1) for d in (True, False)]
    gold = sc["opt"](b0, dec[b0])
    q, lab = ts._fixed_choice(_pick(rng, "What is the correct decision on this %s?", "How must this %s be decided under the "
                                    "document?", "Which outcome applies to this %s?") % sc["skin"]["req"],
                              opts, opts.index(gold), rng)
    return obj, q, lab, sc["why"] + " (%s) → %s." % (sc["ref"], gold), ["case." + sc["skin"]["a_key"],
                                                                        "case." + sc["skin"]["r_key"]]


def _ap_last_day(rng, i, P):
    sc, obj = _ap_scenario(rng, P)
    del obj["case"][sc["skin"]["r_key"]]
    mist = [sc["A"] + timedelta(days=x) for b, x in sc["w"].items() if b != sc["basis"]] + \
        [sc["last"] - timedelta(days=1), sc["last"] + timedelta(days=1)]
    q = _day_choice(_pick(rng, "What is the last day on which the %s can still be made?", "Until which date is a %s in time "
                           "for this case?") % sc["skin"]["req"], sc["last"], mist, rng, False)
    if q is None:
        return None
    why = sc["why"].split("; made")[0] + " → %s." % _d(sc["last"])
    return obj, q[0], q[1], why, ["case." + sc["skin"]["a_key"], "case." + sc["skin"]["c_key"]]


def _ap_in_time(rng, i, P):
    sc, obj = _ap_scenario(rng, P)
    q = ts._noul(_pick(rng, "Was the %s made in time?", "Is this %s within the period that applies to it?",
                       "Under the document as amended, is the %s in time?") % sc["skin"]["req"])
    return obj, q, int(sc["ok"]), sc["why"] + " → %s." % ("in time: yes" if sc["ok"] else "too late: no"), []


# ---- registration -----------------------------------------------------------------------------------------------------


def _gen(kinds, tag, n_train, n_val, noul_share):
    return lambda rng: _rows(kinds, tag, n_train, n_val, noul_share)


LIC = "generated (VisionJev, Apache-2.0); labels computed"
FAMILIES = {  # name: (kinds with their share of the family, tag, share of yes / no rows)
    "gen_deadline_chain": ([("dc_in_time", _dc_in_time, 0.34), ("dc_last_day", _dc_last_day, 0.26), ("dc_local", _dc_local, 0.40)],
                           "deadline", 0.33),
    "gen_period_count": ([("pc_sla", _pc_sla, 0.6), ("pc_storage", _pc_storage, 0.4)], "period", 0.18),
    "gen_fx_aggregate": ([("fx_tier", _fx_tier, 0.55), ("fx_amount", _fx_amount, 0.45)], "fx", 0.0),
    "gen_lookup_chain": ([("lc_limit", _lc_limit, 0.5), ("lc_within", _lc_within, 0.25), ("lc_class", _lc_class, 0.25)],
                         "lookup", 0.25),
    "gen_cond_probability": ([("cp_total", _cp_total, 0.2), ("cp_bayes", _cp_bayes, 0.25), ("cp_atleast", _cp_atleast, 0.2),
                              ("cp_draws", _cp_draws, 0.15), ("cp_ev", _cp_ev, 0.2)], "prob", 0.25),
    "gen_constraint_rank": ([("cr_pick", _cr_pick, 0.7), ("cr_eligible", _cr_eligible, 0.3)], "rank", 0.3),
    "gen_amended_policy": ([("ap_outcome", _ap_outcome, 0.5), ("ap_last_day", _ap_last_day, 0.25),
                            ("ap_in_time", _ap_in_time, 0.25)], "policy", 0.25),
}
USES = {"gen_deadline_chain": "month-end rule, leap years, business-day roll, cut-off across two time zones / DST",
        "gen_period_count": "business-day counting: inclusive / exclusive ends, closures, half days, cut-off receipt",
        "gen_fx_aggregate": "alias resolution, dated FX conversion, rolling aggregation window, approval tiers",
        "gen_lookup_chain": "3-4 hop table joins with a superseded row",
        "gen_cond_probability": "two-stage, Bayes, at-least-one and expected-value problems in prose",
        "gen_constraint_rank": "eliminate on hard constraints (note overrides summary), rank by priority with tie-break",
        "gen_amended_policy": "long policies (2-3.5k tokens): amendment date, exception, contradicting summary"}
SIZES = {"gen_amended_policy": (6000, 600)}
SOURCES: Dict[str, dict] = {}
for _name, (_kinds, _tag, _noul) in FAMILIES.items():
    _tr, _va = SIZES.get(_name, (10000, 1000))
    # 60% head-room over the caps: generators drop infeasible samples and repeated scenarios
    SOURCES[_name] = {"fn": _gen(_kinds, _tag, int(_tr * 1.6), int(_va * 1.6), _noul), "repo": "generated", "license": LIC,
                      "use": USES[_name], "train_cap": _tr, "val_cap": _va}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (list(SOURCES) if a.names == ["all"] else a.names):
        ts.build(n, a.root, spec=SOURCES[n], log=lambda m: print(m, flush=True))
