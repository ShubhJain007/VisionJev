"""Round-3 generators: every answer is recomputed here independently of ``laya.visionjev_gen3`` by re-parsing the state
object's text (numpy business-day calendar and month arithmetic, explicit UTC offsets from ``zoneinfo``, ``Decimal`` /
``Fraction``, re-implemented rules), and the structural guarantees (rank-uniform options, gold position, no length tell,
withheld inputs, rationale carries the answer, train / val disjoint) are checked on samples."""
import json
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from laya import visionjev_gen3 as G

D = "%a %d %b %Y"
FMTS = ["%a %d %b %Y", "%d %B %Y", "%Y-%m-%d"]


def _pd(s):
    for f in FMTS:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            pass
    raise ValueError(s)


def _pdt(s):
    day, hm = s.rsplit(" ", 1)
    return datetime.combine(_pd(day), datetime.strptime(hm, "%H:%M").time())


def _items(fn, n=300, seed=0, pool="train"):
    rng, out = random.Random(seed), []
    for i in range(n):
        it = fn(rng, i, G.POOLS[pool])
        if it is not None:
            out.append(it)
    assert len(out) > 0.6 * n, (fn.__name__, len(out))
    return out


def _gold(q, lab):
    return q["criteria"][lab]


def _money(s):
    return Decimal(re.sub(r"[^0-9.\-]", "", s))


def _q2(x):
    return Decimal(x).quantize(Decimal("0.01"), ROUND_HALF_UP)


# ---- deadline chain ---------------------------------------------------------------------------------------------------
CITY = dict(G.TZ)


def _deadline_utc(obj):
    """(final day, deadline as naive UTC) from the state object only."""
    start = _pd(obj["start_date"])
    n = int(re.search(r"(\d+) (?:calendar )?month", obj["term"]).group(1))
    month = np.datetime64(start.strftime("%Y-%m"), "M") + np.timedelta64(n, "M")
    first = month.astype("datetime64[D]")
    days_in = int(((month + np.timedelta64(1, "M")).astype("datetime64[D]") - first).astype(int))
    end = first + np.timedelta64(min(start.day, days_in) - 1, "D")
    txt = obj["non_business_days"]
    hol = [np.datetime64(_pd(c)) for c in obj["office_closures"]]
    if "next business day" in txt or "following business day" in txt:
        end = np.busday_offset(end, 0, roll="forward", holidays=hol)
    elif "preceding business day" in txt or "business day before it" in txt:
        end = np.busday_offset(end, 0, roll="backward", holidays=hol)
    else:
        assert "nothing is moved" in txt or "No adjustment" in txt
    end = end.astype(object)
    tz = ZoneInfo(CITY[obj["office"]])
    if "23:59" in obj["cut_off"]:
        local = datetime(end.year, end.month, end.day, 23, 59)
        return end, local - tz.utcoffset(local)
    hours = int(re.search(r"(\d+) (?:elapsed )?hours", obj["cut_off"]).group(1))
    local = datetime(end.year, end.month, end.day)
    return end, local - tz.utcoffset(local) + timedelta(hours=hours)


def test_deadline_in_time_matches_numpy_months_and_explicit_offsets():
    labels = Counter()
    for obj, q, lab, why, _ in _items(G._dc_in_time, 500):
        _, dl = _deadline_utc(obj)
        sub = _pdt(obj["submission"]["local_time_at_sender"])
        sub_utc = sub - ZoneInfo(CITY[obj["submission"]["sent_from"]]).utcoffset(sub)
        assert lab == int(sub_utc <= dl), (obj, why)
        labels[lab] += 1
        for city, off in re.findall(r"(\w[\w ]*?) is (UTC[+-][\d:]+) on the dates involved", obj["time_zone_note"]):
            at = (dl if city == obj["office"] else sub_utc).replace(tzinfo=timezone.utc).astimezone(ZoneInfo(CITY[city]))
            assert G._off(at) == off
    assert 0.4 < labels[1] / sum(labels.values()) < 0.6


def test_deadline_last_day_and_local_clock():
    for obj, q, lab, why, _ in _items(G._dc_last_day, 400):
        end, _ = _deadline_utc(obj)
        assert _gold(q, lab) == end.strftime(D), (obj, why)
    for obj, q, lab, why, _ in _items(G._dc_local, 400):
        _, dl = _deadline_utc(obj)
        there = dl.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(CITY[obj["requester_location"]]))
        assert _gold(q, lab) == there.strftime(D + " %H:%M"), (obj, why)


def test_deadline_covers_leap_february_month_end_and_clock_changes():
    seen = Counter()
    for obj, q, lab, why, _ in _items(G._dc_local, 1500, seed=3):
        seen["leap"] += "leap year" in why
        seen["clamp"] += "does not exist" in why
        seen["roll"] += "business day," in why
        seen["dst_in_window"] += "clocks change" in obj["time_zone_note"]
    assert seen["leap"] > 40 and seen["clamp"] > 150 and seen["roll"] > 200 and seen["dst_in_window"] > 5, seen


# ---- period count -----------------------------------------------------------------------------------------------------


ONE = np.timedelta64(1, "D")


def _deemed(obj, received):
    cut = datetime.strptime(re.search(r"(\d\d:\d\d)", obj["receipt_rule"]).group(1), "%H:%M").time()
    hol = [np.datetime64(_pd(c)) for c in obj["full_closures"]]
    day = np.datetime64(received.date())
    if np.is_busday(day, holidays=hol) and received.time() <= cut:
        return day, hol
    return np.busday_offset(day + ONE, 0, roll="forward", holidays=hol), hol


def test_sla_count_matches_numpy_busday_count():
    for obj, q, lab, why, _ in _items(G._pc_sla, 500):
        a, hol = _deemed(obj, _pdt(obj["request"]["received"]))
        e = np.datetime64(_pd(obj["request"]["completed"]))
        rule = obj["counting_rule"]
        if "strictly between" in rule:
            a, e = a + ONE, e - ONE
        elif "following day" in rule or "day after deemed receipt" in rule:
            a = a + ONE
        else:
            assert "oth" in rule  # both ends
        halves = [np.datetime64(_pd(c)) for c in obj["half_days"]]
        count = Fraction(int(np.busday_count(a, e + ONE, holidays=hol))) if e >= a else Fraction(0)
        count -= Fraction(sum(1 for h in halves if a <= h <= e and np.is_busday(h, holidays=hol)), 2)
        if q["type"] == "noul":
            target = int(re.match(r"(\d+) business days", obj["service_target"]).group(1))
            assert lab == int(count <= target), (obj, why)
        else:
            assert _gold(q, lab) == "%.1f business days" % float(count), (obj, why)


def test_storage_days_match_numpy_business_days():
    for obj, q, lab, why, _ in _items(G._pc_storage, 500):
        a, hol = _deemed(obj, _pdt(obj["consignment"]["arrived"]))
        free = int(re.match(r"(\d+) business days", obj["free_period"]).group(1))
        first = a if "arrival day itself" in obj["free_period"] else np.busday_offset(a, 1, holidays=hol)
        last_free = np.busday_offset(first, free - 1, holidays=hol)
        coll = np.datetime64(_pd(obj["consignment"]["collected"]))
        days = int((coll - last_free).astype(int)) - (0 if "and including" in obj["charging_rule"] else 1)
        days = max(0, days)
        got = _gold(q, lab)
        if got.startswith("$"):
            assert _money(got) == days * _money(obj["daily_rate"]), (obj, why)
        else:
            assert got == str(days), (obj, why)


# ---- fx aggregate -----------------------------------------------------------------------------------------------------


def _fx_total(obj):
    rules = obj["manual"]
    window = int(re.search(r"no more than (\d+) days", rules[2]).group(1))
    home = re.search(r"compared in (\w+)", rules[3]).group(1)
    ids = {r["trading_name"]: r["vendor_id"] for r in obj["vendor_aliases"]}
    rates = []
    for r in obj["exchange_rates"]:
        one, val, other = re.match(r"1 (\w+) = ([\d.]+) (\w+)", r["rate"]).groups()
        rates.append((_pd(r["effective_from"]), Decimal(val), one == home))  # True: foreign units per 1 home unit
    rates.sort()

    def to_home(amount, day):
        cur, val = amount.split(" ")
        val = Decimal(val.replace(",", ""))
        if cur == home:
            return val
        _, rate, per_home = [r for r in rates if r[0] <= day][-1]
        return _q2(val / rate if per_home else val * rate)

    inv = obj["invoice_under_review"]
    vid, day = ids[inv["vendor_name"]], _pd(inv["date"])
    total = to_home(inv["amount"], day)
    for r in obj["invoice_ledger"]:
        age = (day - _pd(r["date"])).days
        if ids[r["vendor_name"]] == vid and r["status"] == "posted" and 0 <= age <= window:
            total += to_home(r["amount"], _pd(r["date"]))
    return home, total


def test_fx_aggregated_amount_recomputed_with_decimal():
    for obj, q, lab, why, _ in _items(G._fx_amount, 400):
        home, total = _fx_total(obj)
        assert _gold(q, lab) == "%s %s" % (home, format(total, ",.2f")), (why, total)


def test_fx_tier_recomputed_from_the_band_table():
    tiers = Counter()
    for obj, q, lab, why, _ in _items(G._fx_tier, 500):
        home, total = _fx_total(obj)
        want = None
        for row in obj["approval_tiers"]:
            nums = [Decimal(x.replace(",", "")) for x in re.findall(r"([\d,]+\.\d\d)", row["aggregated_amount"])]
            band = row["aggregated_amount"]
            lo = nums[0] if band.startswith("above") else None
            hi = nums[-1] if "up to" in band else None
            if (lo is None or total > lo) and (hi is None or total <= hi):
                assert want is None
                want = row["tier"]
        assert _gold(q, lab) == want, (why, total)
        tiers[want.split(" (")[0]] += 1
    assert min(tiers.values()) > 0.17 * sum(tiers.values()), tiers  # the gold tier is ~uniform over the four tiers


def test_fx_lookalike_vendor_and_window_matter():
    n = hit = 0
    for obj, q, lab, why, _ in _items(G._fx_amount, 300, seed=5):
        ids = {r["trading_name"]: r["vendor_id"] for r in obj["vendor_aliases"]}
        name = obj["invoice_under_review"]["vendor_name"]
        stems = {k.rsplit(" ", 1)[0] for k in ids if ids[k] == ids[name]}  # every alias of this vendor, suffix removed
        looks = [k for k in ids if k.rsplit(" ", 1)[0] in stems and ids[k] != ids[name]]
        assert looks, "every state has a look-alike trading name with a different id"
        n += 1
        hit += any(r["vendor_name"] in looks for r in obj["invoice_ledger"])
    assert hit == n


# ---- lookup chain -----------------------------------------------------------------------------------------------------


def _lookup(obj):
    claim = obj["claim"]
    when = _pd(claim["expense_date"])
    eid = re.search(r"id (E-\d+)", claim["employee"]).group(1)

    def latest(rows):
        rows = [r for r in rows if _pd(r["from"]) <= when]
        return max(rows, key=lambda r: _pd(r["from"]))

    emp = latest([r for r in obj["employees"] if r["employee_id"] == eid])
    cc = latest([r for r in obj["cost_centres"] if r["cost_centre"] == emp["cost_centre"]])
    lim = latest([r for r in obj["policy_limits"] if r["policy_class"] == cc["policy_class"] and r["category"] == claim["category"]])
    value = _money(lim["limit"])
    for r in obj.get("regional_uplift", []):
        if r["region"] == cc["region"]:
            value = value * (100 + int(r["uplift_on_limit"].strip("+%"))) / 100
    return cc["policy_class"], value


def test_lookup_chain_recomputed_from_the_tables():
    for obj, q, lab, why, _ in _items(G._lc_limit, 400):
        assert _money(_gold(q, lab)) == _q2(_lookup(obj)[1]), why
    for obj, q, lab, why, _ in _items(G._lc_class, 400):
        assert _gold(q, lab) == _lookup(obj)[0], why
    labels = Counter()
    for obj, q, lab, why, _ in _items(G._lc_within, 400):
        assert lab == int(_money(obj["claim"]["amount_claimed"]) <= _lookup(obj)[1]), why
        labels[lab] += 1
    assert 0.4 < labels[1] / sum(labels.values()) < 0.6


def test_lookup_chain_has_a_superseded_row_on_the_path_that_changes_the_answer():
    for obj, q, lab, why, _ in _items(G._lc_limit, 300, seed=2):
        keys = Counter(r["employee_id"] for r in obj["employees"]) + Counter(r["cost_centre"] for r in obj["cost_centres"]) + \
            Counter((r["policy_class"], r["category"]) for r in obj["policy_limits"])
        assert any(v > 1 for v in keys.values())
        late, early = json.loads(json.dumps(obj)), json.loads(json.dumps(obj))
        late["claim"]["expense_date"], early["claim"]["expense_date"] = "2035-01-01", "2025-06-01"
        assert _lookup(late)[1] != _lookup(early)[1]  # before / after the change gives different limits


# ---- conditional probability ------------------------------------------------------------------------------------------
PROB = r"(\d+/\d+|\d+(?:\.\d+)?%|\d+ in \d+|0\.\d+)"


def _p(tok):
    if tok.endswith("%"):
        return Fraction(tok[:-1]) / 100
    if " in " in tok:
        a, b = tok.split(" in ")
        return Fraction(int(a), int(b))
    return Fraction(tok)


def _probs(*texts):
    return [_p(t) for text in texts for t in re.findall(PROB, text)]


def _check_prob(q, lab, value, why):
    if q["type"] == "noul":
        t = Fraction(re.search(r"(\d+(?:\.\d+)?)%\?$", q["instructions"]).group(1))
        assert lab == int(value * 100 > t), why
    else:
        assert _gold(q, lab) in ("%.3f" % float(value), "%.1f%%" % float(value * 100)), (why, value)


def test_two_stage_total_and_bayes_recomputed_with_fractions():
    for fn in (G._cp_total, G._cp_bayes):
        for obj, q, lab, why, _ in _items(fn, 400):
            a, x, y = _probs(obj["split"], obj["first_rate"], obj["second_rate"])
            first, second = re.search(r", (\w+) and (\w+)\.$", obj["setup"]).groups()
            total = a * x + (1 - a) * y
            if fn is G._cp_total:
                value = total
            else:
                ins = q["instructions"]
                from_first = (first in ins) if q["type"] == "choice" else ("came from " + first) in ins
                assert from_first or second in ins
                value = (a * x if from_first else (1 - a) * y) / total
            _check_prob(q, lab, value, why)


def test_at_least_one_and_draws_recomputed_with_fractions():
    for obj, q, lab, why, _ in _items(G._cp_atleast, 400):
        ps = _probs(*[v for k, v in obj.items() if k.endswith("_rate")])
        assert len(ps) == int(re.search(r"(\d+) independent steps", obj["setup"]).group(1))
        none = Fraction(1)
        for p in ps:
            none *= 1 - p
        _check_prob(q, lab, 1 - none if "at least one" in q["instructions"] else none, why)
    for obj, q, lab, why, _ in _items(G._cp_draws, 400):
        n, k = map(int, re.search(r"holds (\d+) .*exactly (\d+) are faulty", obj["lot"]).groups())
        draws = int(re.search(r"draws (\d+)", obj["sampling_plan"]).group(1))
        none = Fraction(1)
        for j in range(draws):
            none *= Fraction(n - k - j, n - j)
        _check_prob(q, lab, 1 - none if "at least one" in q["instructions"] else none, why)


def test_expected_value_with_fee_recomputed():
    labels = Counter()
    for obj, q, lab, why, _ in _items(G._cp_ev, 500):
        fee = _money(re.search(r"\$([\d,]+)", obj["fee"]).group(1))
        prize = _money(re.search(r"\$([\d,]+)", obj["stage_two"]).group(1))
        cons = _money(re.search(r"\$([\d,]+)", obj["consolation"]).group(1))
        (p1,), (p2,) = _probs(obj["stage_one"]), _probs(obj["stage_two"].split(";")[0])
        value = p1 * (p2 * Fraction(prize) + (1 - p2) * Fraction(cons)) - Fraction(fee)
        if q["type"] == "noul":
            assert lab == int(value > 0), why
            labels[lab] += 1
        else:
            amount = _q2(Decimal(abs(value).numerator) / Decimal(abs(value).denominator))
            assert _gold(q, lab) == "%s of $%s" % ("gain" if value >= 0 else "loss", format(amount, ",")), why
    assert 0.25 < labels[1] / sum(labels.values()) < 0.75


def test_probability_states_are_prose_with_mixed_formats():
    kinds = Counter()
    for obj, q, lab, why, _ in _items(G._cp_total, 300):
        state = G._render(obj, random.Random(0))
        assert not state.startswith("{") and "_style" not in state
        for tok in re.findall(PROB, state):
            kinds["percent" if "%" in tok else "fraction" if "/" in tok else "in" if " in " in tok else "decimal"] += 1
    assert all(kinds[k] > 60 for k in ("percent", "fraction", "in", "decimal")), kinds


# ---- constraint rank --------------------------------------------------------------------------------------------------


def _rank_state(obj):
    """Effective option attributes (handling note applied), the hard-requirement checks and the priority order."""
    opts = []
    for row in obj["options"]:
        keys = list(row)  # option, cost, days, score, certification, capacity, handling_note
        o = {"name": row["option"], keys[1]: _money(row[keys[1]]), keys[2]: row[keys[2]], keys[3]: row[keys[3]],
             keys[4]: row[keys[4]] == "yes", keys[5]: row[keys[5]]}
        note = row["handling_note"]
        m = re.search(r"[Aa]dd (\d+) days|(\d+) extra days", note)
        if m:
            o[keys[2]] += int(m.group(1) or m.group(2))
        m = re.search(r"^(\d+) of the listed", note)
        if m:
            o[keys[5]] -= int(m.group(1))
        m = re.search(r"of \$(\d+)", note)
        if m:
            o[keys[1]] += int(m.group(1))
        if re.search(r"lapsed|out of service|suspended", note):
            o[keys[4]] = False
        if re.search(r"awarded|fitted|granted", note):
            o[keys[4]] = True
        opts.append(o)
    checks = []
    for req in obj["hard_requirements"]:
        m = re.search(r"within (\d+) days", req)
        if m:
            checks.append((keys[2], lambda v, x=int(m.group(1)): v <= x))
        elif re.search(r"at least (\d+)", req):
            checks.append((keys[5], lambda v, x=int(re.search(r"at least (\d+)", req).group(1)): v >= x))
        else:
            checks.append((keys[4], lambda v: v))
    order = re.search(r"take the (.+?); if that is equal, the (.+?); if still equal, the (.+?)\.", obj["priority"]).groups()
    pri = [(p.split(" ", 1)[1].replace(" ", "_"), -1 if p.startswith("highest") else 1) for p in order]
    return opts, checks, pri


def test_constraint_rank_recomputed_from_summaries_and_notes():
    decided_by_note = ties = n = 0
    for obj, q, lab, why, _ in _items(G._cr_pick, 500):
        opts, checks, pri = _rank_state(obj)
        ok = [o for o in opts if all(f(o[k]) for k, f in checks)]
        ok.sort(key=lambda o: tuple(sign * o[k] for k, sign in pri))
        assert _gold(q, lab) == ok[0]["name"], why
        n += 1
        ties += len(ok) > 1 and ok[0][pri[0][0]] == ok[1][pri[0][0]]
        plain = json.loads(json.dumps(obj))
        for row in plain["options"]:
            row["handling_note"] = ""
        o2, c2, p2 = _rank_state(plain)
        ok2 = sorted((o for o in o2 if all(f(o[k]) for k, f in c2)), key=lambda o: tuple(s * o[k] for k, s in p2))
        decided_by_note += not ok2 or ok2[0]["name"] != ok[0]["name"]
    assert decided_by_note > 0.7 * n and ties > 0.15 * n, (decided_by_note, ties, n)


def test_constraint_eligibility_recomputed():
    labels = Counter()
    for obj, q, lab, why, _ in _items(G._cr_eligible, 500):
        opts, checks, _ = _rank_state(obj)
        name = re.search(r"(?:Does|Is) (\w+) ", q["instructions"]).group(1)
        o = next(o for o in opts if o["name"] == name)
        assert lab == int(all(f(o[k]) for k, f in checks)), why
        labels[lab] += 1
    assert 0.4 < labels[1] / sum(labels.values()) < 0.6


# ---- amended policy ---------------------------------------------------------------------------------------------------


def _policy(obj):
    doc, case = obj["policy"], list(obj["case"].values())
    old = int(re.search(r"must be made within (\d+) days", doc).group(1))
    exc_cat, exc = re.search(r"For (.+?), the period in clause \d+ is (\d+) days instead", doc).groups()
    eff, was, new = re.search(r"AMENDMENT No\. \d+, effective (.+?)\. In clause \d+, '(\d+) days' is replaced by '(\d+) days'",
                              doc).groups()
    assert int(was) == old
    anchor = _pd(case[2])
    if case[1] == exc_cat:
        basis, days = "special", int(exc)
    elif anchor >= _pd(eff):
        basis, days = "amended", int(new)
    else:
        basis, days = "original", old
    return basis, days, anchor + timedelta(days=days), (_pd(case[3]) if len(case) > 3 else None)


def test_amended_policy_recomputed_from_the_document_text():
    bases = Counter()
    for obj, q, lab, why, _ in _items(G._ap_outcome, 300):
        basis, days, last, made = _policy(obj)
        want = "%s the %s %d-day period" % ("accept: within" if made <= last else "refuse: beyond", basis, days)
        assert _gold(q, lab).startswith(want), why
        bases[basis] += 1
        card = int(re.search(r"(\d+) days (?:after|from) the", obj["policy"].split("1. ")[0] + obj["policy"][-400:]).group(1))
        assert card != days  # the summary card never states the period that really applies
    assert all(bases[b] > 50 for b in ("special", "amended", "original")), bases
    for obj, q, lab, why, _ in _items(G._ap_last_day, 300):
        assert _gold(q, lab) == _policy(obj)[2].strftime(D), why
    labels = Counter()
    for obj, q, lab, why, _ in _items(G._ap_in_time, 300):
        _, _, last, made = _policy(obj)
        assert lab == int(made <= last), why
        labels[lab] += 1
    assert 0.4 < labels[1] / sum(labels.values()) < 0.6


def test_amended_policy_token_length():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", local_files_only=True)
    except Exception:  # tokenizer not cached on this machine
        pytest.skip("Qwen tokenizer not available locally")
    rng = random.Random(0)
    for fn in (G._ap_outcome, G._ap_last_day, G._ap_in_time):
        for obj, q, lab, why, _ in _items(fn, 40, seed=4):
            n = len(tok(G._render(obj, rng))["input_ids"])
            assert 2000 <= n <= 3500, n
            assert len(tok(why)["input_ids"]) <= 190


# ---- structure: options, withheld inputs, rationale, splits -------------------------------------------------------------


def _value(opt):
    try:
        return datetime.strptime(opt, D + " %H:%M").timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(opt, D).timestamp()
    except ValueError:
        pass
    sign = -1 if opt.startswith("loss") else 1
    return sign * float(re.sub(r"[^0-9.]", "", opt.replace("business days", "")))


NUMERIC = [G._dc_last_day, G._dc_local, G._pc_sla, G._pc_storage, G._fx_amount, G._lc_limit, G._cp_total, G._cp_bayes,
           G._cp_atleast, G._cp_draws, G._cp_ev, G._ap_last_day]


@pytest.mark.parametrize("fn", NUMERIC, ids=lambda f: f.__name__)
def test_numeric_and_date_options_are_rank_uniform(fn):
    ranks, pos, longest, n = Counter(), Counter(), 0, 0
    for obj, q, lab, why, _ in _items(fn, 2000, seed=1):
        if q["type"] != "choice" or len(q["criteria"]) != 4:
            continue
        key = [_value(o) for o in q["criteria"]]
        assert len(set(q["criteria"])) == 4
        ranks[sorted(key).index(key[lab])] += 1
        pos[lab] += 1
        longest += max(range(4), key=lambda i: len(q["criteria"][i])) == lab
        n += 1
    assert n > 900
    assert all(0.19 < ranks[r] / n < 0.31 for r in range(4)), (fn.__name__, ranks)
    assert all(0.19 < pos[r] / n < 0.31 for r in range(4)), (fn.__name__, pos)
    assert longest / n < 0.32, (fn.__name__, longest / n)


@pytest.mark.parametrize("fn", [G._fx_tier, G._lc_class, G._cr_pick, G._ap_outcome], ids=lambda f: f.__name__)
def test_categorical_options_have_no_position_or_length_tell(fn):
    pos, longest, chance, n = Counter(), 0, 0.0, 0
    for obj, q, lab, why, _ in _items(fn, 2000, seed=1):
        k = len(q["criteria"])
        pos[lab / (k - 1) < 0.5] += 1
        longest += max(range(k), key=lambda i: len(q["criteria"][i])) == lab
        chance += 1 / k
        n += 1
    assert 0.42 < pos[True] / n < 0.58, pos
    assert longest / n < chance / n + 0.07, (fn.__name__, longest / n, chance / n)


@pytest.mark.parametrize("name", list(G.FAMILIES))
def test_withheld_inputs_rationales_and_split_disjointness(name):
    kinds, tag, noul_share = G.FAMILIES[name]
    n_train, n_val = (900, 200) if name != "gen_amended_policy" else (500, 120)
    rows = list(G._rows(kinds, "test-" + tag, n_train, n_val, noul_share))
    held = with_cbd = nouls = 0
    states = {"train": set(), "val": set()}
    for split, state, q, lab, target, extra in rows:
        crit = q.get("criteria") or []
        is_held = G.TBC in state
        assert extra["kind"] in {k for k, _, _ in kinds}
        why = extra["rationale"]
        assert why and why.rstrip().endswith(".")
        if crit:
            gold = crit[lab]
            assert (gold == G.CBD) == is_held
            assert gold in why, (gold, why)
            with_cbd += G.CBD in crit and gold != G.CBD
        else:
            assert not is_held
            assert why.endswith("yes." if lab else "no."), why
            nouls += 1
        held += is_held
        states[split].add(state)
        others = G.POOLS["val" if split == "train" else "train"]
        assert not any(o in state for o in others["orgs"])
    assert not states["train"] & states["val"]
    n = len(rows)
    assert 0.14 < with_cbd / n < 0.26, with_cbd / n
    assert 0.03 < held / n < 0.075, held / n
    assert abs(nouls / n - noul_share) < 0.08, nouls / n


def test_entity_pools_are_disjoint_between_splits():
    for key in ("orgs", "people", "stems"):
        assert not set(G.POOLS["train"][key]) & set(G.POOLS["val"][key])


def test_render_styles_carry_the_same_content():
    rng = random.Random(0)
    obj = _items(G._fx_amount, 5)[0][0]
    styles = {G._render(obj, rng)[:1] == "{" for _ in range(40)}
    assert styles == {True, False}
    for _ in range(20):
        text = G._render(obj, rng)
        assert obj["invoice_under_review"]["invoice_no"] in text and obj["invoice_ledger"][0]["invoice_no"] in text
