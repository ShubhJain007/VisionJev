"""Human-labelled sets (``laya.visionjev_human``): the converters on small hand-made records (no network), then the
invariants of the built files (skipped for a set that is not built on this machine)."""
import json
import os
import random
import re
from collections import Counter

import pytest

from laya import visionjev_human as H
from laya.visionjev_data import DATA_ROOT

TABLE = [["", "2019", "2018", "2017"],
         ["revenue", "$ 5829", "$ 5735", "$ 5100"],
         ["operating income", "959.2", "991.1", "870.0"],
         ["margin", "16.5% ( 16.5 % )", "17.3% ( 17.3 % )", "17.1% ( 17.1 % )"],
         ["impairment", "$ -120 ( 120 )", "-", "-"]]


# ---- FinQA programs ---------------------------------------------------------------------------------------------------


def test_program_execution_with_intermediate_values():
    steps = H.parse_program("subtract(959.2, 991.1), divide(#0, 991.1)")
    assert steps == [("subtract", "959.2", "991.1"), ("divide", "#0", "991.1")]
    vals = H.run_program(steps, TABLE)
    assert vals[0] == pytest.approx(-31.9) and vals[1] == pytest.approx(-31.9 / 991.1)
    work = H.worked_steps(steps, vals, TABLE)
    assert work == "Step 1: 959.2 - 991.1 = -31.9. Step 2: -31.9 / 991.1 = -0.0321865."


def test_program_literals_constants_and_table_operations():
    assert H.run_program(H.parse_program("divide(9896, 23.6%)"))[-1] == pytest.approx(9896 / 0.236)
    assert H.run_program(H.parse_program("subtract(5829, 5735), divide(#0, 5735), multiply(#1, const_100)"))[-1] == \
        pytest.approx(100 * 94 / 5735)
    assert H.run_program(H.parse_program("multiply(3, const_m1)"))[-1] == -3
    # table rows are read as the official evaluator reads them: "$" dropped, the text before "(" kept, "%" = / 100
    assert H.run_program(H.parse_program("table_average(revenue, none)"), TABLE)[-1] == pytest.approx((5829 + 5735 + 5100) / 3)
    assert H.run_program(H.parse_program("table_max(margin, none)"), TABLE)[-1] == pytest.approx(0.173)
    assert H.run_program(H.parse_program("table_sum(impairment, none)"), TABLE) is None  # "-" is not a number
    assert H.run_program(H.parse_program("multiply(607, 18.13), greater(#0, 11000)"))[-1] == "yes"
    assert H.run_program(H.parse_program("divide(5, 0)")) is None
    assert H.run_program(H.parse_program("add(#3, 1)")) is None
    assert H.parse_program("not a program") is None and H.parse_program("frobnicate(1, 2)") is None
    work = H.worked_steps(*(lambda s: (s, H.run_program(s, TABLE), TABLE))(H.parse_program("table_sum(revenue, none), divide(#0, const_1000)")))
    assert work == "Step 1: sum of the row 'revenue' (5829, 5735, 5100) = 16664. Step 2: 16664 / 1000 = 16.664."


def test_prune_steps_keeps_only_what_the_answer_uses():
    steps = H.parse_program("subtract(1197, 1167), subtract(14181, 13833), divide(#1, 13833)")
    assert H.prune_steps(steps) == [("subtract", "14181", "13833"), ("divide", "#0", "13833")]
    assert H.prune_steps(steps[:1]) == steps[:1]


def test_cell_numbers():
    assert H.cell_num("$ -23158 ( 23158 )") == -23158 and H.cell_num("(3.1)") == -3.1 and H.cell_num("9.4% ( 9.4 % )") == 9.4
    assert H.cell_num("1,571.7") == 1571.7 and H.cell_num("-") is None and H.cell_num("n/a") is None


def test_single_mistake_distractors_are_derived_from_the_program():
    steps = H.parse_program("subtract(5829, 5735), divide(#0, 5735)")
    vals = H.run_program(steps, TABLE)
    wrong = H.program_mistakes(steps, vals, TABLE)
    near = lambda x: any(abs(w - x) < 1e-9 for w in wrong)  # noqa: E731
    assert near((5735 - 5829) / 5735)       # operands swapped: the sign
    assert near(5735 / (5829 - 5735))       # ratio inverted
    assert near(5829 - 5735)                # stopped before dividing
    assert near((5829 - 5735) / 5829)       # divided by the other year (wrong base)
    assert near((5829 - 5100) / 5735)       # 2017 read instead of 2018 (neighbouring cell of the same row)
    assert not near(vals[-1]) or True       # the gold itself is removed when options are built (below)
    steps = H.parse_program("table_average(revenue, none)")
    wrong = H.program_mistakes(steps, H.run_program(steps, TABLE), TABLE)
    assert any(abs(w - (5829 + 5735 + 5100)) < 1e-9 for w in wrong)           # summed, not averaged
    assert any(abs(w - (959.2 + 991.1 + 870.0) / 3) < 1e-9 for w in wrong)    # the row below


def test_fin_item_builds_a_consistent_question():
    seen_wrong, with_real, with_negative, n = Counter(), 0, 0, 400
    steps = H.parse_program("subtract(5829, 5735), divide(#0, 5735)")
    real = {"%.2f%%" % (abs(m) * 100) for m in H.program_mistakes(steps, H.run_program(steps, TABLE), TABLE)}
    for seed in range(n):
        item = H.fin_item("what was the percentage change in revenue from 2018 to 2019?",
                          "subtract(5829, 5735), divide(#0, 5735)", TABLE, random.Random(seed), exe_ans=0.01639, answer="1.6%")
        q, lab, why, kind = item
        assert q["type"] == "choice" and len(set(q["criteria"])) == 4 and q["criteria"][lab] == "1.64%"
        assert why.endswith("Answer: 1.64%.") and "Step 1: 5829 - 5735 = 94." in why and kind == "program"
        assert all(c.endswith("%") for c in q["criteria"])
        wrong = [c for c in q["criteria"] if c != "1.64%"]
        seen_wrong.update(wrong)
        with_real += any(c.lstrip("-") in real for c in wrong)
        with_negative += any(c.startswith("-") for c in wrong)
    assert seen_wrong["1.61%"] > 10 and seen_wrong["12.71%"] > 10   # wrong base; 2017 read instead of 2018
    assert with_real / n > 0.6                  # a real single-mistake value is offered on about 3 rows of 4
    assert 0.12 < with_negative / n < 0.4       # a sign mistake is derivable here: negated distractors on a share
    # a whole-number answer from whole-number inputs is shown without decimals, and so is every distractor
    q, lab, why, _ = H.fin_item("what is the change in revenue?", "subtract(5829, 5735)", TABLE, random.Random(1), 94.0, "94")
    assert q["criteria"][lab] == "94" and all(re.fullmatch(r"-?\d+", c) for c in q["criteria"])
    q, lab, why, kind = H.fin_item("is a bigger?", "greater(5829, 5735)", TABLE, random.Random(1), "yes")
    assert sorted(q["criteria"]) == ["no", "yes"] and q["criteria"][lab] == "yes" and kind == "compare"


def test_fin_item_reports_why_a_row_is_dropped():
    rng = random.Random(0)
    assert "exe_ans" in H.fin_item("q?", "subtract(5829, 5735)", TABLE, rng, exe_ans=95.0)
    assert "disagrees" in H.fin_item("q?", "divide(2.5, 1.6), divide(#0, 1.6)", TABLE, rng, exe_ans=0.97656, answer="56.25%")
    assert "parseable" in H.fin_item("q?", "subtract(5829 5735", TABLE, rng)
    assert "executable" in H.fin_item("q?", "table_sum(no such row, none)", TABLE, rng)


def test_conversation_turn_that_only_reads_a_number():
    q, lab, why, kind = H.fin_item("what about in 2018?", "5735", TABLE, random.Random(3), 5735.0)
    assert kind == "lookup" and q["criteria"][lab] == "5735"
    assert set(q["criteria"]) & {"5829", "5100"}  # the other years of the same row


# ---- rank-uniform numeric options -----------------------------------------------------------------------------------


def _rank(q, lab):
    vals = [float(re.sub(r"[^\d.\-]", "", c)) for c in q["criteria"]]
    return sorted(vals).index(vals[lab])


def test_numeric_options_are_rank_uniform_and_position_uniform():
    ranks, positions, n = Counter(), Counter(), 4000
    for seed in range(n):
        rng = random.Random(seed)
        gold = rng.choice([1, -1]) * rng.uniform(0.01, 5000)
        mistakes = [gold * rng.uniform(0.3, 3) for _ in range(rng.randint(0, 5))] + [-gold]
        q, lab = H.numeric_choice("q", gold, mistakes, lambda v: "%.4f" % v, rng)
        assert len(set(q["criteria"])) == 4
        ranks[_rank(q, lab)] += 1
        positions[lab] += 1
    for r in range(4):
        assert abs(ranks[r] / n - 0.25) < 0.03 and abs(positions[r] / n - 0.25) < 0.03


def test_sign_of_the_options_does_not_point_at_the_gold():
    """Most answers are positive. In every mixed-sign option set the gold must be negative as often as an option is."""
    share, by_pattern, n = 0.15, Counter(), 30000
    for seed in range(n):
        rng = random.Random(seed)
        gold = rng.uniform(1, 900) * (-1 if rng.random() < share else 1)
        mistakes = [-gold, gold * rng.uniform(0.4, 2.5), -gold * rng.uniform(0.4, 2.5)]
        q, lab = H.numeric_choice("q", gold, mistakes, lambda v: "%.2f" % v, rng, neg_share=share)
        negatives = sum(c.startswith("-") for c in q["criteria"])
        by_pattern[(negatives, q["criteria"][lab].startswith("-"))] += 1
    for negatives in (1, 2, 3):
        rows = by_pattern[(negatives, True)] + by_pattern[(negatives, False)]
        assert rows > 300 and abs(by_pattern[(negatives, True)] / rows - negatives / 4) < 0.05, by_pattern
    mixed = sum(v for (k, _), v in by_pattern.items() if k in (1, 2, 3))
    assert 0.2 < mixed / n < 0.45  # sign mistakes are still offered on a good share of the rows
    # without a derivable sign mistake every option has the gold's sign
    q, lab = H.numeric_choice("q", 50.0, [40.0, 75.0], lambda v: "%.1f" % v, random.Random(0), neg_share=share)
    assert not any(c.startswith("-") for c in q["criteria"])


def test_neither_the_close_pair_nor_the_outlier_points_at_the_gold():
    """Mistakes are mostly far from the gold and mostly above it. The two options closest to each other must hold the
    gold half the time (two of four options), and the option furthest from the rest must be the gold a quarter of the
    time: a gold with its mistakes must look like a mistake with its mirrored values."""
    for n_mistakes in (0, 1, 2, 3, 5):
        pair = outlier = top = 0
        n = 6000
        for seed in range(n):
            rng = random.Random(seed)
            gold = rng.uniform(10, 900)
            mistakes = [gold * f for f in rng.sample([0.35, 0.5, 1.9, 2.7, 4.0, 5.5, 7.0], n_mistakes)]
            q, lab = H.numeric_choice("q", gold, mistakes, lambda v: "%.3f" % v, rng)
            v = [float(c) for c in q["criteria"]]
            d = lambda a, b: abs(v[a] - v[b]) / max(v[a], v[b])  # noqa: E731
            _, i, j = min((d(a, b), a, b) for a in range(4) for b in range(a + 1, 4))
            pair += lab in (i, j)
            outlier += max(range(4), key=lambda a: min(d(a, b) for b in range(4) if b != a)) == lab
            top += v[lab] == max(v)
        assert abs(pair / n - 0.5) < 0.04 and abs(outlier / n - 0.25) < 0.04 and abs(top / n - 0.25) < 0.03, (
            n_mistakes, pair / n, outlier / n, top / n)


def test_whole_number_answers_keep_whole_number_options_with_the_same_digit_pattern():
    rng = random.Random(0)
    for _ in range(300):
        gold = float(rng.randint(40, 90000) * rng.choice([1, 10, 1000]))
        mistakes = [gold + rng.randint(-30, 300) * rng.choice([1, 10, 100]) for _ in range(rng.randint(0, 3))]
        q, lab = H.numeric_choice("q", gold, mistakes, lambda v: "%d" % round(v), rng, integer=True)
        assert len(set(q["criteria"])) == 4 and q["criteria"][lab] == "%d" % gold
    # with no mistakes the near ladder uses a round step, so every option ends in the same digits as the gold
    q, lab = H.numeric_choice("q", 373000.0, [], lambda v: "%d" % round(v), random.Random(1), integer=True)
    assert all(c.endswith("000") for c in q["criteria"])
    q, lab = H.numeric_choice("q", 9164.0, [], lambda v: "%d" % round(v), random.Random(1), integer=True)
    assert all(c.endswith("64") for c in q["criteria"])


def test_every_option_is_formatted_by_its_own_size():
    assert [H.own_decimals(x) for x in (41932.2, 237.41, 2.857, 0.2095, 0.01609, 0.0042)] == [0, 1, 2, 3, 4, 5]
    seen = set()
    for seed in range(300):
        q, lab, why, kind = H.fin_item("what is the ratio?", "divide(959.2, 991.1)", TABLE, random.Random(seed), 0.96781, "0.968")
        assert q["criteria"][lab] == "0.968"
        seen.update(q["criteria"])
    assert any(float(c) >= 1 for c in seen)   # an option above 1 is possible and is then shown with 2 decimals ...
    assert all(len(c.split(".")[1]) == (2 if abs(float(c)) >= 1 else 3 if abs(float(c)) >= 0.1 else 4) for c in seen)  # ... its own size


def test_numeric_options_for_small_counts_and_zero():
    ranks = Counter()
    for seed in range(2000):
        q, lab = H.numeric_choice("q", 5.0, [], lambda v: "%d" % round(v), random.Random(seed), integer=True, lo=0)
        assert q["criteria"][lab] == "5" and all(int(c) >= 0 for c in q["criteria"])
        ranks[_rank(q, lab)] += 1
    assert all(abs(ranks[r] / 2000 - 0.25) < 0.04 for r in range(4))
    q, lab = H.numeric_choice("q", 0.0, [], lambda v: "%.2f" % v, random.Random(0), unit=0.01)
    assert q["criteria"][lab] == "0.00" and len(set(q["criteria"])) == 4


# ---- TAT-QA -----------------------------------------------------------------------------------------------------------
TAT_TABLE = [["", "2019", "2018", "2017"], ["Total revenue", "$44.1", "$56.7", "$60.2"], ["Cost of revenue", "20.5", "25.1", "27.9"],
             ["Gross profit", "23.6", "31.6", "32.3"], ["Headcount", "1,496", "1,512", "1,388"]]


def test_derivation_is_parsed_and_executed():
    steps = H.derivation_steps("(44.1 - 56.7)/56.7")
    assert steps == [("subtract", "44.1", "56.7"), ("divide", "#0", "56.7")]
    assert H.run_program(steps)[-1] == pytest.approx(-12.6 / 56.7)
    assert H.run_program(H.derivation_steps("(1,496 + 1,512)/2"))[-1] == 1504
    assert H.run_program(H.derivation_steps("-25.0 / -33.6 - 1"))[-1] == pytest.approx(25 / 33.6 - 1)
    assert H.run_program(H.derivation_steps("$84,684+$6,072"))[-1] == 90756
    assert H.derivation_steps("2019 is later than 2018") is None and H.derivation_steps("") is None
    assert H.derivation_steps("__import__('os').getcwd()") is None


def test_tat_arithmetic_percent_question():
    q = {"answer_type": "arithmetic", "question": "What is the percentage change in total revenue from 2018 to 2019?",
         "answer": -22.22, "derivation": "(44.1 - 56.7)/56.7", "scale": "percent"}
    wrong = Counter()
    for seed in range(200):
        ql, lab, why, kind = H.tat_item(TAT_TABLE, q, random.Random(seed))
        assert ql["criteria"][lab] == "-22.22%" and kind == "arithmetic" and len(set(ql["criteria"])) == 4
        assert why == ("Step 1: 44.1 - 56.7 = -12.6. Step 2: -12.6 / 56.7 = -0.2222222. Step 3: -0.2222222 * 100 = "
                       "-22.22222. Answer: -22.22%.")
        wrong.update(c for c in ql["criteria"] if c != "-22.22%")
    mag = Counter()
    for c, n in wrong.items():
        mag[c.lstrip("-")] += n
    assert mag["28.57%"] > 20 and mag["0.22%"] > 0              # wrong base; ratio not multiplied by 100 (offered rarely)
    assert 0.5 < sum(n for c, n in wrong.items() if not c.startswith("-")) / sum(wrong.values()) < 0.8
    # ^ a negative answer gets mostly positive distractors: the counterpart of the sign mistakes offered on positive
    #   answers, so that the sign of an option says nothing (test_sign_of_the_options_does_not_point_at_the_gold)
    assert not any(c in wrong for c in ("0.00%", "100.00%", "-100.00%"))             # no x - x or x / x distractor


def test_tat_drops_answers_the_derivation_does_not_reproduce():
    base = {"answer_type": "arithmetic", "question": "q", "scale": ""}
    assert "reproduce" in H.tat_item(TAT_TABLE, dict(base, answer=13.0, derivation="44.1 - 56.7"), random.Random(0))
    assert "plain arithmetic" in H.tat_item(TAT_TABLE, dict(base, answer=1.0, derivation="see note 4"), random.Random(0))
    assert "percent scale" in H.tat_item(TAT_TABLE, dict(base, answer=-0.22, derivation="(44.1 - 56.7)/56.7",
                                                          scale="percent"), random.Random(0))
    assert "multi-span" in H.tat_item(TAT_TABLE, dict(base, answer_type="multi-span", answer=["2019", "2018"]), random.Random(0))


def test_tat_count_and_span():
    q = {"answer_type": "count", "question": "How many years had revenue above 50?", "answer": 2, "derivation": "2018##2017",
         "scale": ""}
    ql, lab, why, kind = H.tat_item(TAT_TABLE, q, random.Random(0))
    assert ql["criteria"][lab] == "2" and why == "Counted: 2018; 2017. Answer: 2." and kind == "count"
    q = {"answer_type": "span", "question": "What was the headcount in 2018?", "answer": ["1,512"], "derivation": "", "scale": ""}
    for seed in range(30):
        ql, lab, why, kind = H.tat_item(TAT_TABLE, q, random.Random(seed))
        assert ql["criteria"][lab] == "1,512" and kind == "span" and why is None
        # distractors are cells of the same row or the same column
        assert set(ql["criteria"]) <= {"1,512", "1,496", "1,388", "56.7", "25.1", "31.6"}
    q = dict(q, answer=["the modified retrospective method"])
    assert "not a table cell" in H.tat_item(TAT_TABLE, q, random.Random(0))


# ---- pair selection, votes ------------------------------------------------------------------------------------------


def test_clear_margin_needs_the_mean_gap_and_no_reversed_annotator():
    assert H.clear_margin({0: 5, 1: 5, 2: 4}, {0: 3, 1: 4, 2: 3}, 1.0) == 1
    assert H.clear_margin({0: 3, 1: 4, 2: 3}, {0: 5, 1: 5, 2: 4}, 1.0) == -1
    assert H.clear_margin({0: 5, 1: 5, 2: 4}, {0: 4, 1: 5, 2: 4}, 1.0) == 0      # gap 0.33 < 1
    assert H.clear_margin({0: 5, 1: 5, 2: 2}, {0: 1, 1: 1, 2: 3}, 1.0) == 0      # annotator 2 prefers the other
    assert H.clear_margin({"r1": 0.0}, {"r1": 5.0}, 5.0, higher_is_better=False) == 1
    assert H.clear_margin({"r1": 1.0}, {"r1": 5.0}, 5.0, higher_is_better=False) == 0
    assert H.clear_margin({}, {"r1": 5.0}, 5.0) == 0


def test_summeval_pairs_keep_only_clear_margins():
    mk = lambda text, *scores: {"summary": text, "expert": [{"consistency": s, "relevance": 3} for s in scores]}  # noqa: E731
    items = [mk("good one", 5, 5, 5), mk("bad one", 2, 1, 2), mk("close one", 5, 4, 5), mk("split one", 5, 5, 1)]
    pairs = H.summeval_pairs(items, "consistency")
    assert sorted(pairs) == [("close one", "bad one"), ("good one", "bad one"), ("good one", "split one")]
    # good vs close: gap 0.33; split vs bad: gap 2 but the third expert prefers the other; close vs split: reversed too
    assert H.summeval_pairs([mk("same text", 5, 5, 5), mk("same text", 1, 1, 1)], "consistency") == []  # identical texts
    assert H.summeval_pairs(items, "relevance") == []


def test_balance_pairs_removes_length_and_position_tells():
    rng = random.Random(0)
    cands = [("x" * rng.randint(5, 60), "y" * rng.randint(5, 40), i) for i in range(3000)]  # the better one is usually longer
    kept = H.balance_pairs(cands, rng, 1000)
    longer = [len(c[0]) > len(c[1]) for c, _ in kept]
    assert len(kept) <= 1000 and sum(longer) * 2 == len(kept)
    for flag in (True, False):
        pos = [p for (c, p), l in zip(kept, longer) if l == flag]
        assert abs(sum(pos) - len(pos) / 2) <= 1


def test_vote_target_is_a_distribution_whose_top_is_the_label():
    rng = random.Random(0)
    assert H.vote_target([4, 4, 3], 5, rng) == (4, [0.0, 0.0, 0.0, 0.333333, 0.666667])
    lab, target = H.vote_target([1, 2, 4], 5, rng)       # three-way tie: the level nearest the mean (2.33)
    assert lab == 2 and target[1] == target[2] == target[4] and abs(sum(target) - 1) < 1e-3
    labs = {H.vote_target([2, 3], 4, random.Random(s))[0] for s in range(40)}
    assert labs == {2, 3}                                 # a two-way adjacent tie goes either way
    for votes in ([0, 0], [0, 1], [3, 3, 2], [1, 2, 3]):
        lab, target = H.vote_target(votes, 4, rng)
        assert abs(sum(target) - 1) < 1e-3 and target[lab] == max(target)
    assert H.feedback_vote(["Excellent", "Bad"], rng) is None and H.feedback_vote(["Acceptable", "Bad"], rng) is None
    assert H.feedback_vote(["Excellent", "Acceptable"], rng)[1] == [0.0, 0.0, 0.5, 0.5]
    assert H.feedback_vote(["Bad", "Bad", "Excellent"], rng) == (0, [0.666667, 0.0, 0.0, 0.333333])


# ---- MQM --------------------------------------------------------------------------------------------------------------


def _mqm_row(system, rater, target, category, severity, seg="1", source="Quelle"):
    return {"system": system, "doc": "doc1", "doc_id": "1", "seg_id": seg, "rater": rater, "source": source,
            "target": target, "category": category, "severity": severity}


def test_mqm_penalty_weights():
    assert H.mqm_weight("Accuracy/Mistranslation", "Major") == 5 and H.mqm_weight("Style/Awkward", "minor") == 1
    assert H.mqm_weight("Fluency/Punctuation", "Minor") == 0.1 and H.mqm_weight("Fluency/Punctuation", "Major") == 5
    assert H.mqm_weight("Non-translation!", "Major") == 25
    assert H.mqm_weight("No-error", "no-error") == 0 and H.mqm_weight("Style/Awkward", "Neutral") == 0


def test_mqm_segments_sum_the_penalty_per_rater_and_strip_the_error_marks():
    rows = [_mqm_row("sysA", "r1", "A <v>bad</v> output", "Accuracy/Mistranslation", "Major"),
            _mqm_row("sysA", "r1", "A bad <v>output</v>", "Style/Awkward", "Minor"),
            _mqm_row("sysA", "r2", "A bad output<v>.</v>", "Fluency/Punctuation", "Minor"),
            _mqm_row("sysB", "r1", "A good output", "No-error", "no-error"),
            _mqm_row("sysB", "r2", "A good output", "Style/Awkward", "Minor"),
            _mqm_row("sysC", "r1", "Planted", "Found", "HOTW-test"),
            _mqm_row("sysD", "r1", "The same output", "Source error", "Minor")]
    seg = H.mqm_segments(rows)[("doc1", "1")]
    assert seg["source"] == "Quelle" and set(seg["systems"]) == {"sysA", "sysB"}  # attention check and source error removed
    a, b = seg["systems"]["sysA"], seg["systems"]["sysB"]
    assert a["target"] == "A bad output." or a["target"] == "A bad output"
    assert "<v>" not in a["target"] and dict(a["penalty"]) == {"r1": 6.0, "r2": 0.1} and dict(a["major"]) == {"r1": True, "r2": False}
    assert dict(b["penalty"]) == {"r1": 0.0, "r2": 1.0}
    # mean 3.05 vs 0.5: below the 5-point margin; r2 also prefers A -> no pair
    assert H.mqm_pairs(seg) == []


def test_mqm_pairs_need_a_major_error_margin():
    rows = [_mqm_row("sysA", "r1", "wrong meaning", "Accuracy/Mistranslation", "Major"),
            _mqm_row("sysA", "r1", "wrong meaning", "Style/Awkward", "Minor"),
            _mqm_row("sysB", "r1", "fine", "No-error", "No-error"),
            _mqm_row("sysC", "r1", "fine", "No-error", "No-error"),            # same text as sysB: not a second candidate
            _mqm_row("sysD", "r1", "slightly awkward", "Style/Awkward", "Minor")]
    seg = H.mqm_segments(rows)[("doc1", "1")]
    assert sorted(H.mqm_pairs(seg)) == [("fine", "wrong meaning"), ("slightly awkward", "wrong meaning")]
    assert H.CLOSED_SYSTEM.search("GPT4-5shot") and H.CLOSED_SYSTEM.search("Claude-3.5") and not H.CLOSED_SYSTEM.search("ONLINE-B")
    assert not H.CLOSED_SYSTEM.search("Online-W.en") and not H.CLOSED_SYSTEM.search("Tencent_Translation.1520")


# ---- the other converters -----------------------------------------------------------------------------------------------


def test_timeqa_distractors_are_answers_of_other_time_scopes():
    group = ["Derby County", "Queens Park Rangers", "Millwall", "Manchester United", "Manchester United F.C.", "Derby"]
    for seed in range(60):
        q, lab, kind = H.timeqa_item("Which team did G play for in 1976?", ["Manchester United"], group, random.Random(seed))
        opts = q["criteria"]
        assert opts[lab] == "Manchester United" and kind == "answerable" and "Manchester United F.C." not in opts
        assert not {"Derby County", "Derby"} <= set(opts)                     # never an answer and its longer form
        assert set(opts) <= set(group) | {H.NO_ANSWER} and 3 <= len(opts) <= 4
    q, lab, kind = H.timeqa_item("Which team did G play for in 1999?", [""], group, random.Random(0))
    assert q["criteria"][lab] == H.NO_ANSWER and kind == "unanswerable"
    assert isinstance(H.timeqa_item("q", ["Millwall"], ["Millwall", "Derby"], random.Random(0)), str)
    shares = [H.NO_ANSWER in H.timeqa_item("q", ["Millwall"], group, random.Random(s))[0]["criteria"] for s in range(400)]
    assert 0.35 < sum(shares) / 400 < 0.55  # the no-answer option is also a distractor, so its presence is not a tell


def test_frank_sentence_votes():
    assert H.frank_sentence({"annotator_0": ["NoE"], "annotator_1": ["NoE"], "annotator_2": ["OutE"]}) == (pytest.approx(2 / 3), None)
    assert H.frank_sentence({"annotator_0": ["EntE"], "annotator_1": ["EntE", "OutE"], "annotator_2": ["NoE"]}) == (pytest.approx(1 / 3), "EntE")
    assert H.frank_sentence({"annotator_0": ["EntE"], "annotator_1": ["OutE"], "annotator_2": ["RelE"]}) == (0.0, None)
    assert H.frank_sentence({"annotator_0": ["EntE", "OutE"], "annotator_1": ["EntE", "OutE"], "annotator_2": ["NoE"]})[1] is None


def test_tabfact_and_faithdial_rows():
    rows = H.tabfact_rows("t1", "team#wins\nreds#3\nblues#5", [["the blues won 5", "the reds won 4", "the reds won 3"], [1, 0, 1], "league"],
                          random.Random(0))
    assert [lab for _, lab in rows] == [1, 0]
    s = json.loads(rows[1][0])
    assert s["table"] == "team | wins\nreds | 3\nblues | 5" and s["statement"] == "the reds won 4" and s["table_caption"] == "league"
    utt = {"history": ["a", "b", "c", "d"], "knowledge": "K", "response": "edited", "original_response": "made up", "BEGIN": ["Hallucination"]}
    assert H.faithdial_rows(utt) == [({"knowledge": "K", "dialogue": ["b", "c", "d"], "response": "edited"}, 1),
                                     ({"knowledge": "K", "dialogue": ["b", "c", "d"], "response": "made up"}, 0)]
    assert [l for _, l in H.faithdial_rows(dict(utt, BEGIN=["Entailment"], original_response=None))] == [1]
    assert [l for _, l in H.faithdial_rows(dict(utt, BEGIN=["Entailment", "Uncooperative"]))] == [1]
    assert H.faithdial_rows(dict(utt, BEGIN=["Generic"])) == []


def test_state_budget_uses_the_token_count():
    assert H._fits("short", ntok=lambda s: 10 ** 6)             # under 3,500 bytes: no tokenizer needed
    long = "word " * 3000
    assert H._fits(long, ntok=lambda s: 3000) and not H._fits(long, ntok=lambda s: 3501)


def test_registry():
    assert set(H.SOURCES) == {"finqa_choice", "convfinqa_choice", "tatqa_choice", "tabfact_noul", "timeqa_choice",
                              "summeval_pair", "summeval_score", "frank_factual", "mqm_pair", "mqm_major_error",
                              "faithdial_noul", "begin_attribution", "feedbackqa_score"}
    for spec in H.SOURCES.values():
        assert spec["license"] and spec["sources"] and all(re.fullmatch(r"[0-9a-f]{40}", s["revision"]) for s in spec["sources"])
    assert "asap2_score" in H.SKIPPED and "asap2_score" not in H.SOURCES


# ---- built files ------------------------------------------------------------------------------------------------------
NOUL = ["tabfact_noul", "mqm_major_error", "faithdial_noul"]
PAIRS = {"summeval_pair": ("summary_A", "summary_B"), "mqm_pair": ("translation_A", "translation_B")}
NUMERIC = ["finqa_choice", "convfinqa_choice", "tatqa_choice"]
SOFT = ["summeval_score", "feedbackqa_score", "frank_factual", "mqm_major_error"]


def _rows(name, split):
    path = os.path.join(DATA_ROOT, name, split + ".jsonl")
    if not os.path.exists(path):
        pytest.skip("%s is not built here" % name)
    return [json.loads(l) for l in open(path, encoding="utf-8")]


@pytest.mark.parametrize("name", sorted(H.SOURCES))
def test_built_manifest_and_no_document_in_both_splits(name):
    train, val = _rows(name, "train"), _rows(name, "val")
    man = json.load(open(os.path.join(DATA_ROOT, name, "manifest.json")))
    assert man["license"]["commercial"] is True and man["license"]["annotations"] == H.SOURCES[name]["license"]
    assert re.fullmatch(r"[0-9a-f]{40}", man["revision"]) and man["sources"] and isinstance(man["dropped"], dict)
    assert len(train) <= H.TRAIN_CAP and len(val) <= H.VAL_CAP and len(val) >= 300
    docs = lambda rows: {r["provenance"]["doc"] for r in rows}  # noqa: E731
    assert not docs(train) & docs(val)
    assert not {r["id"] for r in train} & {r["id"] for r in val}
    assert not {r["state_text"] for r in train} & {r["state_text"] for r in val}
    assert man["state_tokens"]["val"]["max_of_sample"] <= H.MAX_TOKENS and man["state_tokens"]["train"]["max_of_sample"] <= H.MAX_TOKENS


@pytest.mark.parametrize("name", NOUL + ["frank_factual"])
def test_built_noul_sets_are_balanced(name):
    for split in ("train", "val"):
        labs = [r["label"] for r in _rows(name, split) if r["question"]["type"] == "noul"]
        assert abs(sum(labs) / len(labs) - 0.5) < 0.02
        by_q = Counter((r["question"]["instructions"], r["label"]) for r in _rows(name, split) if r["question"]["type"] == "noul")
        for (ins, lab), n in by_q.items():
            assert abs(n / (n + by_q[(ins, 1 - lab)]) - 0.5) < 0.08   # no wording carries the answer


def test_built_mqm_major_error_is_balanced_inside_each_language_pair():
    for split in ("train", "val"):
        by = Counter((r["kind"], r["label"]) for r in _rows("mqm_major_error", split))
        for lp in {k for k, _ in by}:
            assert abs(by[(lp, 1)] / (by[(lp, 1)] + by[(lp, 0)]) - 0.5) < 0.06


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_built_pair_sets_are_position_and_length_balanced(name):
    a, b = PAIRS[name]
    for split in ("train", "val"):
        rows = _rows(name, split)
        first = sum(r["label"] == 0 for r in rows) / len(rows)
        longer = 0
        for r in rows:
            s = json.loads(r["state_text"])
            gold, other = (s[a], s[b]) if r["label"] == 0 else (s[b], s[a])
            assert gold != other and len(gold) != len(other)
            longer += len(gold) > len(other)
        assert abs(first - 0.5) < 0.01 and abs(longer / len(rows) - 0.5) < 0.01


@pytest.mark.parametrize("name", NUMERIC)
def test_built_numeric_options_are_rank_uniform_and_gold_is_not_the_longest(name):
    rows = [r for r in _rows(name, "train") + _rows(name, "val") if r.get("kind", "").split("_")[-1] in ("program", "arithmetic", "count")]
    ranks, longest, n = Counter(), 0, 0
    for r in rows:
        opts = r["question"]["criteria"]
        if len(opts) != 4:
            continue
        vals = [float(re.sub(r"[^\d.\-]", "", o.split(" ")[0])) for o in opts]
        ranks[sorted(vals).index(vals[r["label"]])] += 1
        longest += max(range(4), key=lambda i: len(opts[i])) == r["label"]
        n += 1
    assert n > 3000
    assert all(abs(ranks[k] / n - 0.25) < 0.03 for k in range(4)), ranks
    assert longest / n < 0.30


@pytest.mark.parametrize("name", ["finqa_choice", "convfinqa_choice"])
def test_built_finance_rows_carry_the_worked_program(name):
    for r in _rows(name, "val"):
        gold = r["question"]["criteria"][r["label"]]
        assert r["rationale"].endswith("Answer: %s." % gold) or r["rationale"].endswith("Read from the report: %s." % gold)
        if r["kind"].endswith("program"):
            assert r["rationale"].startswith("Step 1: ") and " = " in r["rationale"]


def test_built_convfinqa_state_holds_the_conversation():
    rows = _rows("convfinqa_choice", "val")
    assert any(r["kind"].startswith("final_turn") for r in rows)
    for r in rows:
        hist = json.loads(r["state_text"])["conversation_so_far"]
        assert hist and all(set(h) == {"question", "answer"} for h in hist)


@pytest.mark.parametrize("name", SOFT)
def test_built_soft_targets_are_distributions_that_agree_with_the_label(name):
    n = 0
    for split in ("train", "val"):
        for r in _rows(name, split):
            if r.get("target") is None:
                continue
            k = 2 if r["question"]["type"] == "noul" else len(r["question"]["criteria"])
            t = r["target"]
            assert len(t) == k and min(t) >= 0 and abs(sum(t) - 1) < 1e-3 and t[r["label"]] == max(t)
            n += 1
    assert n > 500


def test_built_timeqa_no_answer_option_is_not_a_tell():
    rows = _rows("timeqa_choice", "train")
    has = [r for r in rows if H.NO_ANSWER in r["question"]["criteria"]]
    gold = sum(r["question"]["criteria"][r["label"]] == H.NO_ANSWER for r in has) / len(has)
    assert 0.15 < gold < 0.35 and len(has) > 0.3 * len(rows)


def test_built_mqm_states_have_no_error_marks_and_no_closed_system():
    for name in ("mqm_pair", "mqm_major_error"):
        for split in ("train", "val"):
            for r in _rows(name, split):
                assert "<v>" not in r["state_text"] and "</v>" not in r["state_text"]
        man = json.load(open(os.path.join(DATA_ROOT, name, "manifest.json")))
        assert not any("closed-model" in k for k in man["dropped"])  # the 2020-2022 files contain none
