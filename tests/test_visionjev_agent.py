"""Agentic trajectory builder (laya.visionjev_agent) on hand-made runs: chat parsing, scratch-file stripping, and the
structural guarantees of the four tasks (pairs from one issue with one resolved side, per-issue cap, token budget, the
task's test names, length balance; outcome rows with a value target cut at 25 / 50 / 75% and balanced per cell;
next-action options never already in the shown history and never the finish call; step-progress labels only where the
log proves them)."""
import json
import random
from collections import Counter, defaultdict

from laya import visionjev_agent as A


def _call(i, name, **args):
    return {"id": "c%d" % i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_steps_pair_calls_with_replies_and_skip_think():
    traj = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "<issue_description>\nBug in foo\n</issue_description> do it"},
            {"role": "assistant", "content": "", "tool_calls": [_call(0, "think", thought="hmm")]},
            {"role": "tool", "content": "logged", "tool_call_id": "c0"},
            {"role": "assistant", "content": "", "tool_calls": [_call(1, "execute_bash", command="ls  -la", is_input="false")]},
            {"role": "tool", "content": "a.py", "tool_call_id": "c1"}]
    st = A.steps(traj)
    assert st == [{"action": "execute_bash: command=ls -la", "observation": "a.py"}]
    assert A.raw_steps(traj)[0]["args"]["command"] == "ls  -la"
    assert A.issue_text(traj) == "Bug in foo"


def test_strip_scratch_keeps_source_and_real_tests():
    patch = ("diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1 +1 @@\n-x\n+y\n"
             "diff --git a/reproduce_issue.py b/reproduce_issue.py\nnew file mode 100644\n--- /dev/null\n+++ b/reproduce_issue.py\n+print(1)\n"
             "diff --git a/tests/test_core.py b/tests/test_core.py\n--- a/tests/test_core.py\n+++ b/tests/test_core.py\n+def test(): pass\n")
    out = A.strip_scratch(patch)
    assert "pkg/core.py" in out and "tests/test_core.py" in out and "reproduce_issue" not in out
    # created files that are not part of a fix: scripts and notes in the repository root, tool and build leftovers
    for junk in ("test_edge_cases.py", "issue.md", "=1.2.0", ".openhands/TASKS.md", "pkg.egg-info/PKG-INFO", "pkg/core.py.bak",
                 "src/pkg/parser.py.backup", "coverage.xml", "SOLUTION_SUMMARY.md", "sub/debug_it.py"):
        assert A.is_scratch(junk), junk
        new = "diff --git a/%s b/%s\nnew file mode 100644\n--- /dev/null\n+++ b/%s\n+x\n" % (junk, junk, junk)
        assert A.strip_scratch(patch + new) == out, junk
        edited = "diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -1 +1 @@\n-x\n+y\n" % (junk, junk, junk, junk)
        assert junk in A.strip_scratch(patch + edited)  # only files the patch CREATES are dropped
    for real in ("pkg/new_module.py", "tests/test_new.py", "setup.py", "conftest.py", "CHANGELOG.md", "requirements-dev.txt",
                 "docs/changes/123.bugfix.rst", "pyproject.toml", "src/pkg/data/table.json"):
        assert not A.is_scratch(real), real
    assert A._unquote(json.dumps("diff --git a/x b/x")) == "diff --git a/x b/x"


def test_test_names_are_names_only_and_capped():
    names = A.test_names(["tests/test_a.py::test_%d" % i for i in range(30)] + ["tests/test_a.py::test_0"])
    assert len(names) == A.TESTS_MAX and names[0] == "tests/test_a.py::test_0" and len(set(names)) == len(names)
    assert A.test_names("['t::a', 't::b']") == ["t::a", "t::b"]  # the stringified list some dumps carry
    assert A.test_names(["x" * 500]) == ["x" * A.TEST_NAME_MAX] and A.test_names(None) == []


def _runs(n_inst=60, seed=0, per=4, n_steps=14):
    rng = random.Random(seed)
    runs = {}
    for i in range(n_inst):
        rs = []
        for j in range(per):
            steps = [{"action": "execute_bash: command=step %d-%d-%d" % (i, j, k), "observation": "ok"} for k in range(n_steps)]
            steps.append({"action": "finish: message=done", "observation": ""})
            body = "+line\n" * rng.randint(3, 12)
            rs.append({"id": "r%d-%d" % (i, j), "repo": "org/repo%d" % (i % 7), "resolved": int(j % 2 == 0),
                       "patch": "diff --git a/m.py b/m.py\n" + body + "# %d %d\n" % (i, j), "issue": "issue %d" % i,
                       "steps": steps, "license": "MIT", "tests": ["tests/test_m.py::test_%d" % i], "progress": []})
        runs["inst%d" % i] = rs
    return runs


def _ntok(text):
    return len(text) // 4


def test_verify_pairs_have_one_resolved_side_and_balanced_lengths():
    runs = _runs()
    by_patch = {r["patch"]: r for rs in runs.values() for r in rs}
    rows = list(A.verify_pair_rows(runs, random.Random(0), ntok=_ntok))
    assert rows
    longer = defaultdict(Counter)
    for split, state, q, lab, _ in rows:
        s = json.loads(state)
        a, b = by_patch[s["patch_A"]], by_patch[s["patch_B"]]
        assert a["issue"] == b["issue"] and a["resolved"] + b["resolved"] == 1
        assert s["tests_that_must_pass"] == a["tests"] and all("def " not in t for t in s["tests_that_must_pass"])
        gold = (a, b)[lab]
        assert gold["resolved"] == 1
        longer[split][len(gold["patch"]) > len((b, a)[lab]["patch"])] += 1
        a_len, b_len = len(s["patch_A"]), len(s["patch_B"])
        assert max(a_len, b_len) <= A.RATIO_MAX * min(a_len, b_len)
    for split, c in longer.items():  # balanced inside each split (and each of the 3 length-ratio bands), not only overall
        assert abs(c[True] - c[False]) <= 33
    assert 0.35 < Counter(r[3] for r in rows)[0] / len(rows) < 0.65


def test_verify_pairs_per_issue_cap_and_token_budget():
    runs = _runs(n_inst=12, per=12)  # 6 resolved x 6 unresolved patches per issue: up to 36 pairs each
    rows = list(A.verify_pair_rows(runs, random.Random(0), ntok=_ntok))
    per_issue = Counter(json.loads(r[1])["issue"] for r in rows)
    assert max(per_issue.values()) <= A.PAIR_CAP and max(per_issue.values()) > 3
    assert len({r[1] for r in rows}) == len(rows)
    use = Counter()
    for r in rows:
        s = json.loads(r[1])
        use[s["patch_A"]] += 1
        use[s["patch_B"]] += 1
    assert max(use.values()) <= 3  # the pairs of an issue are spread over its patches
    # patches may be long (<= PATCH_MAX chars) only while the whole row fits the token budget
    big = _runs(n_inst=6)
    for rs in big.values():
        for r in rs:
            r["patch"] += "+x = 1\n" * 1000  # ~7,000 chars: under PATCH_MAX, but two of them are over the row budget
    assert all(len(r["patch"]) <= A.PATCH_MAX for rs in big.values() for r in rs)
    assert list(A.verify_pair_rows(big, random.Random(0), ntok=_ntok)) == []
    assert A.PAIR_STATS["drop_row_over_%d_tokens" % A.ROW_TOKENS] > 0
    assert len(list(A.verify_pair_rows(big, random.Random(0), ntok=lambda t: len(t) // 8))) > 0
    for rs in big.values():
        for r in rs:
            r["patch"] += "+y = 2\n" * 400  # now over PATCH_MAX
    assert list(A.verify_pair_rows(big, random.Random(0), ntok=lambda t: 1)) == []


def test_next_action_options_are_new_and_never_finish():
    rows = list(A.next_action_rows(_runs(), random.Random(0)))
    assert rows
    for split, state, q, lab, _ in rows:
        s = json.loads(state)
        shown = {l.split("action: ", 1)[1] for l in s["recent_steps"].splitlines() if " action: " in l}
        assert len(set(q["criteria"])) == 4
        assert not shown & set(q["criteria"]) and not any(o.startswith("finish") for o in q["criteria"])
        k = int(q["criteria"][lab].rsplit("-", 1)[1])
        assert k == s["steps_so_far"]  # the gold is the step right after the shown history


def test_gold_repeating_a_shown_action_is_skipped():
    runs = _runs(20)
    for rs in runs.values():
        for r in rs:
            for k in range(0, 14, 2):  # every other step repeats the previous one (re-running the same command)
                r["steps"][k + 1] = dict(r["steps"][k])
    for split, state, q, lab, _ in A.next_action_rows(runs, random.Random(0)):
        assert q["criteria"][lab] not in json.loads(state)["recent_steps"]


def test_outcome_rows_value_target_three_cut_points_and_cell_balance():
    runs = _runs(n_inst=200, n_steps=23)  # 24 steps per run with the finish call
    for rs in runs.values():
        rs.append(dict(rs[0], id=rs[0]["id"] + "-copy"))  # a second run with the same final patch and steps
    rows = list(A.outcome_rows(runs, random.Random(0), caps={"train": 10 ** 6, "val": 10 ** 6}))
    assert len({r[1] for r in rows}) == len(rows)
    cells = defaultdict(Counter)
    cuts = Counter()
    for split, state, q, lab, _, extra in rows:
        s = json.loads(state)
        assert extra == {"value": float(lab)} and lab in (0, 1)  # the value-head target is the outcome
        if "recent_steps" in s:
            assert "finish" not in s["recent_steps"]
            assert s["steps_so_far"] in {A.prefix_cut(24, f) for f in A.FRACTIONS} == {6, 12, 18}
            cuts[s["steps_so_far"]] += 1
            cells[(split, "prefix", s["steps_so_far"])][lab] += 1
        else:
            cells[(split, "final", A._bin(len(s["final_patch"])))][lab] += 1
    assert set(cuts) == {6, 12, 18} and min(cuts.values()) > 0.5 * max(cuts.values())  # the three points, about evenly
    assert any(k[1] == "final" for k in cells)
    assert all(c[0] == c[1] for c in cells.values())  # balanced inside every (split, view, bin)
    # a cap scales every cell down and keeps the balance
    capped = list(A.outcome_rows(runs, random.Random(0), caps={"train": 100, "val": 40}))
    n = Counter((r[0], r[3]) for r in capped)
    assert n[("train", 0)] == n[("train", 1)] <= 50 and n[("val", 0)] == n[("val", 1)] <= 20 and n[("train", 0)] > 25
    # runs too short to be cut 3 steps before their end give no prefix rows
    short = list(A.outcome_rows(_runs(n_inst=40, n_steps=8), random.Random(0)))
    assert short and all("final_patch" in json.loads(r[1]) for r in short)


# ---- step progress ------------------------------------------------------------------------------------------------------

FOOT = "\n[The command completed with exit code %d.]\n[Current working directory: /w/r]\n[Command finished with exit code %d]"


def _bash(cmd, out, code=0):
    return {"name": "execute_bash", "args": {"command": cmd}, "action": "execute_bash: command=" + cmd,
            "observation": out + FOOT % (code, code)}


def _edit(path, old, new, ok=True):
    obs = "The file %s has been edited. Here's the result of running `cat -n` on a snippet" % path if ok else \
        "ERROR:\nNo replacement was performed, old_str `%s` did not appear verbatim in %s." % (old, path)
    action = "str_replace_editor: command=str_replace, path=%s, old_str=%s, new_str=%s" % (path, old, new)
    return {"name": "str_replace_editor", "args": {"command": "str_replace", "path": path, "old_str": old, "new_str": new},
            "action": A._cut(action, A.ACT_MAX), "observation": obs}


def _editor(command, path, obs):
    return {"name": "str_replace_editor", "args": {"command": command, "path": path},
            "action": "str_replace_editor: command=%s, path=%s" % (command, path), "observation": obs}


PATCH = ("diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1,2 +1,3 @@\n def f(x):\n"
         "-    return x\n+    x = normalise(x)\n+    return x + 1\n"
         "diff --git a/tests/test_new.py b/tests/test_new.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/test_new.py\n"
         "@@ -0,0 +1 @@\n+assert f(1) == 2  # agent test\n")
TRACE = "Traceback (most recent call last):\n  File \"t.py\", line 3\nAssertionError: expected 2 got 1"


def _labels(rs, resolved=1, patch=PATCH):
    return {p["t"]: (p["label"], p["kind"]) for p in A.progress_labels(rs, resolved, patch)}


def test_progress_edit_surviving_into_the_final_patch_is_progress_only_in_a_resolved_run():
    good = _edit("/w/r/pkg/core.py", "    return x", "    x = normalise(x)\n    return x + 1")
    rs = [_bash("ls", "core.py"), good, _bash("python -m pytest -q", "1 passed")]
    assert _labels(rs) == {1: (1, "edit_survives")}
    assert _labels(rs, resolved=0) == {}  # an unresolved run proves nothing about its edits
    # only part of the edit is in the final patch: not proven
    assert _labels([_edit("/w/r/pkg/core.py", "    return x", "    x = normalise(x)\n    return x + 100")]) == {}
    # a file the agent created itself (editor or shell): every line "survives", which proves nothing
    made = _editor("create", "/w/r/pkg/core.py", "File created successfully at: /w/r/pkg/core.py")
    assert _labels([made, good]) == {}
    assert _labels([_edit("/w/r/tests/test_new.py", "assert f(1) == 1", "assert f(1) == 2  # agent test")]) == {}
    # a failed edit and an edit adding only trivial lines are not labelled
    assert _labels([_edit("/w/r/pkg/core.py", "    return x", "    x = normalise(x)\n    return x + 1", ok=False)]) == {}
    assert _labels([_edit("/w/r/pkg/core.py", "    return x", "    pass")]) == {}


def test_progress_reverted_edits_are_no_progress():
    bad = _edit("/w/r/pkg/core.py", "    return x", "    print('DEBUG value', x)\n    return x")
    inverse = _edit("/w/r/pkg/core.py", "    print('DEBUG value', x)\n    return x", "    return x")
    for resolved in (0, 1):  # a revert is proven whatever the run's outcome
        assert _labels([bad, _bash("python t.py", "ok"), inverse], resolved) == {0: (0, "edit_reverted")}
    undo = _editor("undo_edit", "/w/r/pkg/core.py", "Last edit to /w/r/pkg/core.py undone successfully.")
    assert _labels([bad, undo], 0) == {0: (0, "edit_reverted")}
    other = _edit("/w/r/pkg/core.py", "def g():", "def g():  # another region of the file")
    assert _labels([bad, other, undo], 0) == {1: (0, "edit_reverted")}  # the undo takes back the later edit, not `bad`
    for cmd in ("cd /w/r && git checkout pkg/core.py", "git checkout HEAD -- pkg/core.py", "git checkout -- .",
                "git restore pkg/core.py", "cd /w/r && git reset --hard"):
        assert _labels([bad, _bash(cmd, "")], 0) == {0: (0, "edit_reverted")}, cmd
    assert _labels([bad, _bash("git checkout -b fix", "")], 0) == {}
    assert _labels([bad, _bash("git checkout pkg/core.py", "error: pathspec", code=1)], 0) == {}
    assert _labels([bad, _bash("git checkout pkg/other.py", "")], 0) == {}
    # reverted, then the same lines put back and kept in the final patch: the first edit is not proven either way
    # (its lines are in the final patch), the revert was itself undone, the last edit survives
    good = _edit("/w/r/pkg/core.py", "    return x", "    x = normalise(x)\n    return x + 1")
    back = _edit("/w/r/pkg/core.py", "    x = normalise(x)\n    return x + 1", "    return x")
    assert _labels([good, back, good], 1) == {1: (0, "edit_reverted"), 2: (1, "edit_survives")}


def test_progress_repeated_failures_and_newly_passing_tests():
    fail = _bash("cd /w/r && python -m pytest tests/test_core.py", TRACE, code=1)
    ok = _bash("cd /w/r && python -m pytest tests/test_core.py", "1 passed in 0.1s")
    view = _editor("view", "/w/r/pkg/core.py", "Here's the result of running `cat -n` on /w/r/pkg/core.py")
    # the same command with the same failing output, the earlier one inside the shown history
    assert _labels([fail, view, fail], 0) == {2: (0, "repeat_same_error")}
    assert _labels([fail] + [view] * A.HIST_STEPS + [fail], 0) == {}  # the first failure is no longer shown
    # a different command that fails with the identical error text right after
    assert _labels([fail, _bash("cd /w/r && python t.py", TRACE, code=1)], 0) == {1: (0, "same_error_as_previous")}
    assert _labels([fail, view, _bash("cd /w/r && python t.py", TRACE, code=1)], 0) == {}
    # two different searches that both find nothing share no error text
    assert _labels([_bash("grep -n foo a.py", "", code=1), _bash("grep -n bar a.py", "", code=1)], 0) == {}
    # the same editor error twice in a row
    err = _edit("/w/r/pkg/core.py", "    return y", "    return y + 1", ok=False)
    assert _labels([err, err], 0) == {1: (0, "same_error_as_previous")}
    # failed at its last use in the shown history, passes now: progress in a resolved run only
    assert _labels([fail, view, ok], 1) == {2: (1, "test_newly_passes")}
    assert _labels([fail, view, ok], 0) == {}
    assert _labels([ok, view, ok], 1) == {} and _labels([fail, ok, ok], 1) == {1: (1, "test_newly_passes")}
    # a test command that exits 0 through a pipe but prints a failure is still a failure
    piped = _bash("python -m pytest tests -q | tail -5", "FAILED tests/test_core.py::test_f - AssertionError: 1 != 2")
    assert _labels([piped, view, piped], 0) == {2: (0, "repeat_same_error")}
    assert _labels([_bash("grep -rn Traceback .", "a.py:1: Traceback (most recent call last)"), view,
                    _bash("grep -rn Traceback .", "a.py:1: Traceback (most recent call last)")], 0) == {}
    # a poll of a running process (no exit code) is not a result
    wait = {"name": "execute_bash", "args": {"command": "python serve.py"}, "action": "execute_bash: command=python serve.py",
            "observation": "[The command has no new output after 30 seconds.]"}
    assert _labels([wait, wait], 0) == {}


def _progress_runs(n_inst=80):
    runs = {}
    for i in range(n_inst):
        rs = []
        for j in range(2):
            raw = [_bash("ls pkg%d" % k, "core.py") for k in range(3)]
            raw.append(_bash("python t%d.py" % i, TRACE + " %d" % i, code=1))
            if j == 0:  # resolved: a kept edit, then the script passes
                raw.append(_edit("/w/r/pkg/core.py", "    return x", "    x = normalise(x)  # %d\n    return x + 1" % i))
                raw.append(_bash("python t%d.py" % i, "all good"))
            else:  # unresolved: a debug edit later reverted, the script fails the same way
                raw.append(_edit("/w/r/pkg/core.py", "    return x", "    print('DEBUG %d', x)\n    return x" % i))
                raw.append(_bash("python t%d.py" % i, TRACE + " %d" % i, code=1))
                raw.append(_edit("/w/r/pkg/core.py", "    print('DEBUG %d', x)\n    return x" % i, "    return x"))
            patch = PATCH.replace("normalise(x)", "normalise(x)  # %d" % i) if j == 0 else ""
            rs.append({"id": "p%d-%d" % (i, j), "repo": "org/repo%d" % (i % 9), "resolved": int(j == 0), "patch": patch,
                       "issue": "issue %d" % i, "steps": [{"action": s["action"], "obs": A._cut(s["observation"])} for s in raw],
                       "license": "MIT", "tests": [], "progress": A.progress_labels(raw, int(j == 0), patch)})
        runs["inst%d" % i] = rs
    return runs


def test_step_progress_rows_show_the_candidate_and_balance_each_tool():
    rows = list(A.step_progress_rows(_progress_runs(), random.Random(0), caps={"train": 10 ** 6, "val": 10 ** 6}))
    assert rows and len({r[1] for r in rows}) == len(rows)
    cells = defaultdict(Counter)
    kinds = Counter()
    for split, state, q, lab, _, extra in rows:
        s = json.loads(state)
        cand, t = s["candidate_step"], s["steps_so_far"]
        assert q == {"type": "noul", "instructions": A.PROGRESS_Q, "criteria": None}
        shown = [int(l.split()[1]) for l in s["recent_steps"].splitlines() if " action: " in l]
        assert shown == list(range(max(1, t - A.HIST_STEPS + 1), t + 1))  # the history stops right before the candidate
        tool = "edit" if cand["action"].startswith("str_replace_editor") else "shell"
        assert ("new_str" in cand) == (tool == "edit") and cand["result"]
        assert (extra["kind"] in ("edit_survives", "test_newly_passes")) == bool(lab)
        if extra["kind"] == "repeat_same_error":  # the earlier identical failure is visible in the history
            assert cand["result"] in s["recent_steps"] and cand["action"] in s["recent_steps"]
        cells[(split, tool, A._bin(t))][lab] += 1
        kinds[extra["kind"]] += 1
    assert set(kinds) == {"edit_survives", "test_newly_passes", "edit_reverted", "repeat_same_error"}
    # balanced inside every (split, tool, step bin): neither the tool nor the step reached answers the question
    assert all(c[0] == c[1] for c in cells.values()) and {k[1] for k in cells} == {"edit", "shell"}
    # a candidate whose history and step also occur in a run with the other outcome is not used
    runs = _progress_runs(10)
    for rs in runs.values():
        twin = dict(rs[0], id=rs[0]["id"] + "-twin", resolved=0, patch="", progress=[])
        twin["steps"] = rs[0]["steps"][:5]  # the same first steps (up to the kept edit), then it stopped
        rs.append(twin)
    kinds = Counter(r[5]["kind"] for r in A.step_progress_rows(runs, random.Random(0), caps={"train": 10 ** 6, "val": 10 ** 6}))
    assert "edit_survives" not in kinds and A.PROGRESS_STATS["skipped_same_prefix_other_outcome"] == 10
    capped = list(A.step_progress_rows(_progress_runs(), random.Random(0), caps={"train": 40, "val": 10}))
    n = Counter((r[0], r[3]) for r in capped)
    assert n[("train", 0)] == n[("train", 1)] <= 20 and n[("val", 0)] == n[("val", 1)] <= 5
