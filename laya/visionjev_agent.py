"""Agentic trajectory decisions (Hypothesis/V6.2.md section 4, V6.3.md section 6) from open-weight code-agent runs with
test outcomes.

Source (licence review 2026-09-29, V6.2 section 4.5): ``nebius/SWE-rebench-openhands-trajectories`` (CC BY 4.0; runs of
Qwen3-Coder-480B-A35B-Instruct, Apache-2.0, in OpenHands 0.54 on real GitHub issues; ``resolved`` from the tests).
Kept only for repositories whose own licence is permissive (``license_name`` of nebius/SWE-rebench, as in
``visionjev_swe``) and never for a SWE-bench repository. Split by *repository* with ``visionjev_swe._split``, so the val
repositories are the same as for the ``swe_*`` sets and val measures new codebases.

* ``agent_verify_pair``   (choice, 2) issue + the names of the tests the fix must make pass (FAIL_TO_PASS of the task:
                          names only, never test code or the reference patch) + two final patches from runs on the same
                          issue, one resolved and one not -> which one solves it? At most 8 pairs per issue, patches up
                          to 8,000 chars as long as the whole row stays within 3,800 Qwen3-VL tokens, the longer patch at
                          most 3x the shorter. Length-balanced per split and length-ratio band (the resolved patch is
                          the longer one half of the time); files the agent created as scratch (reproduce / debug /
                          ad-hoc test scripts, notes, tool leftovers: ``is_scratch``) are stripped.
* ``agent_outcome``       (noul + ``value``) will this run resolve the issue? Four views, equally likely: the final
                          patch, or the run so far cut at 25%, 50% or 75% of its steps (the last actions, observations
                          cut). Classes balanced inside every (split, view, length / step bin); ``value`` = 1.0 resolved
                          / 0.0 not is the value-head target (``laya.vlm_train``: BCE on rows that carry ``value``).
* ``agent_next_action``   (choice, 4) the run so far at step t of a *resolved* run -> which tool call comes next? The
                          others are calls from other runs on the same issue and from later steps of this run, never an
                          action already in the shown history; the gold is never from the last 2 steps (the finish).
* ``agent_step_progress`` (noul) the recent history + ONE candidate step (action and its result) -> did it make progress?
                          Only what the log proves is labelled (``progress_labels``):
                          no  = an edit later reverted (exact inverse edit, ``undo_edit`` or ``git checkout`` of the
                                file, and none of its lines in the final patch); the same command repeated inside the
                                shown history with the same failing output; an error identical to the previous step's.
                          yes = in a resolved run: an edit whose added lines all survive into the final patch and is
                                never reverted; a test / script command that failed earlier in the shown history and now
                                passes.
                          Classes balanced inside every (split, tool, step bin).

    python -m laya.visionjev_agent all --parquet data/raw/swe_rebench_traj/trajectories.parquet
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterator, List, Optional

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT
from .visionjev_swe import COPYLEFT, PERMISSIVE, _split, _tests

PARQUET = os.path.join(os.path.dirname(DATA_ROOT), "raw", "swe_rebench_traj", "trajectories.parquet")
ISSUE_MAX, PATCH_MAX, OBS_MAX, ACT_MAX, HIST_STEPS = 1500, 8000, 300, 220, 6
FINAL_PATCH_MAX = 4000            # agent_outcome, final-patch view (unchanged from round 2)
PAIR_CAP, ROW_TOKENS = 8, 3800    # verifier pairs per issue; whole-row budget (the training budget is 4,096)
TESTS_MAX, TEST_NAME_MAX = 12, 160
FRACTIONS = (0.25, 0.5, 0.75)     # agent_outcome prefix views
TOKENIZER = "Qwen/Qwen3-VL-4B-Instruct"
SCRATCH = re.compile(r"(^|/)(reproduc|debug|verify|check_|test_fix|test_issue|test_reproduc|comprehensive|edge_case|"
                     r"final_|simple_|demo|scratch|temp_|tmp_)[^/]*$", re.I)
JUNK = re.compile(r"(\.(bak|backup|orig|rej|tmp|broken|fixed|swp|log|out|err|pyc)$|~$|(^|/)[^/]*\.egg-info/|^\.openhands/|"
                  r"^\.eggs/|(^|/)__pycache__/|(^|/)\.pytest_cache/|(^|/)(coverage\.xml|\.coverage)$)", re.I)
ROOT_FILE = re.compile(r"^[^/]*\.(py|md|txt|sh|json|diff|patch|xml|html|csv|ipynb|yaml|yml)$|^=[^/]*$", re.I)
ROOT_KEEP = re.compile(r"^(setup\.py|conftest\.py|noxfile\.py|tasks\.py|manage\.py|fabfile\.py|(readme|changelog|changes|news|"
                       r"history|license|licence|authors|contributing|manifest|requirements)[^/]*)$", re.I)
RATIO_MAX = 3.0                   # verifier pairs: the longer patch is at most this many times the shorter one


# ---- reading --------------------------------------------------------------------------------------------------------


def _unquote(p: str) -> str:
    p = p or ""
    if p.startswith('"'):
        try:
            return json.loads(p)
        except ValueError:
            pass
    return p


def is_scratch(path: str) -> bool:
    """A file the agent created that is not part of a fix: a reproduction / debug / ad-hoc test script by name
    (``SCRATCH``); tool and build leftovers (``JUNK``: the OpenHands task list, egg-info, backups, logs, caches); any
    script, note or data file dropped in the repository root (``ROOT_FILE``) other than a usual project file."""
    return bool(SCRATCH.search(path) or JUNK.search(path) or (ROOT_FILE.match(path) and not ROOT_KEEP.match(path)))


def strip_scratch(patch: str) -> str:
    """Drop files the agent *created* as scratch (``is_scratch``): graders ignore them and they differ systematically
    between runs (2026-09-30: `.openhands/TASKS.md` alone was in 9,350 final patches, and root-level ``test_*.py``
    scripts pushed 42% of the candidate pairs over the patch-length limit)."""
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    keep = []
    for part in parts:
        m = re.match(r"diff --git a/(\S+) b/", part)
        if m and "\nnew file mode" in part[:300] and is_scratch(m.group(1)):
            continue
        keep.append(part)
    return "".join(keep).strip()


def _args(call: dict) -> dict:
    args = (call.get("function") or {}).get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {"raw": args}
    return args if isinstance(args, dict) else {}


def _action(call: dict) -> str:
    """A tool call as one short line: ``name: key=value, ...`` (long values cut)."""
    f = call.get("function") or {}
    parts = []
    for k, v in _args(call).items():
        if k in ("is_input", "security_risk", "thought") or v in (None, ""):
            continue
        v = re.sub(r"\s+", " ", str(v)).strip()
        parts.append("%s=%s" % (k, v))
    return ("%s: %s" % (f.get("name", "?"), ", ".join(parts)))[:ACT_MAX]


def raw_steps(traj: List[dict]) -> List[dict]:
    """[{name, args, action, observation}] from the chat: one entry per assistant tool call (``think`` calls skipped),
    with the full tool reply that answers it ("" when the log has none, e.g. the finish call)."""
    replies = {m.get("tool_call_id"): (m.get("content") or "") for m in traj if m.get("role") == "tool"}
    out = []
    for m in traj:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            name = (c.get("function") or {}).get("name")
            if name == "think":
                continue  # a logged thought, not an action on the environment
            out.append({"name": name, "args": _args(c), "action": _action(c), "observation": replies.get(c.get("id"), "")})
    return out


def steps(traj: List[dict]) -> List[dict]:
    """[{action, observation}]: ``raw_steps`` without the parsed call."""
    return [{"action": s["action"], "observation": s["observation"]} for s in raw_steps(traj)]


def issue_text(traj: List[dict]) -> str:
    """The task statement: the first user message (OpenHands wraps the issue in it)."""
    for m in traj:
        if m.get("role") == "user":
            t = m.get("content") or ""
            mm = re.search(r"<issue_description>(.*?)</issue_description>", t, re.S)
            return (mm.group(1) if mm else t).strip()[:ISSUE_MAX]
    return ""


def _rebench_instances():
    """Rows of nebius/SWE-rebench for permissive repositories outside SWE-bench (same rule as visionjev_swe)."""
    sb_repos = {r["repo"] for r in ts._ds("princeton-nlp/SWE-bench", split="test")} | {"pylint-dev/pylint"}
    for split in ("test", "filtered"):
        for r in ts._ds("nebius/SWE-rebench", split=split).select_columns(["instance_id", "repo", "license_name", "FAIL_TO_PASS"]):
            lic = r.get("license_name") or ""
            if r["repo"] in sb_repos or not PERMISSIVE.search(lic) or COPYLEFT.search(lic):
                continue
            yield r, lic


def permissive_instances() -> Dict[str, str]:
    """instance_id -> licence name, for permissive repositories outside SWE-bench."""
    return {r["instance_id"]: lic for r, lic in _rebench_instances()}


def test_names(fail_to_pass) -> List[str]:
    """The task's failing tests as shown in a state: NAMES only (ids such as ``tests/test_x.py::test_y``), at most
    ``TESTS_MAX`` of them, each cut to ``TEST_NAME_MAX`` chars."""
    names = [str(t).strip()[:TEST_NAME_MAX] for t in _tests(fail_to_pass) if str(t).strip()]
    return list(dict.fromkeys(names))[:TESTS_MAX]


test_names.__test__ = False  # not a pytest test


def instance_tests() -> Dict[str, List[str]]:
    """instance_id -> the names of its FAIL_TO_PASS tests (the tests the fix must make pass)."""
    return {r["instance_id"]: test_names(r.get("FAIL_TO_PASS")) for r, _ in _rebench_instances()}


def _cut(text: str, limit: int = OBS_MAX) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit // 2] + " ... " + text[-limit // 2:]


def _obs(step: dict) -> str:
    """The observation as shown in a state (``load_runs`` stores it already cut, to keep the runs small in RAM)."""
    return step["obs"] if "obs" in step else _cut(step["observation"])


def load_runs(parquet: str = PARQUET, allowed: Optional[Dict[str, str]] = None, limit: int = 0,
              tests: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[dict]]:
    """instance_id -> [run]; a run is {id, repo, resolved, patch, issue, steps, license, tests, progress}. ``steps`` are
    {action, obs} with the observation already cut for display; ``progress`` holds the step-progress labels derived
    from the full log (``progress_labels``). Streams the parquet in batches."""
    import pyarrow.parquet as pq

    runs = defaultdict(list)
    n = 0
    pf = pq.ParquetFile(parquet)
    cols = ["trajectory_id", "instance_id", "repo", "trajectory", "model_patch", "resolved"]
    for batch in pf.iter_batches(batch_size=128, columns=cols):
        for r in batch.to_pylist():
            if allowed is not None and r["instance_id"] not in allowed:
                continue
            traj = r["trajectory"] or []
            rs = raw_steps(traj)
            patch = _unquote(r["model_patch"])
            resolved = int(r["resolved"] or 0)
            runs[r["instance_id"]].append({
                "id": r["trajectory_id"], "repo": r["repo"], "resolved": resolved, "patch": strip_scratch(patch),
                "issue": issue_text(traj), "steps": [{"action": s["action"], "obs": _cut(s["observation"])} for s in rs],
                "license": (allowed or {}).get(r["instance_id"], ""), "tests": (tests or {}).get(r["instance_id"], []),
                "progress": progress_labels(rs, resolved, patch)})
            n += 1
            if limit and n >= limit:
                return runs
    return runs


# ---- step progress: what the log proves ---------------------------------------------------------------------------------

EXIT = re.compile(r"\[Command finished with exit code (-?\d+)\]")
ERR_TEXT = re.compile(r"Traceback \(most recent call last\)|\b[1-9]\d* (?:failed|errors?)\b|\bFAILED\b|\b\w*(?:Error|Exception):")
TEST_CMD = re.compile(r"^(?:cd \S+ *(?:&&|;) *)?(?:\w+=\S+ +)*(?:timeout \d+ +)?(?:python3?(?:\.\d+)? |pytest\b|py\.test\b|tox\b|nox\b|"
                      r"nosetests\b|npm (?:run )?test\b|yarn test\b|go test\b|cargo test\b|make\b|jest\b|mocha\b|phpunit\b|"
                      r"rspec\b|mvn\b|\./gradlew\b)")
GIT_PATHS = re.compile(r"git (?:checkout|restore)((?: +[^\s;&|]+)+)")
GIT_RESET = re.compile(r"git reset --hard(?: +HEAD)?\s*(?:$|[;&|])")
EDIT_LINE_MIN = 8  # an added line shorter than this ("pass", "}", "else:") identifies nothing


def _exit_code(obs: str) -> Optional[int]:
    m = EXIT.findall(obs or "")
    return int(m[-1]) if m else None


def _cmd(step: dict) -> str:
    return re.sub(r"\s+", " ", str(step["args"].get("command") or "")).strip()


def _is_bash(step: dict) -> bool:
    return step["name"] == "execute_bash" and str(step["args"].get("is_input", "false")).strip().lower() != "true" \
        and bool(_cmd(step))


def _failed(step: dict) -> bool:
    """The step's result is an error: a shell command that finished with a non-zero exit code (or a test / script
    command that exits 0, e.g. through a pipe, but prints a failure), an editor call answered with ``ERROR``."""
    obs = step["observation"]
    if step["name"] == "str_replace_editor":
        return obs.startswith("ERROR")
    if not _is_bash(step):
        return False
    code = _exit_code(obs)
    if code is None or code == -1:  # still running / timed out: not a result
        return False
    return code != 0 or bool(TEST_CMD.match(_cmd(step)) and ERR_TEXT.search(obs))


def _edit(step: dict) -> Optional[dict]:
    """{path, old, new, added} of a *successful* ``str_replace`` edit; ``added`` = its distinctive new lines."""
    a = step["args"]
    if step["name"] != "str_replace_editor" or a.get("command") != "str_replace":
        return None
    obs = step["observation"]
    if not obs.startswith("The file") or "has been edited" not in obs[:400]:
        return None
    old, new, path = str(a.get("old_str") or ""), str(a.get("new_str") or ""), str(a.get("path") or "")
    had = {l.strip() for l in old.splitlines()}
    added = list(dict.fromkeys(l.strip() for l in new.splitlines() if len(l.strip()) >= EDIT_LINE_MIN and l.strip() not in had))
    return {"path": path, "old": old, "new": new, "added": added}


def patch_added_lines(patch: str) -> Dict[str, Optional[set]]:
    """file -> the stripped lines a unified diff adds to it; None for a file the diff creates (a file the agent made:
    every line of it "survives", which proves nothing)."""
    out = {}
    for part in re.split(r"(?m)^(?=diff --git )", patch or ""):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", part)
        if m:
            new = "\nnew file mode" in part[:300]
            out[m.group(2)] = None if new else {l[1:].strip() for l in part.splitlines()
                                                if l.startswith("+") and not l.startswith("+++")}
    return out


FOOTER = re.compile(r"\[(?:The command completed with exit code|Current working directory|Python interpreter|"
                    r"Command finished with exit code)[^\]]*\]")


def _error_text(step: dict) -> str:
    """The error message itself (the shell footer removed): two steps "fail the same way" only when this is real text,
    not when two different searches both find nothing."""
    return FOOTER.sub("", step["observation"]).strip()


def _reverted_at(rs: List[dict], i: int, e: dict) -> Optional[int]:
    """The later step that restores the region edit ``i`` changed, or None: an exact inverse ``str_replace`` (or one that
    puts the old text back over all of the edit's lines), an ``undo_edit`` of this very edit, or a git checkout /
    restore / hard reset of the file."""
    touched = False  # another successful edit of the same file since step i
    for j in range(i + 1, len(rs)):
        s = rs[j]
        a = s["args"]
        if s["name"] == "str_replace_editor" and str(a.get("path") or "") == e["path"]:
            if a.get("command") == "undo_edit" and s["observation"].startswith("Last edit to"):
                if not touched:
                    return j
                touched = True
                continue
            e2 = _edit(s)
            if e2 is not None:
                old2 = {l.strip() for l in e2["old"].splitlines()}
                if e2["new"] == e["old"] and e["old"].strip() and (e2["old"] == e["new"] or all(l in old2 for l in e["added"])):
                    return j
                touched = True
            elif a.get("command") in ("create", "insert") and not s["observation"].startswith("ERROR"):
                touched = True
        elif _is_bash(s) and _exit_code(s["observation"]) == 0:
            cmd = _cmd(s)
            if GIT_RESET.search(cmd):
                return j
            for m in GIT_PATHS.finditer(cmd):
                for tok in m.group(1).split():
                    tok = tok.strip("'\"")
                    if tok == "." or (tok not in ("--", "HEAD") and not tok.startswith("-")
                                      and e["path"].endswith("/" + tok.lstrip("./")) and tok.lstrip("./")):
                        return j
    return None


def progress_labels(rs: List[dict], resolved: int, patch: str) -> List[dict]:
    """Step-progress labels for the steps of one run where the log proves one: [{t, label, kind, tool, cand}] (``t`` is
    the step index in ``raw_steps`` / ``steps``; ``cand`` the candidate step as shown). Everything else stays unlabelled.

    label 0  ``repeat_same_error``       the same shell command as a step inside the shown history (the last
                                         ``HIST_STEPS``), with the same failing output
             ``same_error_as_previous``  an error whose text is identical to the previous step's error
             ``edit_reverted``           a successful edit of a non-scratch file that a later step reverts, and none of
                                         whose added lines is in the final patch
    label 1  (resolved runs only)
             ``edit_survives``           a successful edit of a non-scratch file, never reverted, all of whose added
                                         lines are in the final patch
             ``test_newly_passes``       a test / script command that failed at its last use inside the shown history
                                         and now exits 0 without a failure in its output
    """
    out = []
    added_by_file = None
    created = set()  # files the agent created in this run: its own scripts and tests, not the repository's code
    for i, s in enumerate(rs):
        lab = kind = None
        obs = s["observation"]
        same_as_previous = bool(i and _failed(s) and _failed(rs[i - 1]) and rs[i - 1]["observation"] == obs
                                and len(_error_text(s)) >= 30)
        if s["name"] == "str_replace_editor" and s["args"].get("command") == "create" and not obs.startswith("ERROR"):
            created.add(str(s["args"].get("path") or ""))
        if _is_bash(s):
            cmd = _cmd(s)
            same = [j for j in range(max(0, i - HIST_STEPS), i) if _is_bash(rs[j]) and _cmd(rs[j]) == cmd]
            if _failed(s):
                if any(rs[j]["observation"] == obs for j in same):
                    lab, kind = 0, "repeat_same_error"
                elif same_as_previous:
                    lab, kind = 0, "same_error_as_previous"
            elif resolved and same and _failed(rs[same[-1]]) and TEST_CMD.match(cmd) and _exit_code(obs) == 0 \
                    and not ERR_TEXT.search(obs):
                lab, kind = 1, "test_newly_passes"
        elif s["name"] == "str_replace_editor":
            if _failed(s):
                if same_as_previous:
                    lab, kind = 0, "same_error_as_previous"
            else:
                e = _edit(s)
                if e is not None and e["added"] and e["path"] not in created and not SCRATCH.search(e["path"]) \
                        and not JUNK.search(e["path"]):
                    if added_by_file is None:
                        added_by_file = patch_added_lines(patch)
                    mine = [lines for f, lines in added_by_file.items() if e["path"].endswith("/" + f)]
                    if None not in mine:  # not a file this run created by other means (shell)
                        final = set().union(*mine) if mine else set()
                        rev = _reverted_at(rs, i, e)
                        if rev is not None and not any(l in final for l in e["added"]):
                            lab, kind = 0, "edit_reverted"
                        elif rev is None and resolved and all(l in final for l in e["added"]):
                            lab, kind = 1, "edit_survives"
        if lab is not None:
            out.append({"t": i, "label": lab, "kind": kind, "tool": "edit" if s["name"] == "str_replace_editor" else "shell",
                        "cand": _candidate(s)})
    return out


def _cut_code(text: str, limit: int) -> str:
    """Head + tail of a code fragment, line structure kept."""
    return text if len(text) <= limit else text[: limit // 2] + "\n...\n" + text[-limit // 2:]


def _candidate(step: dict) -> dict:
    """The candidate step as shown: the call (an edit with its old and new text, a command in full up to 600 chars)."""
    a = step["args"]
    if step["name"] == "str_replace_editor" and a.get("command") == "str_replace":
        return {"action": "str_replace_editor: command=str_replace, path=%s" % a.get("path"),
                "old_str": _cut_code(str(a.get("old_str") or ""), 700), "new_str": _cut_code(str(a.get("new_str") or ""), 1000)}
    if step["name"] == "execute_bash":
        return {"action": "execute_bash: command=" + _cut(str(a.get("command") or ""), 600)}
    return {"action": step["action"]}


# ---- tasks ----------------------------------------------------------------------------------------------------------


def _history(st: List[dict], t: int) -> str:
    """The last HIST_STEPS steps before step t, observations cut."""
    lines = []
    for i in range(max(0, t - HIST_STEPS), t):
        lines.append("step %d action: %s\nstep %d result: %s" % (i + 1, st[i]["action"], i + 1, _obs(st[i])))
    return "\n".join(lines)


_TOK = []


def qwen_tokens(text: str) -> int:
    """Token count with the training tokenizer (cached on the Hub cache; CPU only)."""
    if not _TOK:
        from transformers import AutoTokenizer

        _TOK.append(AutoTokenizer.from_pretrained(TOKENIZER))
    return len(_TOK[0](text)["input_ids"])


PAIR_Q = {"type": "choice", "instructions": "Two agents each produced a final patch for this issue. Which patch "
          "actually resolves it (passes the issue's tests)?", "criteria": ["patch A", "patch B"]}
PAIR_STATS: Counter = Counter()  # the funnel of the last ``verify_pair_rows`` call (what limits the pair count)


def _pair_row_text(state: dict) -> str:
    """The text whose token count is budgeted: state + question + options, as the verifier script counts it."""
    return json.dumps(state, ensure_ascii=False) + " " + PAIR_Q["instructions"] + " " + " | ".join(PAIR_Q["criteria"])


def _ratio(g: dict, b: dict) -> float:
    la, lb = len(g["patch"]), len(b["patch"])
    return max(la, lb) / float(max(1, min(la, lb)))


def _band(g: dict, b: dict) -> int:
    """Length-ratio band of a pair; "the resolved patch is the longer one" is balanced inside each, so neither which
    patch is longer nor by how much says which one is resolved."""
    r = _ratio(g, b)
    return 0 if r < 1.25 else 1 if r <= 2.0 else 2


def verify_pair_rows(runs: Dict[str, List[dict]], rng: random.Random, ntok: Optional[Callable[[str], int]] = None,
                     per_issue: int = PAIR_CAP) -> Iterator[ts.Row]:
    ntok = ntok or qwen_tokens
    PAIR_STATS.clear()
    longer_gold = defaultdict(lambda: [0, 0])  # per (split, length-ratio band): how often the resolved patch is longer
    size: Dict[str, int] = {}  # tokens of a patch as it sits inside the JSON state

    def tokens(p):
        if p not in size:
            size[p] = ntok(json.dumps(p, ensure_ascii=False))
        return size[p]

    for iid in sorted(runs):
        rs = [r for r in runs[iid] if r["patch"] and r["issue"]]
        good = list({r["patch"]: r for r in rs if r["resolved"]}.values())
        bad = list({r["patch"]: r for r in rs if not r["resolved"]}.values())
        PAIR_STATS["issues"] += 1
        if not good or not bad:
            PAIR_STATS["issues_without_both_outcomes"] += 1
            continue
        split = _split(good[0]["repo"], False)
        base = {"repository": good[0]["repo"], "issue": good[0]["issue"], "tests_that_must_pass": good[0].get("tests") or [],
                "patch_A": "", "patch_B": ""}
        overhead = ntok(_pair_row_text(base)) + 8
        pairs = []
        for g in good:
            for b in bad:
                PAIR_STATS["pairs_distinct_patches"] += 1
                if g["patch"] == b["patch"]:
                    PAIR_STATS["drop_identical_patch_both_outcomes"] += 1
                elif len(g["patch"]) > PATCH_MAX or len(b["patch"]) > PATCH_MAX:
                    PAIR_STATS["drop_patch_over_%d_chars" % PATCH_MAX] += 1
                elif _ratio(g, b) > RATIO_MAX:
                    PAIR_STATS["drop_length_ratio_over_%g" % RATIO_MAX] += 1
                elif overhead + tokens(g["patch"]) + tokens(b["patch"]) > ROW_TOKENS:
                    PAIR_STATS["drop_row_over_%d_tokens" % ROW_TOKENS] += 1
                else:
                    pairs.append((g, b))
        PAIR_STATS["pairs_eligible"] += len(pairs)
        PAIR_STATS["issues_with_eligible_pair"] += bool(pairs)
        rng.shuffle(pairs)
        uses: Counter = Counter()  # spread the pairs of one issue over its patches
        used = 0
        while pairs and used < per_issue:
            def surplus(pair):  # > 0: this pair's kind (resolved longer / shorter) is ahead in its band
                lg, longer = longer_gold[(split, _band(*pair))], int(len(pair[0]["patch"]) > len(pair[1]["patch"]))
                return lg[longer] - lg[1 - longer]

            # the least-used patches first, then the rarer kind: keeps "the resolved one is longer" near 50%
            k = min(range(len(pairs)), key=lambda i: (uses[pairs[i][0]["id"]] + uses[pairs[i][1]["id"]], surplus(pairs[i]) > 0, i))
            g, b = pairs.pop(k)
            lg, longer = longer_gold[(split, _band(g, b))], int(len(g["patch"]) > len(b["patch"]))
            if lg[longer] > lg[1 - longer] + 10:
                PAIR_STATS["drop_length_balance"] += 1
                continue
            first = rng.random() < 0.5
            a, c = (g, b) if first else (b, g)
            state = dict(base, patch_A=a["patch"], patch_B=c["patch"])
            if ntok(_pair_row_text(state)) > ROW_TOKENS:  # the estimate above adds parts; this is the exact count
                PAIR_STATS["drop_row_over_%d_tokens" % ROW_TOKENS] += 1
                continue
            lg[longer] += 1
            uses[g["id"]] += 1
            uses[b["id"]] += 1
            used += 1
            PAIR_STATS["rows"] += 1
            yield split, json.dumps(state, ensure_ascii=False), dict(PAIR_Q), 0 if first else 1, None
        PAIR_STATS["drop_over_%d_per_issue" % per_issue] += len(pairs) if used >= per_issue else 0


def _bin(x: int) -> int:
    return max(0, int(x).bit_length() * 2 + (1 if x and x >= 1.5 * (1 << (int(x).bit_length() - 1)) else 0))


def _balanced(cells: Dict[tuple, tuple], caps: Dict[str, int], rng: random.Random) -> List[ts.Row]:
    """From {(split, ...): ([label-0 rows], [label-1 rows])}: equal numbers of both labels inside every cell, scaled
    down evenly over the cells of a split when it holds more than ``caps[split]`` rows."""
    out = []
    for split in sorted({k[0] for k in cells}):
        keys = sorted((k for k in cells if k[0] == split), key=repr)
        n = {k: min(len(cells[k][0]), len(cells[k][1])) for k in keys}
        total, want = sum(n.values()), caps.get(split, 0) // 2
        for k in keys:
            take = n[k] if total <= want else int(n[k] * want / total)
            out += rng.sample(cells[k][0], take) + rng.sample(cells[k][1], take)
    rng.shuffle(out)
    return out


def prefix_cut(n_steps: int, frac: float) -> int:
    """The step a run of ``n_steps`` steps is cut at for the ``frac`` view."""
    return int(round(frac * n_steps))


def outcome_rows(runs: Dict[str, List[dict]], rng: random.Random, caps: Optional[Dict[str, int]] = None) -> Iterator[ts.Row]:
    """Resolved and unresolved runs in equal numbers *within* each (split, view, length bin): patch length (final view)
    and the step reached (prefix views) otherwise predict the outcome on their own (audit 2026-09-29: patch length
    alone 0.60). Every row carries ``value`` = its outcome, the value-head target."""
    caps = caps or {"train": TASKS["agent_outcome"][3], "val": ts.VAL_CAP}
    cells = defaultdict(lambda: ([], []))
    seen = set()  # identical states (the same final patch from several runs of one issue) are kept once
    for iid in sorted(runs):
        for r in runs[iid]:
            if not r["issue"]:
                continue
            split = _split(r["repo"], False)
            view = rng.choice(("final",) + FRACTIONS)
            n = len(r["steps"])
            if view == "final":
                if not r["patch"] or len(r["patch"]) > FINAL_PATCH_MAX:
                    continue
                state = {"repository": r["repo"], "issue": r["issue"], "final_patch": r["patch"]}
                q = ts._noul("Will this final patch resolve the issue (pass its tests)?")
                key = (split, "final", _bin(len(r["patch"])))
            else:
                t = prefix_cut(n, view)
                if n < 12 or t < 3 or n - t < 3:
                    continue
                state = {"repository": r["repo"], "issue": r["issue"], "steps_so_far": t,
                         "recent_steps": _history(r["steps"], t)}
                q = ts._noul("Judging from the run so far, will this agent end up resolving the issue?")
                key = (split, "prefix-%d" % round(100 * view), _bin(t))
            text = json.dumps(state, ensure_ascii=False)
            if text in seen:
                continue
            seen.add(text)
            cells[key][r["resolved"]].append((split, text, q, r["resolved"], None, {"value": float(r["resolved"])}))
    yield from _balanced(cells, caps, rng)


def next_action_rows(runs: Dict[str, List[dict]], rng: random.Random) -> Iterator[ts.Row]:
    for iid in sorted(runs):
        good = [r for r in runs[iid] if r["resolved"] and len(r["steps"]) >= 8 and r["issue"]]
        others = [s["action"] for r in runs[iid] for s in r["steps"]]
        for r in good[:2]:
            st = r["steps"]
            for _ in range(2):
                t = rng.randint(2, len(st) - 3)
                gold = st[t]["action"]
                shown = {s["action"] for s in st[max(0, t - HIST_STEPS):t]}
                pool = list(dict.fromkeys(a for a in [s["action"] for s in st[t + 2:]] + others
                                          if a != gold and a not in shown and a.split(":")[0] != "finish"))
                if len(pool) < 3 or gold.split(":")[0] == "finish" or gold in shown:
                    continue  # a gold repeating a shown action would be the only option seen before (verify 2026-09-29)
                opts = [gold] + rng.sample(pool, 3)
                q, lab = ts._fixed_choice("Which action should the agent take next?", opts, 0, rng)
                state = {"repository": r["repo"], "issue": r["issue"], "steps_so_far": t,
                         "recent_steps": _history(st, t)}
                yield _split(r["repo"], False), json.dumps(state, ensure_ascii=False), q, lab, None


PROGRESS_Q = "Did this step make progress toward resolving the issue?"
PROGRESS_PER_RUN = 2       # labelled steps of one kind taken from one run
ERROR_SHARE_MAX = 0.25     # of the edit-tool "no" rows, at most this share are repeated editor errors (the easy ones)
PROGRESS_STATS: Counter = Counter()


def step_progress_rows(runs: Dict[str, List[dict]], rng: random.Random, caps: Optional[Dict[str, int]] = None) -> Iterator[ts.Row]:
    """One row per labelled step (``progress_labels``): the history before it + the step and its result. Classes are
    balanced inside every (split, tool, step bin), so neither the tool nor how far the run has come answers the
    question (first build: "a step after step 30 made no progress" scored 0.62); among the edit-tool "no" rows,
    reverted edits (successful edits, like every "yes" edit) come first."""
    caps = caps or {"train": TASKS["agent_step_progress"][3], "val": ts.VAL_CAP}
    PROGRESS_STATS.clear()
    pool = defaultdict(list)  # (split, tool, label, kind) -> rows
    seen = set()
    for iid in sorted(runs):
        # runs of one issue often share their first steps: a candidate whose history + step also occur in a run with
        # the other outcome is skipped (verify 2026-09-30: 2 of 1,982 checked rows were provable in one run only)
        ts_ = {p["t"] for r in runs[iid] for p in r.get("progress") or []}
        outcomes = defaultdict(set)
        for r in runs[iid]:
            for t in ts_:
                if t < len(r["steps"]):
                    outcomes[(t, hash(tuple(s["action"] for s in r["steps"][: t + 1])))].add(r["resolved"])
        for r in runs[iid]:
            if not r["issue"] or not r.get("progress"):
                continue
            split = _split(r["repo"], False)
            by_kind = defaultdict(list)
            for p in r["progress"]:
                PROGRESS_STATS["labelled_" + p["kind"]] += 1
                if len(outcomes[(p["t"], hash(tuple(s["action"] for s in r["steps"][: p["t"] + 1])))]) > 1:
                    PROGRESS_STATS["skipped_same_prefix_other_outcome"] += 1
                    continue
                by_kind[p["kind"]].append(p)
            for kind in sorted(by_kind):
                for p in rng.sample(by_kind[kind], min(PROGRESS_PER_RUN, len(by_kind[kind]))):
                    t = p["t"]
                    cand = dict(p["cand"], result=_obs(r["steps"][t]))
                    state = {"repository": r["repo"], "issue": r["issue"], "steps_so_far": t,
                             "recent_steps": _history(r["steps"], t), "candidate_step": cand}
                    text = json.dumps(state, ensure_ascii=False)
                    if text in seen:
                        continue
                    seen.add(text)
                    pool[(split, p["tool"], p["label"], kind)].append((_bin(t), (split, text, ts._noul(PROGRESS_Q), p["label"],
                                                                                 None, {"kind": kind})))
    cells = defaultdict(lambda: ([], []))
    for (split, tool, label, kind), rows in sorted(pool.items()):
        PROGRESS_STATS["rows_%s_%s" % (split, kind)] += len(rows)
        if kind == "same_error_as_previous" and tool == "edit":
            hard = len(pool.get((split, tool, 0, "edit_reverted"), []))
            rows = rng.sample(rows, min(len(rows), int(hard * ERROR_SHARE_MAX / (1 - ERROR_SHARE_MAX))))
        for step_bin, row in rows:  # the step reached is a cell too: reverts and repeats come later than kept edits
            cells[(split, tool, step_bin)][label].append(row)
    yield from _balanced(cells, caps, rng)


_RUNS: Dict[str, Dict] = {}


def _cached_runs(parquet: str) -> Dict[str, List[dict]]:
    if parquet not in _RUNS:
        _RUNS[parquet] = load_runs(parquet, permissive_instances(), tests=instance_tests())
    return _RUNS[parquet]


LIC = ("nebius/SWE-rebench-openhands-trajectories CC BY 4.0 (runs of Qwen3-Coder-480B-A35B-Instruct, Apache-2.0); "
       "permissive-licensed repositories only (nebius/SWE-rebench license_name); attribution: Nebius SWE-rebench")
TASKS = {"agent_verify_pair": (verify_pair_rows, "verifier: which final patch resolves the issue", False, 20000),
         "agent_outcome": (outcome_rows, "value / early stop: will this run resolve the issue (noul + value target)", True, 20000),
         "agent_next_action": (next_action_rows, "agent policy: which tool call next (successful runs)", False, 20000),
         "agent_step_progress": (step_progress_rows, "agent monitor: did this step make progress (exact from the log)", True, 15000)}


def sources(parquet: str = PARQUET) -> Dict[str, dict]:
    out = {}
    for name, (fn, use, balance, cap) in TASKS.items():
        out[name] = {"fn": (lambda f: lambda rng: f(_cached_runs(parquet), rng))(fn),
                     "repo": "nebius/SWE-rebench-openhands-trajectories", "license": LIC, "use": use,
                     "balance": balance, "train_cap": cap}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from TASKS, or 'all'")
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    srcs = sources(a.parquet)
    for n in (list(TASKS) if a.names == ["all"] else a.names):
        ts.build(n, a.root, spec=srcs[n], log=lambda m: print(m, flush=True))
        for stats in ((PAIR_STATS,) if n == "agent_verify_pair" else (PROGRESS_STATS,) if n == "agent_step_progress" else ()):
            print("%s funnel: %s" % (n, json.dumps(dict(stats), sort_keys=True)), flush=True)
