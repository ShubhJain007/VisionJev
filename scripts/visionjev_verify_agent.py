"""Independent verification of the agentic trajectory sets (laya/visionjev_agent.py) against the raw parquet: every val
row and a train sample are traced back to the run(s) they came from, with the label rules written again here (nothing
below calls the builder's labelling code; only its chat parsing and scratch-file stripping are reused).

  labels     verify_pair: the gold patch comes from resolved runs only, the other from unresolved runs only, same issue;
               ``tests_that_must_pass`` are exactly the FAIL_TO_PASS names of the task (names only, <= 12), no patch is
               the task's reference patch; <= 8 pairs per issue; patches <= 8,000 chars, length ratio <= 3.
             outcome: ``value`` == label on every row; final-patch rows match their run's `resolved` (in every
               instance showing that issue); prefix rows match
               a run whose actions at the shown steps are exactly the shown ones and that is cut at 25 / 50 / 75% of its
               steps, at least 3 steps before its end; classes equal inside every (split, view, length / step bin).
             next_action: the gold is the action right after the shown history in a resolved run; no option is already
               in the shown history or a finish call.
             step_progress: the candidate is step t of a run with the shown history, and the raw log proves the label:
               no  = same shell command + same failing output inside the shown history | the previous step's identical
                     error | an edit later reverted (inverse edit / undo / git checkout) with none of its lines in the
                     final patch;
               yes = resolved run and (an edit never reverted whose added lines are all in the final patch of a file the
                     run did not create | a test command that failed at its last use in the shown history and passes).
               Classes equal inside every (split, tool, step bin).
  splits     no repository in both train and val (in any of the four sets or the swe_* sets); no duplicate rows.
  licence    every repository permissive and outside SWE-bench (same rule as the builder, recomputed).
  length     prompt tokens with the Qwen3-VL-4B tokenizer vs the 4096 training budget (ALL pair rows: <= 3,800).
  tells      simple heuristics on val: tool-name prior (next action); shorter patch / fewer files / fewer new files /
             adds tests (pairs); step reached (outcome, progress); exit code and debug prints (progress).

    .venv/bin/python scripts/visionjev_verify_agent.py [--train-sample 1000]
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from laya import visionjev_agent as A  # noqa: E402
from laya.visionjev_data import DATA_ROOT  # noqa: E402
from laya.visionjev_swe import _split  # noqa: E402

SETS = ("agent_verify_pair", "agent_outcome", "agent_next_action", "agent_step_progress")
STEP = re.compile(r"^step (\d+) action: (.*)$", re.M)
WINDOW = 6  # steps of history a state shows

# ---- step-progress rules, written again from the definition ----------------------------------------------------------------
EXIT_RE = re.compile(r"\[Command finished with exit code (-?\d+)\]")
FAIL_TEXT = re.compile(r"Traceback \(most recent call last\)|\b[1-9][0-9]* (?:failed|error|errors)\b|\bFAILED\b|\b\w*(?:Error|Exception):")
RUNNER = re.compile(r"^(?:cd \S+ *(?:&&|;) *)?(?:\w+=\S+ +)*(?:timeout \d+ +)?(?:python3?(?:\.\d+)? |pytest\b|py\.test\b|tox\b|nox\b|"
                    r"nosetests\b|npm (?:run )?test\b|yarn test\b|go test\b|cargo test\b|make\b|jest\b|mocha\b|phpunit\b|rspec\b|"
                    r"mvn\b|\./gradlew\b)")
SHELL_FOOTER = re.compile(r"\[(?:The command completed with exit code|Current working directory|Python interpreter|"
                          r"Command finished with exit code)[^\]]*\]")


def compact(raw):
    """What the proofs need from one raw step, without keeping the observation text."""
    obs, args = raw["observation"], raw["args"]
    codes = EXIT_RE.findall(obs)
    s = {"a": raw["action"], "n": raw["name"], "oh": hashlib.md5(obs.encode("utf-8", "replace")).digest(),
         "code": int(codes[-1]) if codes else None, "ftxt": bool(FAIL_TEXT.search(obs)),
         "body": len(SHELL_FOOTER.sub("", obs).strip())}
    if raw["name"] == "execute_bash":
        s["c"] = re.sub(r"\s+", " ", str(args.get("command") or "")).strip()
        s["inp"] = str(args.get("is_input", "false")).strip().lower() == "true"
    elif raw["name"] == "str_replace_editor":
        s["ed"] = (args.get("command"), str(args.get("path") or ""), str(args.get("old_str") or ""), str(args.get("new_str") or ""))
        s["err"] = obs.startswith("ERROR")
        s["edited"] = obs.startswith("The file") and "has been edited" in obs[:400]
        s["undone"] = obs.startswith("Last edit to")
    return s


def is_shell(s):
    return s["n"] == "execute_bash" and not s["inp"] and bool(s["c"])


def fails(s):
    if s["n"] == "str_replace_editor":
        return s["err"]
    if not is_shell(s) or s["code"] is None or s["code"] == -1:
        return False
    return s["code"] != 0 or bool(RUNNER.match(s["c"]) and s["ftxt"])


def added_lines(old, new):
    had = {l.strip() for l in old.splitlines()}
    return list(dict.fromkeys(l.strip() for l in new.splitlines() if len(l.strip()) >= 8 and l.strip() not in had))


def final_additions(patch):
    """file -> set of added lines, or None when the diff creates the file."""
    out = {}
    for block in re.split(r"(?m)^(?=diff --git )", patch or ""):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", block)
        if m:
            out[m.group(2)] = None if "\nnew file mode" in block[:300] else \
                {l[1:].strip() for l in block.splitlines() if l.startswith("+") and not l.startswith("+++")}
    return out


def reverted_later(steps, t):
    """Does a later step restore what the edit at step t changed?"""
    _, path, old, new = steps[t]["ed"]
    mine = added_lines(old, new)
    other_edit = False
    for s in steps[t + 1:]:
        if s["n"] == "str_replace_editor" and s["ed"][1] == path:
            cmd, _, old2, new2 = s["ed"]
            if cmd == "undo_edit" and s["undone"]:
                if not other_edit:
                    return True
                other_edit = True
            elif cmd == "str_replace" and s["edited"]:
                gone = {l.strip() for l in old2.splitlines()}
                if new2 == old and old.strip() and (old2 == new or all(l in gone for l in mine)):
                    return True
                other_edit = True
            elif cmd in ("create", "insert") and not s["err"]:
                other_edit = True
        elif s["n"] == "execute_bash" and is_shell(s) and s["code"] == 0:
            if re.search(r"git reset --hard(?: +HEAD)?\s*(?:$|[;&|])", s["c"]):
                return True
            for m in re.finditer(r"git (?:checkout|restore)((?: +[^\s;&|]+)+)", s["c"]):
                for tok in m.group(1).split():
                    tok = tok.strip("'\"")
                    rel = tok.lstrip("./")
                    if tok == "." or (rel and tok not in ("--", "HEAD") and not tok.startswith("-") and path.endswith("/" + rel)):
                        return True
    return False


def proofs(run, t):
    """The labels the raw log proves for step t of a run: a set of 'no:*' / 'yes:*' names."""
    steps, s, out = run["steps"], run["steps"][t], set()
    prev_same = t > 0 and fails(s) and fails(steps[t - 1]) and steps[t - 1]["oh"] == s["oh"] and s["body"] >= 30
    if s["n"] == "execute_bash" and is_shell(s):
        same = [j for j in range(max(0, t - WINDOW), t) if steps[j]["n"] == "execute_bash" and is_shell(steps[j])
                and steps[j]["c"] == s["c"]]
        if fails(s):
            if any(steps[j]["oh"] == s["oh"] for j in same):
                out.add("no:repeat_same_error")
            if prev_same:
                out.add("no:same_error_as_previous")
        elif run["res"] and same and fails(steps[same[-1]]) and RUNNER.match(s["c"]) and s["code"] == 0 and not s["ftxt"]:
            out.add("yes:test_newly_passes")
    elif s["n"] == "str_replace_editor":
        if prev_same:
            out.add("no:same_error_as_previous")
        cmd, path, old, new = s["ed"]
        mine = added_lines(old, new)
        made_here = any(x["n"] == "str_replace_editor" and x["ed"][0] == "create" and x["ed"][1] == path and not x["err"]
                        for x in steps[:t])
        if cmd == "str_replace" and s["edited"] and mine and not made_here:
            finals = [v for f, v in run["final"].items() if path.endswith("/" + f)]
            if None not in finals:
                kept = set().union(*finals) if finals else set()
                rev = reverted_later(steps, t)
                if rev and not any(l in kept for l in mine):
                    out.add("no:edit_reverted")
                if not rev and run["res"] and all(l in kept for l in mine):
                    out.add("yes:edit_survives")
    return out


def shown_matches(shown, raw, limit, joiner):
    """Is ``shown`` the display form (head + joiner + tail beyond ``limit`` chars) of ``raw``?"""
    if len(raw) <= limit:
        return shown == raw
    return shown == raw[: limit // 2] + joiner + raw[-limit // 2:]


def size_bin(x):
    """Half-octave bins of a length or a step index (the builder's cells; recomputed here)."""
    x = int(x)
    if x <= 0:
        return 0
    top = 1 << (x.bit_length() - 1)
    return 2 * x.bit_length() + (1 if x >= 1.5 * top else 0)


def rows(name, split):
    return [json.loads(l) for l in open(os.path.join(DATA_ROOT, name, split + ".jsonl"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-sample", type=int, default=1000)
    a = ap.parse_args()
    rng = random.Random(0)
    data = {n: {"train": rows(n, "train"), "val": rows(n, "val")} for n in SETS}
    check = {n: data[n]["val"] + rng.sample(data[n]["train"], min(a.train_sample, len(data[n]["train"]))) for n in SETS}
    check["agent_verify_pair"] = data["agent_verify_pair"]["val"] + data["agent_verify_pair"]["train"]  # pairs: all rows
    report, fails_ = {}, defaultdict(list)

    # ---- index the raw runs we need ----------------------------------------------------------------------------------
    need_patch, need_issue, need_steps = set(), set(), set()
    for n in SETS:
        for r in check[n]:
            s = json.loads(r["state_text"])
            need_issue.add(s["issue"])
            if n == "agent_step_progress":
                need_steps.add(s["issue"])
            for k in ("patch_A", "patch_B", "final_patch"):
                if k in s:
                    need_patch.add(s[k])
    import pyarrow.parquet as pq

    # Everything is keyed by INSTANCE: several SWE-rebench instances can share one issue text (the same issue at two
    # base commits: a patch may resolve one and not the other), and a run of an instance may show the issue in another
    # wrapping (its issue text then differs from its siblings'); a row names an instance through (issue, repo, tests).
    patch_res = defaultdict(lambda: defaultdict(Counter))  # instance -> stripped patch -> Counter(resolved)
    runs_by_iid, iids_by_issue = defaultdict(list), defaultdict(set)
    pf = pq.ParquetFile(A.PARQUET)
    for batch in pf.iter_batches(batch_size=128, columns=["instance_id", "repo", "trajectory", "model_patch", "resolved"]):
        for r in batch.to_pylist():
            traj = r["trajectory"] or []
            issue = A.issue_text(traj)
            iid = r["instance_id"]
            raw_patch = A._unquote(r["model_patch"])
            p = A.strip_scratch(raw_patch)
            patch_res[iid][p][int(r["resolved"] or 0)] += 1
            raw = A.raw_steps(traj)
            run = {"repo": r["repo"], "res": int(r["resolved"] or 0), "acts": [s["action"] for s in raw], "issue": issue}
            if issue in need_steps:
                run["steps"] = [compact(s) for s in raw]
                run["final"] = final_additions(raw_patch)
            runs_by_iid[iid].append(run)
            iids_by_issue[(issue, r["repo"])].add(iid)
    runs_by_issue = {}  # (issue, repo) -> every run of every instance that shows this issue

    def runs_of(s):
        key = (s["issue"], s["repository"])
        if key not in runs_by_issue:
            runs_by_issue[key] = [u for iid in sorted(iids_by_issue[key]) for u in runs_by_iid[iid]]
        return runs_by_issue[key]

    print("indexed %d instances, %d issue texts" % (len(runs_by_iid), len(iids_by_issue)), flush=True)

    # ---- labels: verifier pairs ------------------------------------------------------------------------------------------
    import datasets

    f2p, ref_patch = {}, {}
    need_iids = set().union(*iids_by_issue.values()) if iids_by_issue else set()
    for split in ("test", "filtered"):
        for r in datasets.load_dataset("nebius/SWE-rebench", split=split).select_columns(["instance_id", "FAIL_TO_PASS", "patch"]):
            if r["instance_id"] in need_iids:
                names = r["FAIL_TO_PASS"] if isinstance(r["FAIL_TO_PASS"], list) else json.loads(r["FAIL_TO_PASS"] or "[]")
                f2p[r["instance_id"]] = list(dict.fromkeys(str(t).strip()[:160] for t in names if str(t).strip()))[:12]
                ref_patch[r["instance_id"]] = (r["patch"] or "").strip()
    c, per_issue = Counter(), Counter()
    longer = defaultdict(Counter)
    for r in check["agent_verify_pair"]:
        s = json.loads(r["state_text"])
        gold, other = (s["patch_A"], s["patch_B"]) if r["label"] == 0 else (s["patch_B"], s["patch_A"])
        tests = s.get("tests_that_must_pass")
        iids = iids_by_issue[(s["issue"], s["repository"])]
        mine = sorted(i for i in iids if f2p.get(i) == tests)  # the instance(s) this row can come from
        ok = gold != other and any(set(patch_res[i][gold]) == {1} and set(patch_res[i][other]) == {0} for i in mine)
        c["ok" if ok else "bad"] += 1
        if not ok:
            fails_["verify_pair"].append({"id": r["id"], "instances": mine[:3],
                                          "gold_res": [dict(patch_res[i][gold]) for i in mine][:3], "other_res": [dict(patch_res[i][other]) for i in mine][:3]})
        t_ok = (set(s) == {"repository", "issue", "tests_that_must_pass", "patch_A", "patch_B"} and isinstance(tests, list)
                and 0 < len(tests) <= 12 and bool(mine)
                and all(isinstance(t, str) and "\n" not in t and len(t) <= 160 for t in tests))
        c["tests_ok" if t_ok else "tests_bad"] += 1
        if not t_ok:
            fails_["verify_pair_tests"].append({"id": r["id"], "tests": tests, "instances": sorted(iids)[:3]})
        # an agent may well produce the upstream fix verbatim; counted, not a failure (the state never shows it)
        c["agent_patch_equals_reference_patch"] += any(ref_patch.get(i) and ref_patch[i] in (s["patch_A"].strip(), s["patch_B"].strip()) for i in mine)
        la, lb = len(gold), len(other)
        ratio = max(la, lb) / max(1, min(la, lb))
        c["len_ok" if max(la, lb) <= 8000 and ratio <= 3.0 else "len_bad"] += 1
        band = "<1.25" if ratio < 1.25 else "<=2" if ratio <= 2 else "<=3"
        longer[(r["id"].split("-")[1], band)]["gold_longer" if la > lb else "gold_shorter"] += 1
        per_issue[(s["issue"], s["repository"], tuple(tests or ()))] += 1
    over = {k: n for k, n in per_issue.items() if n > 8 * len([i for i in iids_by_issue[k[:2]] if f2p.get(i) == list(k[2])])}
    c["max_pairs_per_issue"] = max(per_issue.values())
    c["issues"] = len(per_issue)
    c["issues_over_8_per_instance"] = len(over)
    if over:
        fails_["verify_pair_cap"].append({"%s|%s" % (k[1], k[0][:40]): n for k, n in list(over.items())[:5]})
    report["verify_pair_labels"] = dict(c)
    report["verify_pair_length_balance"] = {"%s ratio %s" % k: dict(v) for k, v in sorted(longer.items())}

    # ---- labels: outcome ---------------------------------------------------------------------------------------------------
    c = Counter()
    for r in check["agent_outcome"]:
        s = json.loads(r["state_text"])
        if "final_patch" in s:
            res = [patch_res[i][s["final_patch"]] for i in iids_by_issue[(s["issue"], s["repository"])]]
            res = [x for x in res if x]
            ok = bool(res) and all(set(x) == {r["label"]} for x in res)
            c["final_" + ("ok" if ok else ("ambiguous" if res else "bad"))] += 1
            if not ok:
                fails_["outcome_final"].append({"id": r["id"], "label": r["label"], "raw": [dict(x) for x in res][:3]})
        else:
            t, shown = s["steps_so_far"], STEP.findall(s["recent_steps"])
            acts = [x for _, x in shown]
            idx = [int(i) for i, _ in shown]
            cand = [u for u in runs_of(s) if u["acts"][t - len(acts):t] == acts and len(u["acts"]) >= t + 3]
            cuts = [u for u in cand if len(u["acts"]) >= 12 and t in {int(round(f * len(u["acts"]))) for f in (0.25, 0.5, 0.75)}]
            labs = {u["res"] for u in cuts}
            ok = bool(cuts) and labs == {r["label"]} and idx == list(range(t - len(acts) + 1, t + 1))
            c["prefix_" + ("ok" if ok else ("ambiguous" if len(labs) > 1 else "not_at_25_50_75" if cand and not cuts else "bad"))] += 1
            if not ok:
                fails_["outcome_prefix"].append({"id": r["id"], "label": r["label"], "matches": len(cand), "at_cut": len(cuts),
                                                 "labels": sorted(labs)})
    cells = defaultdict(Counter)
    for sp in ("train", "val"):
        for r in data["agent_outcome"][sp]:  # every row
            s = json.loads(r["state_text"])
            c["value_ok" if isinstance(r.get("value"), float) and r["value"] == float(r["label"]) else "value_bad"] += 1
            view = ("final", size_bin(len(s["final_patch"]))) if "final_patch" in s else ("prefix", size_bin(s["steps_so_far"]))
            cells[(sp,) + view][r["label"]] += 1
            c[sp + "_" + view[0]] += 1
    c["cells"] = len(cells)
    c["cells_unbalanced"] = sum(v[0] != v[1] for v in cells.values())
    if c["value_bad"] or c["cells_unbalanced"]:
        fails_["outcome_value_or_balance"].append({"value_bad": c["value_bad"], "cells_unbalanced": c["cells_unbalanced"]})
    report["outcome_labels"] = dict(c)

    # ---- labels: next action -------------------------------------------------------------------------------------------------
    c = Counter()
    for r in check["agent_next_action"]:
        s = json.loads(r["state_text"])
        t, acts = s["steps_so_far"], [x for _, x in STEP.findall(s["recent_steps"])]
        opts, gold = r["question"]["criteria"], r["question"]["criteria"][r["label"]]
        cand = [u for u in runs_of(s) if u["res"] == 1 and u["acts"][t - len(acts):t] == acts
                and len(u["acts"]) > t and u["acts"][t] == gold]
        clean = not (set(opts) & set(acts)) and not any(o.startswith("finish") for o in opts) and len(set(opts)) == len(opts)
        # would another option ALSO be a correct next step (same history, different resolved run)?
        alt = {u["acts"][t] for u in runs_of(s) if u["res"] == 1 and len(u["acts"]) > t
               and u["acts"][t - len(acts):t] == acts} - {gold}
        ok = bool(cand) and clean
        c["ok" if ok else "bad"] += 1
        c["other_option_also_valid"] += bool(alt & set(opts))
        if not ok:
            fails_["next_action"].append({"id": r["id"], "matches": len(cand), "clean": clean})
    report["next_action_labels"] = dict(c)

    # ---- labels: step progress -----------------------------------------------------------------------------------------------
    c = Counter()
    for r in check["agent_step_progress"]:
        s = json.loads(r["state_text"])
        t, shown, cand_step = s["steps_so_far"], STEP.findall(s["recent_steps"]), s["candidate_step"]
        acts, idx = [x for _, x in shown], [int(i) for i, _ in shown]
        found = []
        for u in runs_of(s):
            if "steps" not in u or len(u["acts"]) <= t or u["acts"][t - len(acts):t] != acts:
                continue
            st = u["steps"][t]
            if "new_str" in cand_step:
                same = (st["n"] == "str_replace_editor" and st["ed"][0] == "str_replace"
                        and cand_step["action"] == "str_replace_editor: command=str_replace, path=%s" % st["ed"][1]
                        and shown_matches(cand_step["old_str"], st["ed"][2], 700, "\n...\n")
                        and shown_matches(cand_step["new_str"], st["ed"][3], 1000, "\n...\n"))
            elif cand_step["action"].startswith("execute_bash: command="):
                same = st["n"] == "execute_bash" and shown_matches(cand_step["action"][len("execute_bash: command="):], st["c"], 600, " ... ")
            else:
                same = cand_step["action"] == st["a"]
            if same:
                found.append(proofs(u, t))
        want = "yes:" if r["label"] == 1 else "no:"
        ok = (bool(found) and idx == list(range(t - len(acts) + 1, t + 1)) and len(acts) == min(t, WINDOW)
              and all(any(p.startswith(want) for p in f) and not any(not p.startswith(want) for p in f) for f in found)
              and any(("yes:" if r["label"] else "no:") + r.get("kind", "?") in f for f in found))
        verdict = "ok" if ok else "no_matching_step" if not found else "ambiguous" if any(f != found[0] for f in found) else "not_proven"
        c[verdict] += 1
        if not ok:
            fails_["step_progress"].append({"id": r["id"], "label": r["label"], "kind": r.get("kind"), "proofs": [sorted(f) for f in found][:4]})
    cells = defaultdict(Counter)
    for sp in ("train", "val"):
        for r in data["agent_step_progress"][sp]:  # every row
            s = json.loads(r["state_text"])
            tool = "edit" if s["candidate_step"]["action"].startswith("str_replace_editor") else "shell"
            cells[(sp, tool, size_bin(s["steps_so_far"]))][r["label"]] += 1
            c["%s_%s_%s" % (sp, r.get("kind"), r["label"])] += 1
    c["cells"] = len(cells)
    c["cells_unbalanced"] = sum(v[0] != v[1] for v in cells.values())
    if c["cells_unbalanced"]:
        fails_["step_progress_balance"].append(c["cells_unbalanced"])
    report["step_progress_labels"] = dict(c)

    # ---- splits, duplicates, licence ----------------------------------------------------------------------------------
    repos = {sp: set() for sp in ("train", "val")}
    dup = Counter()
    for n in SETS:
        seen = {}
        for sp in ("train", "val"):
            for r in data[n][sp]:
                repo = json.loads(r["state_text"])["repository"]
                repos[sp].add(repo)
                if _split(repo, False) != sp:
                    fails_["split_rule"].append(r["id"])
                key = (r["state_text"], json.dumps(r["question"], sort_keys=True))
                if key in seen:
                    dup["%s:%s-%s" % (n, seen[key], sp)] += 1
                seen[key] = sp
    swe_repos = {sp: set() for sp in ("train", "val")}
    for n in ("swe_file_loc", "swe_patch_match", "swe_test_fail"):
        for sp in ("train", "val"):
            p = os.path.join(DATA_ROOT, n, sp + ".jsonl")
            if os.path.exists(p):
                swe_repos[sp] |= {json.loads(json.loads(l)["state_text"])["repository"] for l in open(p)}
    allowed = A.permissive_instances()
    ok_repos = {iid.rsplit("-", 1)[0] for iid in allowed}  # instance ids are owner__repo-N
    ok_repos = {x.replace("__", "/", 1) for x in ok_repos}
    report["splits"] = {"repos_train": len(repos["train"]), "repos_val": len(repos["val"]),
                        "repo_overlap_train_val": len(repos["train"] & repos["val"]),
                        "overlap_with_swe_sets_other_split": len((repos["train"] & swe_repos["val"]) | (repos["val"] & swe_repos["train"])),
                        "split_rule_violations": len(fails_["split_rule"]), "duplicates": dict(dup),
                        "rows": {n: {sp: len(data[n][sp]) for sp in ("train", "val")} for n in SETS}}
    report["licence"] = {"repos_not_permissive_or_swebench": sorted((repos["train"] | repos["val"]) - ok_repos)[:20]}

    # ---- length ---------------------------------------------------------------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    lens = {}
    for n in SETS:
        ls = sorted(len(tok(r["state_text"] + " " + r["question"]["instructions"] + " " +
                            " | ".join(r["question"].get("criteria") or []))["input_ids"]) for r in check[n])
        lens[n] = {"rows": len(ls), "p50": ls[len(ls) // 2], "p95": ls[int(0.95 * len(ls))], "max": ls[-1],
                   "over_3500": round(sum(x > 3500 for x in ls) / len(ls), 3), "over_3800": sum(x > 3800 for x in ls)}
    if lens["agent_verify_pair"]["over_3800"]:
        fails_["pair_tokens"].append(lens["agent_verify_pair"]["over_3800"])
    report["tokens"] = lens

    # ---- tells ----------------------------------------------------------------------------------------------------------
    tool = lambda o: o.split(":")[0]  # noqa: E731
    g, d = Counter(), Counter()
    for r in data["agent_next_action"]["train"]:
        for i, o in enumerate(r["question"]["criteria"]):
            (g if i == r["label"] else d)[tool(o)] += 1
    prior = {k: (g[k] + 1) / (g[k] + d[k] + 2) for k in set(g) | set(d)}
    hit = sum(max(range(4), key=lambda i: prior.get(tool(r["question"]["criteria"][i]), 0.25)) == r["label"]
              for r in data["agent_next_action"]["val"])
    tells = {"next_action_tool_prior": round(hit / len(data["agent_next_action"]["val"]), 3), "chance": 0.25}
    files = lambda p: len(re.findall(r"^diff --git ", p, re.M))  # noqa: E731
    newf = lambda p: p.count("\nnew file mode")  # noqa: E731
    tests = lambda p: int(bool(re.search(r"^diff --git a/\S*test", p, re.M)))  # noqa: E731
    for name, f in (("shorter", len), ("fewer_files", files), ("fewer_new_files", newf), ("no_test_files", tests)):
        h = n_ = 0
        for r in data["agent_verify_pair"]["val"]:
            s = json.loads(r["state_text"])
            fa, fb = f(s["patch_A"]), f(s["patch_B"])
            if fa != fb:
                n_ += 1
                h += (0 if fa < fb else 1) == r["label"]
        tells["pair_" + name] = {"acc_when_differs": round(h / max(1, n_), 3), "n": n_}

    def rule(rs, fn):  # accuracy of a yes / no rule on the rows where it applies
        xs = [(fn(json.loads(r["state_text"])), r["label"]) for r in rs]
        xs = [(p, l) for p, l in xs if p is not None]
        return {"acc": round(sum(int(p) == l for p, l in xs) / max(1, len(xs)), 3), "n": len(xs)}

    is_edit = lambda s: "new_str" in s["candidate_step"]  # noqa: E731
    is_cmd = lambda s: s["candidate_step"]["action"].startswith("execute_bash")  # noqa: E731
    pv, ov = data["agent_step_progress"]["val"], data["agent_outcome"]["val"]
    tells["progress_late_step_is_no"] = rule(pv, lambda s: s["steps_so_far"] <= 30)
    tells["progress_shell_exit0_without_error_text_is_yes"] = rule(pv, lambda s: (
        "exit code 0]" in s["candidate_step"]["result"] and not FAIL_TEXT.search(s["candidate_step"]["result"])) if is_cmd(s) else None)
    tells["progress_edit_with_print_or_debug_is_no"] = rule(pv, lambda s: (
        not re.search(r"print\(|debug", s["candidate_step"]["new_str"], re.I)) if is_edit(s) else None)
    tells["progress_editor_error_is_no"] = rule(pv, lambda s: (
        not s["candidate_step"]["result"].startswith("ERROR")) if s["candidate_step"]["action"].startswith("str_replace_editor") else None)
    tells["progress_edit_of_a_test_file_is_no"] = rule(pv, lambda s: ("test" not in s["candidate_step"]["action"]) if is_edit(s) else None)
    tells["outcome_prefix_late_step_is_no"] = rule(ov, lambda s: (s["steps_so_far"] <= 30) if "steps_so_far" in s else None)
    tells["outcome_final_short_patch_is_yes"] = rule(ov, lambda s: (len(s["final_patch"]) <= 1500) if "final_patch" in s else None)
    report["tells"] = tells
    report["fail_examples"] = {k: v[:5] for k, v in fails_.items() if v}
    print(json.dumps(report, indent=1), flush=True)

    from laya.visionjev_log import log_experiment

    log_experiment("verify", "verify:agent-trajectories", config={"checker": "scripts/visionjev_verify_agent.py",
                   "train_sample": a.train_sample, "pairs": "all rows"}, metrics=report, datasets=list(SETS))


if __name__ == "__main__":
    main()
