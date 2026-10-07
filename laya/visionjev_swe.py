"""Software-engineering decisions from real GitHub issues and the pull requests that fixed them (SWE-bench family).
Labels are exact from the merged fix and its tests; nothing is executed or generated.

Sources (licenses checked 2026-09-28): SWE-Gym (MIT; permissive repos) and SWE-rebench (CC BY 4.0), keeping only
repositories whose own license is permissive (MIT / Apache / BSD / ISC / Unlicense / Zope / PSF / MIT-CMU / CC0) --
patch code carries its repository's license. SWE-bench (MIT, GitHub) is the public benchmark: eval only, pylint (GPL)
dropped, and every repository in it is excluded from the training sources so the benchmark stays clean. SWE-smith is
not used (its issue texts are written by language models). Training rows are split by *repository* (15% of
repositories are val only), so val measures new codebases.

* ``swe_file_loc``    (choice) issue -> which file must change to fix it? Distractors: files changed by other fixes in
                      the same repository (never files of this fix; tests excluded), the ones sharing the most words
                      with the issue.
* ``swe_patch_match`` (choice, 3) issue + three diffs -> which diff fixes it? The others are real fixes of other issues
                      in the same repository.
* ``swe_test_fail``   (choice, 4) issue + the fix -> which test fails before the fix and passes after it? One
                      FAIL_TO_PASS test and the three PASS_TO_PASS tests of the same instance sharing the most words
                      with the issue and fix.
* ``swebench_file_loc`` / ``swebench_patch_match`` / ``swebench_test_fail``: the same on SWE-bench, eval only.

    python -m laya.visionjev_swe all
"""
import argparse
import ast
import hashlib
import json
import random
import re
from collections import defaultdict
from typing import Dict, Iterator, List

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT

PERMISSIVE = re.compile(r"\b(MIT|Apache|BSD|ISC|Unlicense|Zope|Python Software Foundation|PSF|CC0|0BSD)\b", re.I)
COPYLEFT = re.compile(r"GPL|MPL|Mozilla|EPL|Eclipse|CDDL|EUPL|Affero|Creative Commons Attribution-ShareAlike", re.I)
ISSUE_MAX, PATCH_MAX = 2000, 1200


def _words(s: str) -> set:
    return set(w for w in re.findall(r"[a-z][a-z0-9]{2,}", re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower()))


def _hardest(pool, text, k, rng):
    """The ``k`` candidates sharing the most words with ``text`` (ties at random): keyword matching alone must not
    find the gold (audit: file / test names that echo the issue scored ~0.47-0.49 vs chance 0.25)."""
    tw = _words(text)
    return sorted(pool, key=lambda c: (-len(_words(c) & tw), rng.random()))[:k]


def _files(patch: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"^diff --git a/(\S+) b/", patch, re.M)))


def _is_test(path: str) -> bool:
    p = path.lower()
    return "test" in p.split("/")[-1] or "/tests/" in "/" + p or "/test/" in "/" + p or p.startswith("tests/")


def _tests(v) -> List[str]:
    if isinstance(v, list):
        return v
    try:
        return list(ast.literal_eval(v or "[]"))
    except (ValueError, SyntaxError):
        try:
            return list(json.loads(v))
        except ValueError:
            return []


def _instances(bench: bool):
    """(instances, SWE-bench repos). Training: SWE-Gym + permissive SWE-rebench minus SWE-bench repos."""
    sb = [dict(r) for r in ts._ds("princeton-nlp/SWE-bench", split="test") if r["repo"] != "pylint-dev/pylint"]
    sb_repos = {r["repo"] for r in sb} | {"pylint-dev/pylint"}
    if bench:
        return sb, sb_repos
    seen, out = set(), []
    for r in ts._ds("SWE-Gym/SWE-Gym", split="train"):
        if r["repo"] not in sb_repos and r["instance_id"] not in seen:
            seen.add(r["instance_id"])
            out.append(dict(r, _license="MIT (SWE-Gym; permissive repo)"))
    for split in ("test", "filtered"):
        for r in ts._ds("nebius/SWE-rebench", split=split):
            lic = r.get("license_name") or ""
            if r["instance_id"] in seen or r["repo"] in sb_repos or not PERMISSIVE.search(lic) or COPYLEFT.search(lic):
                continue
            seen.add(r["instance_id"])
            out.append(dict(r, _license=lic))
    return out, sb_repos


def _split(repo: str, bench: bool) -> str:
    if bench:
        return "val"
    return "val" if int(hashlib.md5(repo.encode()).hexdigest(), 16) % 100 < 15 else "train"


def _by_repo(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["repo"]].append(r)
    return d


def file_loc_rows(rng, bench) -> Iterator[ts.Row]:
    rows, _ = _instances(bench)
    repo_files = {repo: sorted({f for r in rs for f in _files(r["patch"]) if not _is_test(f)}) for repo, rs in _by_repo(rows).items()}
    for r in rows:
        mine = [f for f in _files(r["patch"]) if not _is_test(f)]
        if not mine or not r["problem_statement"].strip():
            continue
        gold = rng.choice(mine)
        pool = [f for f in repo_files[r["repo"]] if f not in mine]
        if len(pool) < 2:
            continue
        opts = [gold] + _hardest(pool, r["problem_statement"], 3, rng)
        q, lab = ts._fixed_choice("Which file needs to be changed to fix this issue?", opts, 0, rng)
        state = {"repository": r["repo"], "issue": r["problem_statement"][:ISSUE_MAX]}
        yield _split(r["repo"], bench), json.dumps(state, ensure_ascii=False), q, lab, None


def patch_match_rows(rng, bench) -> Iterator[ts.Row]:
    rows, _ = _instances(bench)
    by_repo = _by_repo(rows)
    for r in rows:
        others = [o for o in by_repo[r["repo"]] if o["instance_id"] != r["instance_id"] and o["patch"] != r["patch"]]
        if len(others) < 2 or not r["problem_statement"].strip():
            continue
        cands = [r["patch"]] + [o["patch"] for o in rng.sample(others, 2)]
        order = [0, 1, 2]
        rng.shuffle(order)
        state = {"repository": r["repo"], "issue": r["problem_statement"][:ISSUE_MAX]}
        for letter, i in zip("ABC", order):
            state["patch_" + letter] = cands[i][:PATCH_MAX]
        q = {"type": "choice", "instructions": "Which patch fixes this issue?", "criteria": ["patch A", "patch B", "patch C"]}
        yield _split(r["repo"], bench), json.dumps(state, ensure_ascii=False), q, order.index(0), None


def test_fail_rows(rng, bench) -> Iterator[ts.Row]:
    rows, _ = _instances(bench)
    for r in rows:
        f2p = [t for t in _tests(r.get("FAIL_TO_PASS")) if len(t) <= 160]
        p2p = [t for t in _tests(r.get("PASS_TO_PASS")) if len(t) <= 160 and t not in f2p]
        if not f2p or len(p2p) < 3 or not r["problem_statement"].strip():
            continue
        gold = rng.choice(f2p)
        opts = [gold] + _hardest(p2p, r["problem_statement"] + " " + r["patch"], 3, rng)
        if len(set(opts)) < 4:
            continue
        q, lab = ts._fixed_choice("Which test fails before this fix and passes after it?", opts, 0, rng)
        state = {"repository": r["repo"], "issue": r["problem_statement"][:1200], "fix": r["patch"][:PATCH_MAX]}
        yield _split(r["repo"], bench), json.dumps(state, ensure_ascii=False), q, lab, None


def _src(kind, bench):
    fn = {"file_loc": file_loc_rows, "patch_match": patch_match_rows, "test_fail": test_fail_rows}[kind]
    return lambda rng: fn(rng, bench)


TRAIN_LIC = "SWE-Gym MIT + SWE-rebench CC BY 4.0; permissive-licensed repositories only (per-instance license_name)"
BENCH_LIC = "SWE-bench MIT (GitHub SWE-bench/SWE-bench); pylint (GPL) excluded"
SOURCES: Dict[str, dict] = {}
for kind, use in (("file_loc", "code agent: which file must change to fix this issue"),
                  ("patch_match", "code agent: which diff fixes this issue"),
                  ("test_fail", "code agent: which test fails before the fix")):
    SOURCES["swe_" + kind] = {"fn": _src(kind, False), "repo": "nebius/SWE-rebench", "license": TRAIN_LIC, "use": use,
                              "train_cap": 20000}
    SOURCES["swebench_" + kind] = {"fn": _src(kind, True), "repo": "princeton-nlp/SWE-bench", "license": BENCH_LIC,
                                   "use": use + " (SWE-bench, eval only)", "eval_only": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (list(SOURCES) if a.names == ["all"] else a.names):
        try:
            ts.build(n, a.root, spec=SOURCES[n], log=lambda m: print(m, flush=True))
        except Exception as e:  # one broken source must not stop the rest
            print("%s: FAILED %r" % (n, e), flush=True)
