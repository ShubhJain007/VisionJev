"""Execution-verified code decisions: every label comes from running code, never from a model or a heuristic.

Sources (licenses checked 2026-09-27 on the Hub cards): MBPP full (CC BY 4.0), TACO (Apache-2.0), HumanEval (MIT),
CRUXEval (MIT).
Wrong programs are *mutants* of the reference solution: one AST edit (comparison ``<`` <-> ``<=``, ``==`` <-> ``!=``,
arithmetic ``+`` <-> ``-``, ``and`` <-> ``or``, an integer constant +-1, a boolean flipped, ``min`` <-> ``max``,
``any`` <-> ``all``). The reference and every mutant are re-printed with ``ast.unparse``, so formatting, comments and
whitespace cannot tell them apart; then each is executed against the tests in a separate, resource-limited Python
process and labelled by the result. A mutant that still passes every shown test is labelled as passing.

Sets (MBPP split by problem, 15% val; HumanEval and CRUXEval are eval-only so the public benchmarks stay clean):

* ``code_exec_pass``   noul    problem + tests + one implementation: will it pass all of these tests?
* ``code_exec_pair``   choice  problem + tests + implementations A and B (reference vs failing mutant): which passes?
* ``code_exec_output`` choice  function + call: what does it return? Distractors are what failing mutants return on
                               the same call, then the function's outputs on other calls, so every option is a value
                               some nearby program really produces.
* ``taco_exec_pass`` / ``taco_exec_pair``  the same on TACO (Apache-2.0) competitive-programming problems: whole
                       stdin/stdout programs, run on the (up to 3) test cases shown; TACO train -> train, test -> val.
* ``humaneval_exec``   eval-only pass + pair questions on HumanEval.
* ``cruxeval_output``  eval-only output prediction on CRUXEval.

    python -m laya.visionjev_codeexec code_exec_pass code_exec_pair code_exec_output humaneval_exec cruxeval_output
"""
import argparse
import ast
import copy
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional

from . import visionjev_textsets as ts
from .visionjev_data import DATA_ROOT, RAW_ROOT

MBPP = ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c")
HUMANEVAL = ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544")
CRUXEVAL = ("cruxeval-org/cruxeval", "b96af0450242eb4da433032b90998f25588a5d0f")
VAL_PCT = 15
MAX_MUTANTS = 14  # sampled per problem, before execution
TIMEOUT_S = 3

CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
       ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.Mod: ast.FloorDiv, ast.FloorDiv: ast.Mod,
       ast.Pow: ast.Mult}
BOOL = {ast.And: ast.Or, ast.Or: ast.And}
NAMES = {"min": "max", "max": "min", "any": "all", "all": "any"}

# ---- sandboxed execution ----------------------------------------------------------------------------------------

_HARNESS = r'''
import json, signal, sys
def _alarm(s, f):
    raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
job = json.loads(sys.stdin.read())
out = {"ok": True, "tests": [], "calls": []}
ns = {"__name__": "__candidate__"}
try:
    signal.alarm(%(t)d); exec(job["code"], ns); signal.alarm(0)
except BaseException as e:
    signal.alarm(0); out["ok"] = False; out["error"] = type(e).__name__
if out["ok"]:
    for t in job["tests"]:
        try:
            signal.alarm(%(t)d); exec(t, dict(ns)); signal.alarm(0); out["tests"].append(True)
        except BaseException:
            signal.alarm(0); out["tests"].append(False)
    for c in job["calls"]:
        try:
            signal.alarm(%(t)d); v = eval(c, dict(ns)); signal.alarm(0); out["calls"].append(repr(v)[:300])
        except BaseException as e:
            signal.alarm(0); out["calls"].append(None)
sys.stdout.write("\n\x00RESULT" + json.dumps(out) + "\n")
''' % {"t": TIMEOUT_S}


_HARNESS_STDIO = r'''
import io, json, signal, sys
def _alarm(s, f):
    raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
job = json.loads(sys.stdin.read())
real_out = sys.stdout
out = {"ok": True, "tests": [], "calls": []}
try:
    prog = compile(job["code"], "<program>", "exec")
except BaseException as e:
    out["ok"] = False; out["error"] = type(e).__name__; prog = None
for inp, exp in (job["cases"] if prog else []):
    sys.stdin = io.TextIOWrapper(io.BytesIO(inp.encode()))
    buf = io.StringIO(); sys.stdout = buf; good = True
    try:
        signal.alarm(%(t)d); exec(prog, {"__name__": "__main__"}); signal.alarm(0)
    except SystemExit:
        signal.alarm(0)
    except BaseException:
        signal.alarm(0); good = False
    sys.stdout = real_out
    out["tests"].append(good and buf.getvalue().split() == exp.split())
real_out.write("\n\x00RESULT" + json.dumps(out) + "\n")
''' % {"t": TIMEOUT_S}


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    # C-level loops ignore the per-test alarm; the CPU limit kills them (counted as not passing)
    resource.setrlimit(resource.RLIMIT_CPU, (2 * TIMEOUT_S + 2, 2 * TIMEOUT_S + 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    os.setsid()


def _no_network_prefix() -> List[str]:
    """``unshare -rn`` (unprivileged user + network namespace: no network at all) when the kernel allows it."""
    try:
        ok = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return ["unshare", "-rn"] if ok else []


NO_NETWORK = _no_network_prefix()


_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()
CACHE_PATH = os.path.join(RAW_ROOT, "codeexec", "runs.jsonl")


def _load_cache():
    if not _CACHE and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            for line in f:
                k, v = json.loads(line)
                _CACHE[k] = v


def run(code: str, tests: List[str] = (), calls: List[str] = (), stdio: Optional[List[List[str]]] = None) -> dict:
    """``_run`` memoised on disk by the exact job (code, tests, calls or stdin/stdout cases, harness), so rebuilding a
    set or building the next set from the same problems does not execute the same program again. With ``stdio``
    (``[[input, expected output], ...]``) the code is a whole program run once per case with that stdin; a case
    passes when its stdout matches the expected output token for token (``tests`` in the result)."""
    harness = _HARNESS_STDIO if stdio is not None else _HARNESS
    body = {"code": code, "cases": stdio} if stdio is not None else {"code": code, "tests": list(tests), "calls": list(calls)}
    job = json.dumps(body, sort_keys=True)
    key = hashlib.sha256((job + harness).encode()).hexdigest()
    with _CACHE_LOCK:
        _load_cache()
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    out = _run(job, harness, n=(len(stdio) if stdio is not None else len(tests) + len(calls)) + 1)
    with _CACHE_LOCK:
        _CACHE[key] = out
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "a") as f:
            f.write(json.dumps([key, out]) + "\n")
    return out


def _run(job: str, harness: str, n: int) -> dict:
    """Execute a harness job in a fresh isolated interpreter (``-I``, no network namespace when available, empty
    temp cwd, minimal env, memory/CPU/file-size limits, own session). Never raises."""
    with tempfile.TemporaryDirectory() as cwd:
        try:
            p = subprocess.run(NO_NETWORK + [sys.executable, "-I", "-c", harness], input=job, capture_output=True, text=True,
                               cwd=cwd, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                               timeout=TIMEOUT_S * n + 10, preexec_fn=_limits)
            line = [l for l in p.stdout.split("\n") if l.startswith("\x00RESULT")]
            if line:
                return json.loads(line[-1][len("\x00RESULT"):])
            return {"ok": False, "error": "no result (rc %s)" % p.returncode, "tests": [], "calls": []}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "tests": [], "calls": []}


def passes(r: dict, n_tests: int) -> bool:
    return bool(r.get("ok")) and len(r["tests"]) == n_tests and all(r["tests"])


# ---- mutation ---------------------------------------------------------------------------------------------------


def _sites(tree) -> List[tuple]:
    out = []
    for i, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Compare):
            out += [(i, "cmp", j) for j, op in enumerate(node.ops) if type(op) in CMP]
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in BIN:
            out.append((i, "bin", None))
        elif isinstance(node, ast.BoolOp):
            out.append((i, "bool", None))
        elif isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) < 10 ** 6:
            out += [(i, "int", 1), (i, "int", -1)]
        elif isinstance(node, ast.Constant) and type(node.value) is bool:
            out.append((i, "flip", None))
        elif isinstance(node, ast.Name) and node.id in NAMES and isinstance(node.ctx, ast.Load):
            out.append((i, "name", None))
    return out


def _apply(tree, site):
    i, kind, arg = site
    node = next(n for k, n in enumerate(ast.walk(tree)) if k == i)
    if kind == "cmp":
        node.ops[arg] = CMP[type(node.ops[arg])]()
    elif kind == "bin":
        node.op = BIN[type(node.op)]()
    elif kind == "bool":
        node.op = BOOL[type(node.op)]()
    elif kind == "int":
        node.value = node.value + arg
    elif kind == "flip":
        node.value = not node.value
    elif kind == "name":
        node.id = NAMES[node.id]
    return tree


def canonical(src: str) -> Optional[str]:
    try:
        return ast.unparse(ast.parse(src.replace("\r\n", "\n").replace("\t", "    ")))
    except (SyntaxError, ValueError):
        return None


def mutants(src: str, rng: random.Random, k: int = MAX_MUTANTS) -> List[str]:
    """Up to ``k`` distinct single-edit mutants of canonical source ``src`` (also canonical)."""
    tree = ast.parse(src)
    sites = _sites(tree)
    rng.shuffle(sites)
    out, seen = [], {src}
    for s in sites:
        try:
            m = ast.unparse(_apply(copy.deepcopy(tree), s))
            compile(m, "<mutant>", "exec")
        except Exception:
            continue
        if m not in seen:
            seen.add(m)
            out.append(m)
        if len(out) >= k:
            break
    return out


# ---- problems ---------------------------------------------------------------------------------------------------


def _split(key: str) -> str:
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < VAL_PCT else "train"


def _assert_call(test: str) -> Optional[str]:
    """``assert f(x) == y`` -> ``f(x)``."""
    try:
        node = ast.parse(test).body[0]
    except (SyntaxError, IndexError):
        return None
    if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 \
            and isinstance(node.test.ops[0], ast.Eq) and isinstance(node.test.left, ast.Call):
        return ast.unparse(node.test.left)
    return None


def mbpp_problems() -> Iterator[dict]:
    for hub in ("train", "validation", "test", "prompt"):
        for ex in ts._ds(MBPP[0], "full", split=hub, revision=MBPP[1]):
            code = canonical((ex["test_setup_code"] or "") + "\n" + ex["code"])
            if code:
                calls = [c for c in map(_assert_call, ex["test_list"]) if c]
                yield {"key": "mbpp-%d" % ex["task_id"], "split": _split("mbpp-%d" % ex["task_id"]),
                       "problem": ex["text"], "code": code, "tests": list(ex["test_list"]),
                       "tests_shown": "\n".join(ex["test_list"]), "calls": calls}


def humaneval_problems() -> Iterator[dict]:
    for ex in ts._ds(HUMANEVAL[0], split="test", revision=HUMANEVAL[1]):
        code = canonical(ex["prompt"] + ex["canonical_solution"])
        test = canonical(ex["test"])
        if not code or not test:
            continue
        test = "\n".join(l for l in test.split("\n") if not l.startswith("METADATA")) \
            if "METADATA" not in test else ast.unparse(ast.Module(
                body=[n for n in ast.parse(test).body if not (isinstance(n, ast.Assign) and
                      any(getattr(t, "id", "") == "METADATA" for t in n.targets))], type_ignores=[]))
        shown = test + "\n\ncheck(%s)" % ex["entry_point"]
        yield {"key": ex["task_id"], "split": "val", "problem": ex["prompt"].strip()[:1200], "code": code,
               "tests": [shown], "tests_shown": shown, "calls": []}


def _evaluate_all(problems, pool: ThreadPoolExecutor) -> list:
    """Every problem with ``ref`` (the reference's result) and ``muts`` ([(mutant, result)]). Mutants are drawn from
    a per-problem seed, so every set built from the same problems runs the same programs (and hits the run cache);
    all candidates of all problems go through one pool, so a slow mutant never blocks the others."""
    problems = list(problems)
    jobs = []
    for prob in problems:
        cands = [prob["code"]] + mutants(prob["code"], random.Random(prob["key"]))
        jobs += [(prob, i, c) for i, c in enumerate(cands)]
    results = list(pool.map(lambda j: run(j[2], j[0]["tests"], j[0]["calls"]) if not j[0].get("stdio")
                            else run(j[2], stdio=j[0]["tests"]), jobs))
    per = {}
    for (prob, i, c), r in zip(jobs, results):
        per.setdefault(id(prob), []).append((c, r))
    for prob in problems:
        (_, prob["ref"]), *prob["muts"] = per[id(prob)]
    return problems


PASS_Q = ("Will this implementation pass all of the given tests?", "Does this code pass every one of these tests?",
          "Do all of the listed tests pass for this implementation?")
PAIR_Q = ("Which implementation passes all of the given tests?", "Which of the two implementations passes every test?")
OUT_Q = ("What does this call return?", "What is the value of this expression?", "What does the call evaluate to?")


def _problems_rows(problems, rng, kinds, pool) -> Iterator[ts.Row]:
    for prob in _evaluate_all(problems, pool):
        ref, muts = prob["ref"], prob["muts"]
        n = len(prob["tests"])
        if not passes(ref, n):
            continue  # a reference that fails its own tests (or crashes) is dropped with all its mutants
        ok = [m for m, r in muts if passes(r, n)]
        bad = [(m, r) for m, r in muts if r.get("ok") is not None and not passes(r, n)]
        base = {"problem": prob["problem"], "tests": prob["tests_shown"][:1500]}
        if "pass" in kinds:
            for code, lab in [(prob["code"], 1)] + [(m, 1) for m in ok[:2]] + [(m, 0) for m, _ in bad[:4]]:
                if len(code) <= 2500:
                    yield prob["split"], json.dumps(dict(base, implementation=code)), ts._noul(rng.choice(PASS_Q)), lab, None
        if "pair" in kinds:
            for m, _ in bad[:3]:
                if len(prob["code"]) > 1500 or len(m) > 1500:
                    continue
                pair = [prob["code"], m]
                gold = rng.randrange(2)
                a, b = (pair[0], pair[1]) if gold == 0 else (pair[1], pair[0])
                q = {"type": "choice", "instructions": rng.choice(PAIR_Q),
                     "criteria": ["implementation A", "implementation B"]}
                yield prob["split"], json.dumps(dict(base, implementation_A=a, implementation_B=b)), q, gold, None
        if "output" in kinds:
            yield from _output_rows(prob["split"], prob["code"], prob["calls"], ref["calls"],
                                    [r["calls"] for _, r in bad if r.get("ok")], rng)


def _usable(v: Optional[str]) -> bool:
    return v is not None and len(v) <= 80 and " at 0x" not in v and "object at" not in v


def _num(v: str) -> Optional[float]:
    try:
        return float(v)
    except ValueError:
        return None


def _numeric_distractors(true: str, pool: List[str], rng: random.Random, k: int = 3) -> Optional[List[str]]:
    """For a numeric answer, up to ``k`` numeric distractors chosen so the true value's rank among the options is
    uniform: +-1 mutants otherwise bracket it and "pick the median" finds it (audit). None when not numeric."""
    t = _num(true)
    vals = [(v, _num(v)) for v in pool]
    if t is None or any(x is None for _, x in vals):
        return None
    below = [v for v, x in vals if x < t]
    above = [v for v, x in vals if x > t]
    n = min(k, len(below) + len(above))
    if n == 0:
        return []
    feasible = [r for r in range(n + 1) if r <= len(below) and n - r <= len(above)]
    r = rng.choice(feasible)  # how many options sit below the true value
    return rng.sample(below, r) + rng.sample(above, n - r)


def _output_rows(split, code, calls, ref_vals, mutant_vals, rng, max_per=3) -> Iterator[ts.Row]:
    if len(code) > 2000:
        return
    for ci, (call, true) in enumerate(list(zip(calls, ref_vals))[:max_per]):
        if not _usable(true):
            continue
        near = [mv[ci] for mv in mutant_vals if ci < len(mv) and _usable(mv[ci]) and mv[ci] != true]
        other = [v for j, v in enumerate(ref_vals) if j != ci and _usable(v) and v != true]
        opts = _numeric_distractors(true, list(dict.fromkeys(near + other)), rng)
        if opts is None:
            opts = list(dict.fromkeys(near))
            rng.shuffle(opts)
            opts = opts[:3]
            for v in other:
                if len(opts) >= 3:
                    break
                if v not in opts:
                    opts.append(v)
        if len(opts) < 2:
            continue
        opts = [true] + opts
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": rng.choice(OUT_Q), "criteria": opts}
        yield split, json.dumps({"code": code, "call": call}), q, opts.index(true), None


TACO = ("BAAI/TACO", "Apache-2.0")
TACO_TRAIN_SHARDS = 9  # all ~25k problems
TACO_PROBLEMS = {"train": 12000, "val": 600}


def taco_problems(split: str, rng: random.Random) -> Iterator[dict]:
    """TACO stdin/stdout problems with a Python reference: up to 3 short test cases are shown and executed."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    files = (["ALL/train-%05d-of-00009.parquet" % i for i in range(TACO_TRAIN_SHARDS)] if split == "train"
             else ["ALL/test-00000-of-00001.parquet"])
    probs = []
    for f in files:
        t = pq.read_table(hf_hub_download(TACO[0], f, repo_type="dataset"),
                          columns=["question", "solutions", "input_output", "url"])
        for q, sols, io_, url in zip(*(t.column(c).to_pylist() for c in ("question", "solutions", "input_output", "url"))):
            try:
                io_, sols = json.loads(io_ or "{}"), json.loads(sols or "[]")
            except ValueError:
                continue
            if "fn_name" in io_ or not sols:
                continue
            cases = [[i, o] for i, o in zip(io_.get("inputs", []), io_.get("outputs", []))
                     if isinstance(i, str) and isinstance(o, str) and len(i) + len(o) <= 400][:3]
            code = next((c for c in map(canonical, sols) if c and len(c) <= 1500), None)
            if len(cases) < 2 or not code:
                continue
            shown = "\n\n".join("Input:\n%s\nExpected output:\n%s" % (i.strip(), o.strip()) for i, o in cases)
            probs.append({"key": "taco-%s" % (url or hashlib.md5(q.encode()).hexdigest()), "split": split,
                          "problem": q[:1500], "code": code, "tests": cases, "tests_shown": shown, "calls": [],
                          "stdio": True})
    rng.shuffle(probs)
    return iter(probs[:TACO_PROBLEMS[split]])


def _taco(kinds):
    def fn(rng):
        with ThreadPoolExecutor(WORKERS) as pool:
            for split in ("train", "val"):
                yield from _problems_rows(taco_problems(split, rng), rng, kinds, pool)
    return fn


WORKERS = max(4, (os.cpu_count() or 8) - 4)


def _mbpp(kinds):
    def fn(rng):
        with ThreadPoolExecutor(WORKERS) as pool:
            yield from _problems_rows(mbpp_problems(), rng, kinds, pool)
    return fn


def humaneval_exec(rng):
    with ThreadPoolExecutor(WORKERS) as pool:
        yield from _problems_rows(humaneval_problems(), rng, ("pass", "pair"), pool)


def cruxeval_output(rng):
    """CRUXEval gives (f, input, output); the output is re-derived by execution, distractors from mutants of f."""
    probs = []
    for ex in ts._ds(CRUXEVAL[0], split="test", revision=CRUXEVAL[1]):
        code = canonical(ex["code"])
        if code:
            probs.append({"key": ex["id"], "code": code, "tests": [], "calls": ["f(%s)" % ex["input"]]})
    with ThreadPoolExecutor(WORKERS) as pool:
        for prob in _evaluate_all(probs, pool):
            ref = prob["ref"]
            if not ref.get("ok") or not ref["calls"] or ref["calls"][0] is None:
                continue
            yield from _output_rows("val", prob["code"], prob["calls"], ref["calls"],
                                    [r["calls"] for _, r in prob["muts"] if r.get("ok")], rng, max_per=1)


SOURCES: Dict[str, dict] = {
    "code_exec_pass": {"fn": _mbpp(("pass",)), "repo": MBPP[0], "license": "CC BY 4.0 (MBPP); labels by execution",
                       "use": "code: will this implementation pass these tests?", "balance": True},
    "code_exec_pair": {"fn": _mbpp(("pair",)), "repo": MBPP[0], "license": "CC BY 4.0 (MBPP); labels by execution",
                       "use": "code: which of two implementations passes the tests?"},
    "code_exec_output": {"fn": _mbpp(("output",)), "repo": MBPP[0],
                         "license": "CC BY 4.0 (MBPP); outputs by execution", "use": "code: what does this call return?"},
    "taco_exec_pass": {"train_cap": 40000, "fn": _taco(("pass",)), "repo": TACO[0], "license": "Apache-2.0 (TACO); labels by execution",
                       "use": "code: will this stdin/stdout program pass these tests?", "balance": True},
    "taco_exec_pair": {"train_cap": 25000, "fn": _taco(("pair",)), "repo": TACO[0], "license": "Apache-2.0 (TACO); labels by execution",
                       "use": "code: which of two programs passes the tests?"},
    "humaneval_exec": {"fn": humaneval_exec, "repo": HUMANEVAL[0], "license": "MIT; labels by execution",
                       "use": "code: pass / pair on HumanEval (eval only)", "eval_only": True},
    "cruxeval_output": {"fn": cruxeval_output, "repo": CRUXEVAL[0], "license": "MIT; outputs by execution",
                        "use": "code: output prediction on CRUXEval (eval only)", "eval_only": True},
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="names from SOURCES, or 'all'")
    ap.add_argument("--root", default=DATA_ROOT)
    a = ap.parse_args()
    for n in (list(SOURCES) if a.names == ["all"] else a.names):
        ts.build(n, a.root, spec=SOURCES[n], log=lambda m: print(m, flush=True))
