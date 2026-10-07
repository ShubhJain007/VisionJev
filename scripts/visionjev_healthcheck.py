"""Health check of a running (or finished) visionjev_finetune.py log, in the spirit of autoresearch: decide from the
numbers, early, whether a run is worth continuing, so a broken run is killed after minutes rather than hours.

Checks (exit 0 = healthy, 2 = unhealthy, 3 = too early to judge, 4 = log not found):
  fatal   Traceback / out-of-memory / Killed in the log; non-finite loss or gradient norm; loss diverging (mean of the
          last window > 1.5x the first window after >= 100 steps); more than 30% of logged steps flagged as spikes;
          data-loader wait above 40%; the baseline eval (step 0) below --min-acc0; no step line --min-wait minutes
          after the baseline eval; slower than --max-s-per-step.
  warn    reward not improving, checkpointing on every step (padded tokens above the threshold), eval accuracy dropping.

    .venv/bin/python scripts/visionjev_healthcheck.py data/logs/b3-b3.log
"""
import argparse
import json
import math
import os
import re
import sys
import time

STEP = re.compile(r"^step (\d+) \| ([0-9.]+) min \| loss ([-0-9.einfa]+) \(avg ([-0-9.einfa]+)\) \| reward ([-0-9.einfa]+) "
                  r"\| lr ([0-9.e+-]+) \| w_ce [0-9.]+ \| sigma [0-9.]+ \| grad ([-0-9.einfa]+) \| data wait (\d+)%")
EVAL = re.compile(r"^\[eval step (\d+)[^\]]*\] all n=(\d+) acc=([0-9.]+) ece=([0-9.]+) nll=([0-9.]+)")


def _f(x):
    try:
        v = float(x)
    except ValueError:
        return float("nan")
    return v


def analyse(text: str, min_wait: float = 8.0, min_acc0: float = 0.2, max_s_per_step: float = 0.0,
            now: float = None, mtime: float = None) -> dict:
    steps, evals, spikes = [], [], 0
    fatal, warn = [], []
    for line in text.splitlines():
        m = STEP.match(line)
        if m:
            steps.append({"step": int(m.group(1)), "min": float(m.group(2)), "loss": _f(m.group(3)), "avg": _f(m.group(4)),
                          "reward": _f(m.group(5)), "grad": _f(m.group(7)), "wait": int(m.group(8))})
            continue
        m = EVAL.match(line)
        if m:
            evals.append({"step": int(m.group(1)), "acc": float(m.group(3)), "ece": float(m.group(4)), "nll": float(m.group(5))})
        elif line.startswith("loss spike step"):
            spikes += 1
        elif re.search(r"Traceback|OutOfMemoryError|CUDA out of memory|Killed|NaN loss", line):
            fatal.append("error in log: " + line.strip()[:160])
    if evals and evals[0]["step"] == 0 and evals[0]["acc"] < min_acc0:
        fatal.append("baseline eval accuracy %.3f below %.2f: wrong checkpoint or broken inputs" % (evals[0]["acc"], min_acc0))
    if not steps:
        if evals and mtime is not None and now is not None and now - mtime > min_wait * 60:
            fatal.append("no training step logged %.0f minutes after the last log line" % ((now - mtime) / 60))
        if not fatal:
            return {"verdict": "early", "steps": 0, "fatal": [], "warn": warn}
    else:
        if any(not math.isfinite(s["loss"]) or not math.isfinite(s["grad"]) for s in steps):
            fatal.append("non-finite loss or gradient norm")
        n = len(steps)
        if steps[-1]["step"] >= 100 and n >= 4:
            k = max(1, n // 4)
            first, last = sum(s["avg"] for s in steps[:k]) / k, sum(s["avg"] for s in steps[-k:]) / k
            if last > 1.5 * first and last > 0.5:
                fatal.append("loss diverging: last-window avg %.3f vs first-window %.3f" % (last, first))
            elif last > 0.98 * first:
                warn.append("loss flat: last-window avg %.3f vs first-window %.3f" % (last, first))
        logged = max(1, steps[-1]["step"] - steps[0]["step"] + 1)
        if spikes > 0.3 * logged:
            fatal.append("%d loss spikes over %d steps" % (spikes, logged))
        wait = sum(s["wait"] for s in steps[-5:]) / min(5, n)
        if wait > 40:
            fatal.append("data-loader wait %.0f%%" % wait)
        if n >= 2:
            sps = (steps[-1]["min"] - steps[0]["min"]) * 60 / max(1, steps[-1]["step"] - steps[0]["step"])
            if max_s_per_step and sps > max_s_per_step:
                fatal.append("%.1f s/step, slower than %.1f" % (sps, max_s_per_step))
        else:
            sps = None
        if n >= 4 and steps[-1]["reward"] <= steps[0]["reward"] - 0.05:
            warn.append("reward fell from %.3f to %.3f" % (steps[0]["reward"], steps[-1]["reward"]))
    if len(evals) >= 3 and evals[-1]["acc"] < evals[-2]["acc"] - 0.03 and evals[-1]["acc"] < evals[-3]["acc"] - 0.03:
        warn.append("eval accuracy dropping: %s" % [e["acc"] for e in evals[-3:]])
    out = {"verdict": "unhealthy" if fatal else "healthy", "steps": steps[-1]["step"] if steps else 0,
           "minutes": steps[-1]["min"] if steps else 0.0, "s_per_step": round(sps, 2) if steps and sps else None,
           "loss_first": steps[0]["avg"] if steps else None, "loss_last": steps[-1]["avg"] if steps else None,
           "grad_last": steps[-1]["grad"] if steps else None, "data_wait_last": steps[-1]["wait"] if steps else None,
           "spikes": spikes, "evals": [(e["step"], e["acc"], e["ece"]) for e in evals], "fatal": fatal, "warn": warn}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--min-wait", type=float, default=8.0, help="minutes without a step after the baseline eval = fatal")
    ap.add_argument("--min-acc0", type=float, default=0.2)
    ap.add_argument("--max-s-per-step", type=float, default=0.0)
    a = ap.parse_args(argv)
    if not os.path.exists(a.log):
        print(json.dumps({"verdict": "no log"}))
        return 4
    res = analyse(open(a.log, errors="replace").read(), a.min_wait, a.min_acc0, a.max_s_per_step,
                  now=time.time(), mtime=os.path.getmtime(a.log))
    print(json.dumps(res))
    print("verdict: %s | steps %s | %.1f min | %s s/step | loss %s -> %s | grad %s | wait %s%% | spikes %d | fatal %s | warn %s"
          % (res["verdict"], res["steps"], res.get("minutes", 0), res.get("s_per_step"), res.get("loss_first"),
             res.get("loss_last"), res.get("grad_last"), res.get("data_wait_last"), res["spikes"], res["fatal"], res["warn"]))
    return {"healthy": 0, "unhealthy": 2, "early": 3}[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
