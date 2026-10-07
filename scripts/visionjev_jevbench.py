"""Run a VisionJev checkpoint on JevBench's public tasks with JevBench's own runner and scorer, and log the result.

JevBench (github.com/fstandhartinger/jevbench, MIT) ships a ``laya_local`` adapter that calls ``laya.load(path)`` and
``agent.predict(state, {"decision": question})``; our ``VLMAgent`` answers in the same Jev shape, so this script only
swaps the loader (``laya.load_vlm``) and then runs their unmodified ``jevbench run`` / ``summarize``. Only the public
tiers can be run locally; official rank needs the maintainers to run the sealed tasks.

    .venv/bin/python scripts/visionjev_jevbench.py ../checkpoints/qwen2b-mix --jevbench ~/jevbench --device cuda
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--jevbench", required=True, help="path to a clone of fstandhartinger/jevbench")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(WORKSPACE, "results", "jevbench"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thinking", choices=["off", "auto", "always"], default="off",
                    help="think-if-unsure: a second pass with the model's own working when the first is unsure")
    ap.add_argument("--think-threshold", type=float, default=0.7)
    ap.add_argument("--think-max-tokens", type=int, default=160)
    ap.add_argument("--max-len", type=int, default=0,
                    help="raise the checkpoint's input token budget (0 = keep it); long states are otherwise cut")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.jevbench)
    import jevbench.cli as C
    from jevbench.adapters.laya_local import LayaLocalAdapter

    from laya.visionjev_log import log_experiment

    device = a.device

    class VisionJevAdapter(LayaLocalAdapter):
        name = "visionjev_local"
        cost_basis = "local_gpu_no_provider_tariff" if device == "cuda" else "local_cpu_no_provider_tariff"

        def load(self):
            if self._agent is None:
                import laya

                self._agent = laya.load_vlm(self.path, device=device)
                if a.max_len:
                    self._agent.cfg["max_len"] = self._agent.processor.laya_max_len = a.max_len
                if a.thinking != "off":
                    import functools

                    self._agent.predict = functools.partial(self._agent.predict, thinking={
                        "mode": a.thinking, "threshold": a.think_threshold, "max_tokens": a.think_max_tokens})
                self.laya_version = "visionjev " + getattr(laya, "__version__", "")
            return self._agent

    C.LayaLocalAdapter = VisionJevAdapter  # the CLI builds its adapter table at call time
    run = os.path.basename(os.path.normpath(a.checkpoint)) + ("-len%d" % a.max_len if a.max_len else "")
    if a.thinking != "off":
        run += "-think%s%d" % ("" if a.thinking == "auto" else "all", round(100 * a.think_threshold))
    out = os.path.join(a.out, run)
    os.makedirs(out, exist_ok=True)
    tasks = os.path.join(out, "public_tasks.jsonl")
    with open(tasks, "w") as f:
        for p in sorted(glob.glob(os.path.join(a.jevbench, "datasets", "public", "*.jsonl"))):
            f.writelines(l for l in open(p) if l.strip())
    results = os.path.join(out, "results.jsonl")
    # a re-run starts clean: the ledger keeps the first cap it ever saw, and jevbench refuses to overwrite raw files
    for stale in (results, os.path.join(out, "ledger.jsonl"), os.path.join(out, "summary.json"),
                  os.path.join(out, "manifest.json")):
        if os.path.exists(stale):
            os.remove(stale)
    shutil.rmtree(os.path.join(out, "raw"), ignore_errors=True)
    os.environ["JEVBENCH_WARM_LOAD"] = "1"
    t0 = time.time()
    rc = C.main(["run", "--tasks", tasks, "--adapter", "laya_local", "--endpoint", a.checkpoint, "--model",
                 "visionjev/" + run, "--results", results, "--ledger", os.path.join(out, "ledger.jsonl"),
                 "--raw-dir", os.path.join(out, "raw"), "--cap-usd", "1", "--reserve-usd", "0",  # local: nothing is charged; a zero cap blocks every request
                 "--run-label", "visionjev/" + run,
                 "--manifest", os.path.join(out, "manifest.json")] + (["--limit", str(a.limit)] if a.limit else []))
    summary = os.path.join(out, "summary.json")
    C.main(["summarize", "--tasks", tasks, "--results", results, "--public-export", summary])
    metrics = json.load(open(summary)) if os.path.exists(summary) else {}
    log_experiment("jevbench_public", "jevbench-public:%s" % run,
                   config={"checkpoint": a.checkpoint, "device": device, "jevbench_commit": os.popen(
                       "git -C %s rev-parse --short HEAD" % a.jevbench).read().strip(), "tasks": "public tiers only", "max_len": a.max_len or "checkpoint"},
                   metrics={"run_rc": rc, "minutes": round((time.time() - t0) / 60, 1), "summary": metrics},
                   checkpoint=a.checkpoint, artifacts=[results, summary])
    return rc


if __name__ == "__main__":
    main()
