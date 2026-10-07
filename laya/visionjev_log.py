"""The VisionJev experiment log: every eval, test, benchmark and graph is recorded in ONE append-only file,
``<workspace>/results/experiments.jsonl`` (one JSON object per line), with everything needed to review it later:

    {"id", "time", "kind" (train | eval | zero_shot | latency | cascade | test | ...), "name",
     "config": {...the exact arguments...}, "checkpoint": {...its vlm_agent_config.json...} | null,
     "datasets": {name: {"revision", "license", "counts"}}, "metrics": {...}, "artifacts": [paths to graphs/files],
     "env": {"git_commit", "git_dirty", "host", "gpu", "torch", "transformers", "python"}, "notes": str}

Records are never rewritten or deleted (results are create-only, as in AGENTS.md); a re-run is a new record.
``scripts/visionjev_report.py`` renders the file into ``results/EXPERIMENTS.md``.
"""
import json
import os
import platform
import subprocess
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))
LOG = os.environ.get("VISIONJEV_LOG", os.path.join(WORKSPACE, "results", "experiments.jsonl"))
DATA_ROOT = os.path.join(WORKSPACE, "data", "vqa")


def _git(*args) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", os.path.join(HERE, ".."), *args], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return None


def environment() -> Dict[str, Any]:
    env = {"git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
           "host": platform.node(), "python": platform.python_version()}
    try:
        import torch
        import transformers

        env.update(torch=torch.__version__, transformers=transformers.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    return env


def dataset_info(names: Iterable[str], root: str = DATA_ROOT) -> Dict[str, Any]:
    out = {}
    for n in names:
        path = os.path.join(root, n, "manifest.json")
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            out[n] = {"source": m.get("source"), "revision": m.get("revision"), "license": m.get("license"),
                      "counts": m.get("counts")}
        else:
            out[n] = None
    return out


def checkpoint_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    cfg = os.path.join(path, "vlm_agent_config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            return dict(json.load(f), path=os.path.abspath(path))
    return {"path": path}


def log_experiment(kind: str, name: str, config: Dict[str, Any], metrics: Dict[str, Any],
                   datasets: Iterable[str] = (), checkpoint: Optional[str] = None, artifacts: List[str] = (),
                   notes: str = "", log: str = LOG) -> Dict[str, Any]:
    """Append one record to the experiment log and return it."""
    rec = {"id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, "name": name,
           "config": config, "checkpoint": checkpoint_config(checkpoint), "datasets": dataset_info(datasets),
           "metrics": metrics, "artifacts": [os.path.abspath(a) for a in artifacts], "env": environment(),
           "notes": notes}
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print("logged experiment %s (%s/%s) -> %s" % (rec["id"], kind, name, log), flush=True)
    return rec


def read_log(log: str = LOG) -> List[Dict[str, Any]]:
    if not os.path.exists(log):
        return []
    with open(log) as f:
        return [json.loads(line) for line in f if line.strip()]
