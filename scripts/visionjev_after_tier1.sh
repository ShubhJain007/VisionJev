#!/usr/bin/env bash
# Runs once checkpoints/qwen2b-mix is saved: log its training record (it started before the log hook existed),
# zero-shot eval of tier 1 (2B) and tier 0 (500M) on Open Images + Mind2Web, the cascade analysis on the 7 val
# sets, and the report. Every step logs itself to results/experiments.jsonl.
set -u
WS="$(cd "$(dirname "$0")/../.." && pwd)"; REPO="$WS/laya-vision"; PY="$REPO/.venv/bin/python"
unset PYTHONPATH; export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
log() { echo "[$(date '+%F %T')] $*"; }
until [ -f "$WS/checkpoints/qwen2b-mix/metrics.json" ]; do
  pgrep -f "run-name qwen2b-mix" >/dev/null || { log "training process gone without metrics.json"; exit 1; }
  sleep 30
done
log "qwen2b-mix saved"
cd "$REPO"
"$PY" - <<'PY'
import json, os
from laya.visionjev_log import log_experiment
d = "/home/kneepolean/shubhj/VisionJev/checkpoints/qwen2b-mix"
m = json.load(open(d + "/metrics.json")); args = m.get("args", {})
log_experiment("train", "qwen2b-mix", config=args, datasets=[x for x in args.get("datasets", "").split(",") if x],
               checkpoint=d, metrics={k: m[k] for k in ("steps", "best_step", "temperature", "val_raw", "val_calibrated",
               "val_shuffled_images", "selective_calibrated", "act_policies", "minutes") if k in m},
               artifacts=[d + "/metrics.json"], notes="logged after training (run started before the log hook)")
PY
log "zero-shot eval"
"$PY" scripts/visionjev_eval.py "$WS/checkpoints/qwen2b-mix" "$WS/checkpoints/mix-split1024" --kind zero_shot \
  --datasets openimages_presence,openimages_count,mind2web_element,mind2web_operation --tag zeroshot-oi-m2w
log "cascade"
"$PY" scripts/visionjev_cascade.py --small "$WS/checkpoints/mix-split1024" --large "$WS/checkpoints/qwen2b-mix" \
  --t-small 40 --t-large 120 --out "$WS/results/cascade"
log "report"
"$PY" scripts/visionjev_report.py
log "done"
