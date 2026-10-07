#!/usr/bin/env bash
# Hands-off pod run, started once code + data are on the pod:
#   setup -> model download -> smoke test (40 steps) -> full train + evals (visionjev_pod_run.sh <run> train).
# Any failure before the full run appends "EXIT_CODE=99" to the run's train log, which is what the local watcher
# (pod_fetch_results.sh) waits for, so the logs are fetched and the pod gets deleted instead of idling.
#
#   bash scripts/pod_orchestrate.sh <run> "<train args>" "<full-run eval args>"
set -uo pipefail
RUN="$1"; TRAIN_ARGS="$2"; EVAL_ARGS="$3"
WS=/workspace/VisionJev
LOGS=$WS/data/logs
TLOG=$LOGS/$RUN.train.log
mkdir -p "$LOGS"
fail() {
  echo "ORCHESTRATION FAILED at $1 ($(date -u +%H:%M)); see $LOGS/$RUN.orchestrate.log" | tee -a "$TLOG"
  echo "EXIT_CODE=99" >> "$TLOG"
  exit 1
}
echo "[$(date -u +%H:%M)] setup"
bash $WS/laya-vision/scripts/setup_runpod.sh || fail setup
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export VISIONJEV_DATA=$WS/data/vqa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -u +%H:%M)] model download"
/workspace/venv/bin/hf download Qwen/Qwen3-VL-30B-A3B-Instruct > /dev/null || fail model-download
echo "[$(date -u +%H:%M)] smoke test"
cd $WS/laya-vision
# shellcheck disable=SC2086
/workspace/venv/bin/python scripts/visionjev_finetune.py --run-name $RUN-smoke --out $WS/checkpoints $TRAIN_ARGS \
  --steps 40 --eval-every 1000 --eval-per-dataset 4 --n-calib 4 --max-val 4 --no-keep-best \
  > $LOGS/$RUN.smoke.log 2>&1 || fail smoke
grep -q "^saved " $LOGS/$RUN.smoke.log || fail smoke-no-checkpoint
grep "^step " $LOGS/$RUN.smoke.log | tail -2
echo "[$(date -u +%H:%M)] full run"
# shellcheck disable=SC2086
bash scripts/visionjev_pod_run.sh $RUN train $TRAIN_ARGS $EVAL_ARGS
