#!/usr/bin/env bash
# Eval-only pod run for existing checkpoints under /workspace/VisionJev/checkpoints: the same suite as pod_night_4b.sh
# (JevBench public at 4096 first, then zero-shot, never-seen, held-out wording, robustness, latency) so results compare
# one to one. The last line of $DONE is always "EXIT_CODE=<rc>" (trap), for the local fetch watcher (PROBLEMS I5).
#
#   setsid nohup bash scripts/pod_eval.sh <tag> <checkpoint-name>... > /dev/null 2>&1 < /dev/null &
set -uo pipefail
TAG=$1; shift
WS=/workspace/VisionJev
LOGS=$WS/data/logs
DONE=$LOGS/$TAG.log
mkdir -p "$LOGS"
rc=0
trap 'echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/$TAG-$name.log" 2>&1
  local r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export VISIONJEV_DATA=$WS/data/vqa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv/bin/python

step setup bash $WS/laya-vision/scripts/setup_runpod.sh || exit
REV=$(python3 -c "import json,sys; print(json.load(open('$WS/checkpoints/$1/vlm_agent_config.json'))['backbone_revision'])")
BB=$(python3 -c "import json,sys; print(json.load(open('$WS/checkpoints/$1/vlm_agent_config.json'))['backbone'])")
step download /workspace/venv/bin/hf download "$BB" --revision "$REV" || exit
cd $WS/laya-vision
ZS="openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts"
ROB="aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence"
HOF=$(ls $WS/data/vqa | grep '_hof$' | paste -sd, -)
for R in "$@"; do
  CK=$WS/checkpoints/$R
  step $R-jevbench $PY scripts/visionjev_jevbench.py $CK --jevbench /workspace/jevbench --device cuda --max-len 4096 \
    --out $WS/results/jevbench
  step $R-zeroshot $PY scripts/visionjev_eval.py $CK --kind zero_shot --datasets $ZS --tag zeroshot
  step $R-novel $PY scripts/visionjev_eval.py $CK --datasets novel_text,novel_vision --kind never_seen --tag novel \
    --no-shuffle-control
  step $R-hof $PY scripts/visionjev_eval.py $CK --kind heldout_formulation --datasets $HOF --tag hof
  step $R-robust $PY scripts/visionjev_robustness.py $CK --datasets $ROB --n 100 --tag robust
  step $R-latency $PY scripts/visionjev_latency.py $CK --datasets aokvqa,vqav2_yesno,doclaynet_type,synth_ui --n 100
done
