#!/usr/bin/env bash
# One-night Qwen3-VL-4B run (Hypothesis/V6.1.md, Appendix F, 2026-09-29), hands-off on one pod:
#   setup -> model download -> smoke -> Phase A (broad, mix_v6) -> wait for the hard set -> Phase B (short delta from A,
#   mix_v6_hard) -> evals of A and B (JevBench at 4096 first, then zero-shot, held-out wording, never-seen,
#   robustness, latency).
# The last line of $DONE is always "EXIT_CODE=<rc>" (trap), whatever fails, so the local fetch watcher never waits on
# an idle pod (PROBLEMS I5).
#
#   setsid nohup bash scripts/pod_night_4b.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
DONE=$LOGS/night4b.log
A=qwen4b-v6a
B=qwen4b-v6b
A_MINUTES=${A_MINUTES:-480}
B_MINUTES=${B_MINUTES:-60}
HARD_WAIT_MIN=${HARD_WAIT_MIN:-90}
mkdir -p "$LOGS"
rc=0
trap 'echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {  # step <name> <command...>: own log per step; remember the first failure
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/night4b-$name.log" 2>&1
  local r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}

export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export VISIONJEV_DATA=$WS/data/vqa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv/bin/python
BACKBONE=Qwen/Qwen3-VL-4B-Instruct
COMMON="--batch-size 16 --dtype bf16 --option-attention causal --w-ce-schedule const --grad-checkpointing \
  --qwen-max-pixels 589824 --max-len 4096 --n-calib 40 --max-val 150 --eval-per-dataset 30 --num-workers 16 \
  --augment --out $WS/checkpoints"
A_ARGS="--backbone $BACKBONE --mix-config configs/mix_v6.json --freeze lora --lora-rank 32 --lr-head 1e-4 \
  --lr-backbone 5e-5 --logit-cap 10 --eval-every 1500 --patience 4 $COMMON"
B_ARGS="--init-checkpoint $WS/checkpoints/$A --mix-config configs/mix_v6_hard.json --freeze lora --lr-head 5e-5 \
  --lr-backbone 2e-5 --eval-every 400 $COMMON"

step setup bash $WS/laya-vision/scripts/setup_runpod.sh || exit
step download /workspace/venv/bin/hf download $BACKBONE || exit
cd $WS/laya-vision
# shellcheck disable=SC2086
step smoke $PY scripts/visionjev_finetune.py --run-name $A-smoke $A_ARGS --steps 40 --eval-every 1000 \
  --eval-per-dataset 4 --n-calib 4 --max-val 4 --no-keep-best || exit
grep "^step " $LOGS/night4b-smoke.log | tail -2 >> "$DONE"

# shellcheck disable=SC2086
step phase-a $PY scripts/visionjev_finetune.py --run-name $A $A_ARGS --minutes $A_MINUTES
[ -f $WS/checkpoints/$A/vlm_agent_config.json ] || { note "no Phase A checkpoint"; exit; }

waited=0
until [ -f $WS/data/vqa/HARD_READY ] && [ -f configs/mix_v6_hard.json ] || [ $waited -ge $HARD_WAIT_MIN ]; do
  sleep 60; waited=$((waited + 1))
done
CKPTS=$WS/checkpoints/$A
if [ -f $WS/data/vqa/HARD_READY ] && [ -f configs/mix_v6_hard.json ]; then
  # shellcheck disable=SC2086
  step phase-b $PY scripts/visionjev_finetune.py --run-name $B $B_ARGS --minutes $B_MINUTES \
    && [ -f $WS/checkpoints/$B/vlm_agent_config.json ] && CKPTS="$CKPTS $WS/checkpoints/$B"
else
  note "hard set not ready after $HARD_WAIT_MIN min: Phase B skipped"
fi

ZS="openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts"
ROB="aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence"
HOF=$(ls $WS/data/vqa | grep '_hof$' | paste -sd, -)
for CK in $CKPTS; do
  R=$(basename $CK)
  step $R-jevbench $PY scripts/visionjev_jevbench.py $CK --jevbench /workspace/jevbench --device cuda --max-len 4096 \
    --out $WS/results/jevbench
  step $R-zeroshot $PY scripts/visionjev_eval.py $CK --kind zero_shot --datasets $ZS --tag zeroshot
  step $R-novel $PY scripts/visionjev_eval.py $CK --datasets novel_text,novel_vision --kind never_seen --tag novel \
    --no-shuffle-control
  step $R-hof $PY scripts/visionjev_eval.py $CK --kind heldout_formulation --datasets $HOF --tag hof
  step $R-robust $PY scripts/visionjev_robustness.py $CK --datasets $ROB --n 100 --tag robust
  step $R-latency $PY scripts/visionjev_latency.py $CK --datasets aokvqa,vqav2_yesno,doclaynet_type,synth_ui --n 100
done
step report $PY scripts/visionjev_report.py
