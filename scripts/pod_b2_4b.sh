#!/usr/bin/env bash
# Phase B2 of Qwen3-VL-4B (Hypothesis/V6.2.md section 2), hands-off on one pod:
#   setup -> backbone download -> smoke -> B2 (delta from qwen4b-v6b-soup75 on mix_v6b2 with the rationale loss and the
#   abstain + reason head) -> soups of soup75 and B2 -> the same eval suite as pod_eval.sh for B2 and every soup.
# Needs checkpoints/qwen4b-v6b-soup75 and the data under $WS/data/vqa (tar-streamed before launch).
# The last line of $DONE is always "EXIT_CODE=<rc>" (trap), for the local fetch watcher (PROBLEMS I5).
#
#   setsid nohup bash scripts/pod_b2_4b.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
DONE=$LOGS/b2.log
INIT=qwen4b-v6b-soup75
B2=qwen4b-v6b2
B2_MINUTES=${B2_MINUTES:-150}
SOUP_W=${SOUP_W:-"0.5 0.75"}
mkdir -p "$LOGS"
rc=0
trap 'echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {  # step <name> <command...>: own log per step; remember the first failure
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/b2-$name.log" 2>&1
  local r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}

export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export VISIONJEV_DATA=$WS/data/vqa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv/bin/python
[ -f $WS/checkpoints/$INIT/vlm_agent_config.json ] || { note "missing $INIT"; rc=2; exit; }
REV=$(python3 -c "import json; print(json.load(open('$WS/checkpoints/$INIT/vlm_agent_config.json'))['backbone_revision'])")
BB=$(python3 -c "import json; print(json.load(open('$WS/checkpoints/$INIT/vlm_agent_config.json'))['backbone'])")
COMMON="--batch-size 16 --dtype bf16 --option-attention causal --w-ce-schedule const --grad-checkpointing \
  --qwen-max-pixels 589824 --max-len 4096 --n-calib 40 --max-val 150 --eval-per-dataset 30 --num-workers 16 \
  --augment --out $WS/checkpoints"
B2_ARGS="--init-checkpoint $WS/checkpoints/$INIT --mix-config configs/mix_v6b2.json --freeze lora --lr-head 5e-5 \
  --lr-backbone 2e-5 --eval-every 400 --rationale-weight 0.3 --rationale-max-tokens 96 --abstain 0.5 $COMMON"

step setup bash $WS/laya-vision/scripts/setup_runpod.sh || exit
step download /workspace/venv/bin/hf download "$BB" --revision "$REV" || exit
cd $WS/laya-vision
# shellcheck disable=SC2086
step smoke $PY scripts/visionjev_finetune.py --run-name $B2-smoke $B2_ARGS --steps 40 --eval-every 1000 \
  --eval-per-dataset 4 --n-calib 4 --max-val 4 --no-keep-best || exit
grep "^step " $LOGS/b2-smoke.log | tail -2 >> "$DONE"
rm -rf $WS/checkpoints/$B2-smoke

# shellcheck disable=SC2086
step b2 $PY scripts/visionjev_finetune.py --run-name $B2 $B2_ARGS --minutes $B2_MINUTES
[ -f $WS/checkpoints/$B2/vlm_agent_config.json ] || { note "no B2 checkpoint"; exit; }
CKPTS="$B2"
# shellcheck disable=SC2086
if step soup $PY scripts/visionjev_soup.py $WS/checkpoints/$INIT $WS/checkpoints/$B2 --w $SOUP_W; then
  for w in $SOUP_W; do CKPTS="$CKPTS $B2-soup$(python3 -c "print(round(100*$w))")"; done
fi
note "checkpoints to evaluate: $CKPTS"

ZS="openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts"
ROB="aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence"
HOF=$(ls $WS/data/vqa | grep '_hof$' | paste -sd, -)
EVAL_ONLY=$(for d in teacher_hard_eval teacher_calib; do [ -f $WS/data/vqa/$d/val.jsonl ] && echo $d; done | paste -sd, -)
for R in $CKPTS; do
  CK=$WS/checkpoints/$R
  step $R-jevbench $PY scripts/visionjev_jevbench.py $CK --jevbench /workspace/jevbench --device cuda --max-len 4096 \
    --out $WS/results/jevbench
  step $R-zeroshot $PY scripts/visionjev_eval.py $CK --kind zero_shot --datasets $ZS --tag zeroshot
  step $R-novel $PY scripts/visionjev_eval.py $CK --datasets novel_text,novel_vision --kind never_seen --tag novel \
    --no-shuffle-control
  step $R-hof $PY scripts/visionjev_eval.py $CK --kind heldout_formulation --datasets $HOF --tag hof
  [ -n "$EVAL_ONLY" ] && step $R-teacher $PY scripts/visionjev_eval.py $CK --kind never_seen --datasets $EVAL_ONLY \
    --tag teacher --no-shuffle-control
  step $R-robust $PY scripts/visionjev_robustness.py $CK --datasets $ROB --n 100 --tag robust
  step $R-latency $PY scripts/visionjev_latency.py $CK --datasets aokvqa,vqav2_yesno,doclaynet_type,synth_ui --n 100
done
step report $PY scripts/visionjev_report.py
