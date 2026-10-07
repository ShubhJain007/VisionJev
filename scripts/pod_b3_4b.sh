#!/usr/bin/env bash
# Phase B3 of Qwen3-VL-4B (Hypothesis/V6.3.md section 7), hands-off on one pod (launched by pod_autopilot.sh):
#   setup -> backbone -> student notes for the mix (think-if-unsure training rows) -> smoke with the speed options,
#   falling back to the B2 setting if it fails -> B3 from qwen4b-v6b2 -> soups -> evals of B3 and the 0.75 soup
#   (JevBench plain and think-if-unsure, zero-shot, never-seen, held-out wording, teacher eval sets, the new
#   round-3 val sets, robustness) -> latency of B3, qwen4b-v6b2 and round-1 soup75 on the same GPU -> report.
# Last line of $DONE is always EXIT_CODE=<rc> (trap).
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
DONE=$LOGS/b3.log
INIT=qwen4b-v6b2
B3=qwen4b-v6b3
B3_MINUTES=${B3_MINUTES:-240}
NOTES_PER_DATASET=${NOTES_PER_DATASET:-500}
mkdir -p "$LOGS"
rc=0
trap 'echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/b3-$name.log" 2>&1
  local r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}
train_watched() {  # train_watched <name> <command...>: like step, but health-checked (autoresearch rule: decide early)
  # 5 minutes after the first logged step, then every 20 minutes, scripts/visionjev_healthcheck.py reads the log; an
  # unhealthy verdict (diverging or non-finite loss, spikes, loader stalls, a bad baseline eval, a crash) kills THIS
  # run's process only, so a broken run costs minutes, not the whole budget. Verdicts go to $DONE.
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/b3-$name.log" 2>&1 &
  local pid=$! t_first="" next=0 r
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    if [ -z "$t_first" ] && grep -q "^step " "$LOGS/b3-$name.log"; then t_first=$(date +%s); next=$((t_first + 300)); fi
    if [ -n "$t_first" ] && [ "$(date +%s)" -ge "$next" ]; then
      next=$(( $(date +%s) + 1200 ))
      $PY scripts/visionjev_healthcheck.py "$LOGS/b3-$name.log" --min-wait 15 > "$LOGS/b3-$name-health.log" 2>&1
      r=$?
      note "health ($name) rc=$r: $(grep '^verdict' "$LOGS/b3-$name-health.log" | tail -1 | cut -c1-230)"
      if [ $r -eq 2 ]; then note "HEALTH CHECK FAILED: stopping $name"; kill $pid; sleep 5; kill -9 $pid 2>/dev/null; fi
    fi
  done
  wait $pid; r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export VISIONJEV_DATA=$WS/data/vqa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv/bin/python
[ -f $WS/checkpoints/$INIT/vlm_agent_config.json ] || { note "missing $INIT"; rc=2; exit; }
REV=$(python3 -c "import json; print(json.load(open('$WS/checkpoints/$INIT/vlm_agent_config.json'))['backbone_revision'])")
BB=$(python3 -c "import json; print(json.load(open('$WS/checkpoints/$INIT/vlm_agent_config.json'))['backbone'])")
step setup bash $WS/laya-vision/scripts/setup_runpod.sh || exit
step download /workspace/venv/bin/hf download "$BB" --revision "$REV" || exit
cd $WS/laya-vision
NGPU=$(nvidia-smi -L | wc -l)
DEV=$(python3 -c "print(','.join('cuda:%d' % i for i in range($NGPU)) if $NGPU > 1 else '')")
note "gpus: $NGPU"

# student notes for think-if-unsure rows (the model's own working, right or wrong; never the gold rationale)
# one process per GPU, each taking every NGPU-th dataset (the single-GPU pass took 2.5 h for 118 datasets)
notes_all() {
  local i r=0
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i $PY scripts/visionjev_notes.py $WS/checkpoints/$INIT --mix-config configs/mix_v6b3.json \
      --per-dataset $NOTES_PER_DATASET --max-tokens 96 --tag b3 --max-len 4096 --batch-tokens 60000 --shard $i/$NGPU \
      > $LOGS/b3-notes-$i.log 2>&1 &
  done
  for i in $(seq 0 $((NGPU - 1))); do wait -n || r=1; done
  cat $LOGS/b3-notes-*.log
  return $r
}
step notes notes_all
grep -E "^selfcheck" $LOGS/b3-notes.log | cut -c1-200 >> "$DONE"
if grep -q "MISMATCH" $LOGS/b3-notes.log || ! grep -q "^selfcheck.*OK" $LOGS/b3-notes.log; then
  note "notes self-check did not pass: training without notes"
  TRAIN_NOTES=""
else
  TRAIN_NOTES="--notes-tag b3 --notes-prob 0.5"
fi

COMMON="--batch-size 16 --dtype bf16 --option-attention causal --w-ce-schedule const --qwen-max-pixels 589824 \
  --max-len 4096 --n-calib 40 --max-val 150 --eval-per-dataset 30 --num-workers 16 --augment --out $WS/checkpoints"
TRAIN="--init-checkpoint $WS/checkpoints/$INIT --mix-config configs/mix_v6b3.json --freeze lora --lr-head 5e-5 \
  --lr-backbone 2e-5 --eval-every 500 --rationale-weight 0.3 --rationale-max-tokens 192 --abstain 0.5 \
  $TRAIN_NOTES $COMMON"
FAST="--length-groups 4 --grad-checkpointing-above 12000 ${DEV:+--devices $DEV}"
SLOW="--grad-checkpointing"
SPEED="$FAST"
# shellcheck disable=SC2086
if ! step smoke-fast $PY scripts/visionjev_finetune.py --run-name $B3-smoke $TRAIN $FAST --steps 40 --eval-every 1000 \
    --eval-per-dataset 4 --n-calib 4 --max-val 4 --no-keep-best; then
  note "fast smoke failed (see b3-smoke-fast.log): falling back to the B2 setting"
  SPEED="$SLOW"
  # shellcheck disable=SC2086
  step smoke-slow $PY scripts/visionjev_finetune.py --run-name $B3-smoke $TRAIN $SLOW --steps 40 --eval-every 1000 \
    --eval-per-dataset 4 --n-calib 4 --max-val 4 --no-keep-best || exit
fi
grep -E "^train stats|^step " $LOGS/b3-smoke-*.log | tail -3 | cut -c1-220 >> "$DONE"
$PY scripts/visionjev_healthcheck.py $LOGS/b3-smoke-$( [ "$SPEED" = "$FAST" ] && echo fast || echo slow ).log | tail -1 | cut -c1-230 >> "$DONE"
rm -rf $WS/checkpoints/$B3-smoke

# shellcheck disable=SC2086
train_watched b3 $PY scripts/visionjev_finetune.py --run-name $B3 $TRAIN $SPEED --minutes $B3_MINUTES
[ -f $WS/checkpoints/$B3/vlm_agent_config.json ] || { note "no B3 checkpoint"; exit; }
grep -E "^train stats" $LOGS/b3-b3.log | cut -c1-300 >> "$DONE"
CKPTS="$B3"
# shellcheck disable=SC2086
if step soup $PY scripts/visionjev_soup.py $WS/checkpoints/$INIT $WS/checkpoints/$B3 --w 0.5 0.75; then
  CKPTS="$CKPTS $B3-soup75"
fi
note "checkpoints to evaluate: $CKPTS"

ZS="openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts"
ROB="aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence"
HOF=$(ls $WS/data/vqa | grep '_hof$' | paste -sd, -)
EVAL_ONLY=$(for d in teacher_hard_eval teacher_calib; do [ -f $WS/data/vqa/$d/val.jsonl ] && echo $d; done | paste -sd, -)
NEW=$(for d in gen_deadline_chain gen_period_count gen_fx_aggregate gen_lookup_chain gen_cond_probability gen_constraint_rank gen_amended_policy judge_step_check judge_step_pair judge_rubric_cap judge_adequacy finqa_choice tatqa_choice tabfact_noul timeqa_choice summeval_pair mqm_pair faithdial_noul agent_verify_pair agent_outcome agent_step_progress ac_next_action ac_element ac_goal_reached ui_refexp ui_widget_find cad_feature_count cad_view_match cad_dimension_read; do [ -f $WS/data/vqa/$d/val.jsonl ] && echo $d; done | paste -sd, -)
for R in $CKPTS; do
  CK=$WS/checkpoints/$R
  step $R-jevbench $PY scripts/visionjev_jevbench.py $CK --jevbench /workspace/jevbench --device cuda --max-len 4096 --out $WS/results/jevbench
  step $R-jevbench-think $PY scripts/visionjev_jevbench.py $CK --jevbench /workspace/jevbench --device cuda --max-len 4096 \
    --thinking auto --think-threshold 0.7 --think-max-tokens 160 --out $WS/results/jevbench
  step $R-zeroshot $PY scripts/visionjev_eval.py $CK --kind zero_shot --datasets $ZS --tag zeroshot
  step $R-novel $PY scripts/visionjev_eval.py $CK --datasets novel_text,novel_vision --kind never_seen --tag novel --no-shuffle-control
  step $R-hof $PY scripts/visionjev_eval.py $CK --kind heldout_formulation --datasets $HOF --tag hof
  [ -n "$EVAL_ONLY" ] && step $R-teacher $PY scripts/visionjev_eval.py $CK --kind never_seen --datasets $EVAL_ONLY --tag teacher --no-shuffle-control
  [ -n "$NEW" ] && step $R-newsets $PY scripts/visionjev_eval.py $CK --kind eval --datasets $NEW --tag round3 --max-per-dataset 300 --no-shuffle-control
  step $R-robust $PY scripts/visionjev_robustness.py $CK --datasets $ROB --n 100 --tag robust
done
# same-GPU latency: B3 vs the two earlier checkpoints (the B2 pod's 286 ms vs round 1's 96 ms is unexplained)
LAT="$WS/checkpoints/$B3"
for R in $INIT qwen4b-v6b-soup75; do [ -f $WS/checkpoints/$R/vlm_agent_config.json ] && LAT="$LAT $WS/checkpoints/$R"; done
# shellcheck disable=SC2086
step latency $PY scripts/visionjev_latency.py $LAT --datasets aokvqa,vqav2_yesno,doclaynet_type,synth_ui --n 100
step report $PY scripts/visionjev_report.py
