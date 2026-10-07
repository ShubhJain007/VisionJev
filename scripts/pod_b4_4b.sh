#!/usr/bin/env bash
# Focused stage B4 on one H100 (derived from pod_b3_4b.sh): continue qwen4b-v6b3 on configs/mix_v6b4.json (30% teacher
# hard questions rounds 1+2 as text, Laya typed, hard reasoning sets, screens / games / object images, text replay), 3,400 steps or 215 min, length
# groups, health-checked; then 50/50 and 75/25 averages with the start adapter (Imajev ships the 50/50 one) and the
# evals of B4 and the 50/50 average. No notes step (think-if-unsure did not help). Run through pod_autopilot.sh.
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
DONE=$LOGS/b4.log
INIT=qwen4b-v6b3
B3=qwen4b-v6b4
B3_MINUTES=${B3_MINUTES:-215}
NOTES_PER_DATASET=${NOTES_PER_DATASET:-500}
mkdir -p "$LOGS"
rc=0
trap 'echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/b4-$name.log" 2>&1
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
  "$@" > "$LOGS/b4-$name.log" 2>&1 &
  local pid=$! t_first="" next=0 r
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    if [ -z "$t_first" ] && grep -q "^step " "$LOGS/b4-$name.log"; then t_first=$(date +%s); next=$((t_first + 300)); fi
    if [ -n "$t_first" ] && [ "$(date +%s)" -ge "$next" ]; then
      next=$(( $(date +%s) + 1200 ))
      $PY scripts/visionjev_healthcheck.py "$LOGS/b4-$name.log" --min-wait 15 > "$LOGS/b4-$name-health.log" 2>&1
      r=$?
      note "health ($name) rc=$r: $(grep '^verdict' "$LOGS/b4-$name-health.log" | tail -1 | cut -c1-230)"
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

TRAIN_NOTES=""  # think-if-unsure did not help (PROBLEMS E13): no notes step

COMMON="--batch-size 16 --dtype bf16 --option-attention causal --w-ce-schedule const --qwen-max-pixels 589824 \
  --max-len 4096 --n-calib 40 --max-val 150 --eval-per-dataset 30 --num-workers 16 --augment --out $WS/checkpoints"
TRAIN="--init-checkpoint $WS/checkpoints/$INIT --mix-config configs/mix_v6b4.json --freeze lora --lr-head 5e-5 \
  --lr-backbone 2e-5 --eval-every 1000 --rationale-weight 0.3 --rationale-max-tokens 192 --abstain 0.5 \
  $TRAIN_NOTES $COMMON"
FAST="--length-groups 4 --grad-checkpointing"  # what B3 ran for 3,500 steps (3.05 s/step, 27 GB); no replicas on one GPU
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
grep -E "^train stats|^step " $LOGS/b4-smoke-*.log | tail -3 | cut -c1-220 >> "$DONE"
$PY scripts/visionjev_healthcheck.py $LOGS/b4-smoke-$( [ "$SPEED" = "$FAST" ] && echo fast || echo slow ).log | tail -1 | cut -c1-230 >> "$DONE"
rm -rf $WS/checkpoints/$B3-smoke

# shellcheck disable=SC2086
train_watched b3 $PY scripts/visionjev_finetune.py --run-name $B3 $TRAIN $SPEED --minutes $B3_MINUTES --steps 3400
[ -f $WS/checkpoints/$B3/vlm_agent_config.json ] || { note "no B3 checkpoint"; exit; }
grep -E "^train stats" $LOGS/b4-b3.log | cut -c1-300 >> "$DONE"
CKPTS="$B3"
# shellcheck disable=SC2086
if step soup $PY scripts/visionjev_soup.py $WS/checkpoints/$INIT $WS/checkpoints/$B3 --w 0.5 0.75; then
  CKPTS="$CKPTS $B3-soup50"
fi
note "checkpoints to evaluate: $CKPTS"

ZS="openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts"
ROB="aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence"
HOF=$(ls $WS/data/vqa | grep '_hof$' | paste -sd, -)
EVAL_ONLY=$(for d in teacher_hard_eval teacher_hard_eval2 teacher_calib; do [ -f $WS/data/vqa/$d/val.jsonl ] && echo $d; done | paste -sd, -)
NEW=$(for d in game_tictactoe_move game_connect4_win game_maze_path game_minesweeper_safe game_2048_move game_snake_safe game_chess_attack oi_where oi_count gen_deadline_chain gen_period_count gen_fx_aggregate gen_lookup_chain gen_cond_probability gen_constraint_rank gen_amended_policy judge_step_check judge_step_pair judge_rubric_cap judge_adequacy finqa_choice tatqa_choice tabfact_noul timeqa_choice summeval_pair mqm_pair faithdial_noul agent_verify_pair agent_outcome agent_step_progress ac_next_action ac_element ac_goal_reached ui_refexp ui_widget_find cad_feature_count cad_view_match cad_dimension_read; do [ -f $WS/data/vqa/$d/val.jsonl ] && echo $d; done | paste -sd, -)
for R in $CKPTS; do
  CK=$WS/checkpoints/$R
  step $R-jevbench $PY scripts/visionjev_jevbench.py $CK --jevbench /workspace/jevbench --device cuda --max-len 4096 --out $WS/results/jevbench
  step $R-zeroshot $PY scripts/visionjev_eval.py $CK --kind zero_shot --datasets $ZS --tag zeroshot
  step $R-novel $PY scripts/visionjev_eval.py $CK --datasets novel_text,novel_vision --kind never_seen --tag novel --no-shuffle-control
  step $R-hof $PY scripts/visionjev_eval.py $CK --kind heldout_formulation --datasets $HOF --tag hof
  [ -n "$EVAL_ONLY" ] && step $R-teacher $PY scripts/visionjev_eval.py $CK --kind never_seen --datasets $EVAL_ONLY --tag teacher --no-shuffle-control
  [ -n "$NEW" ] && step $R-newsets $PY scripts/visionjev_eval.py $CK --kind eval --datasets $NEW --tag round3 --max-per-dataset 300 --no-shuffle-control
  step $R-robust $PY scripts/visionjev_robustness.py $CK --datasets $ROB --n 100 --tag robust
done
step $INIT-robust $PY scripts/visionjev_robustness.py $WS/checkpoints/$INIT --datasets $ROB --n 100 --tag robust  # missing for B3 (I14)
step report $PY scripts/visionjev_report.py
