#!/usr/bin/env bash
# Eval-only pod (1 GPU): what the B3 pod's hard cap cut off. Most valuable first: the two B2/B3 averages on the four
# verdict metrics, then the round-3 sets, robustness and same-GPU latency. No training. Run through pod_autopilot.sh
# (jobs/eval_soups.env); results merge into results/experiments.jsonl and the verdict is written locally.
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
DONE=$LOGS/evalsoups.log
mkdir -p "$LOGS"
rc=0
trap 'echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/evalsoups-$name.log" 2>&1
  local r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export VISIONJEV_DATA=$WS/data/vqa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv/bin/python
B3=qwen4b-v6b3
SOUPS="$B3-soup50 $B3-soup75"
for R in $SOUPS; do [ -f $WS/checkpoints/$R/vlm_agent_config.json ] || { note "missing $R"; rc=2; exit; }; done
REV=$(python3 -c "import json; print(json.load(open('$WS/checkpoints/$B3-soup50/vlm_agent_config.json'))['backbone_revision'])")
BB=$(python3 -c "import json; print(json.load(open('$WS/checkpoints/$B3-soup50/vlm_agent_config.json'))['backbone'])")
step setup bash $WS/laya-vision/scripts/setup_runpod.sh || exit
step download /workspace/venv/bin/hf download "$BB" --revision "$REV" || exit
cd $WS/laya-vision

ZS="openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts"
ROB="aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence"
HOF=$(ls $WS/data/vqa | grep '_hof$' | paste -sd, -)
NEW=$(for d in gen_deadline_chain gen_period_count gen_fx_aggregate gen_lookup_chain gen_cond_probability gen_constraint_rank gen_amended_policy judge_step_check judge_step_pair judge_rubric_cap judge_adequacy finqa_choice tatqa_choice tabfact_noul timeqa_choice summeval_pair mqm_pair faithdial_noul agent_verify_pair agent_outcome agent_step_progress ac_next_action ac_element ac_goal_reached ui_refexp ui_widget_find cad_feature_count cad_view_match cad_dimension_read; do [ -f $WS/data/vqa/$d/val.jsonl ] && echo $d; done | paste -sd, -)
for R in $SOUPS; do  # 1. the verdict metrics of both averages
  CK=$WS/checkpoints/$R
  step $R-jevbench $PY scripts/visionjev_jevbench.py $CK --jevbench /workspace/jevbench --device cuda --max-len 4096 --out $WS/results/jevbench
  step $R-novel $PY scripts/visionjev_eval.py $CK --datasets novel_text,novel_vision --kind never_seen --tag novel --no-shuffle-control
  step $R-zeroshot $PY scripts/visionjev_eval.py $CK --kind zero_shot --datasets $ZS --tag zeroshot
  step $R-hof $PY scripts/visionjev_eval.py $CK --kind heldout_formulation --datasets $HOF --tag hof
done
for R in $B3 $SOUPS qwen4b-v6b2; do  # 2. round-3 sets (B2 as the reference) and the teacher sets
  CK=$WS/checkpoints/$R
  [ -f $CK/vlm_agent_config.json ] || continue
  step $R-newsets $PY scripts/visionjev_eval.py $CK --kind eval --datasets $NEW --tag round3 --max-per-dataset 300 --no-shuffle-control
  [ $R = $B3 ] || [ $R = qwen4b-v6b2 ] || step $R-teacher $PY scripts/visionjev_eval.py $CK --kind never_seen --datasets teacher_hard_eval,teacher_calib --tag teacher --no-shuffle-control
done
LAT=""  # 3. same-GPU latency (the B2 pod's 286 ms vs round 1's 96 ms is unexplained)
for R in $B3 qwen4b-v6b2; do [ -f $WS/checkpoints/$R/vlm_agent_config.json ] && LAT="$LAT $WS/checkpoints/$R"; done
# shellcheck disable=SC2086
step latency $PY scripts/visionjev_latency.py $LAT --datasets aokvqa,vqav2_yesno,doclaynet_type,synth_ui --n 100
for R in $B3 $SOUPS; do  # 4. robustness
  step $R-robust $PY scripts/visionjev_robustness.py $WS/checkpoints/$R --datasets $ROB --n 100 --tag robust
done
step report $PY scripts/visionjev_report.py
