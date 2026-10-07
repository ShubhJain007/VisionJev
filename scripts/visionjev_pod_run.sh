#!/usr/bin/env bash
# Launch one VisionJev experiment on a Runpod pod, detached (survives SSH disconnect), logging to the volume.
#
#   bash visionjev_pod_run.sh <run-name> <smoke|train|eval> [extra finetune args...]
#
#   smoke  60 steps, tiny val: checks memory and step time before paying for a full run
#   train  the full mix_v1 run, then the zero-shot + held-out-formulation evals and the report
#   eval   evals only, for an existing checkpoint
# Every stage logs itself to /workspace/VisionJev/results/experiments.jsonl.
set -euo pipefail
RUN="$1"; MODE="$2"; shift 2
WS=/workspace/VisionJev
PY=/workspace/venv/bin/python
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export VISIONJEV_DATA="$WS/data/vqa" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$WS/data/logs/$RUN.$MODE.log"
ZS="openimages_presence,openimages_count,mind2web_element,mind2web_operation,mmlu,openimages_where,openimages_relation,openimages_framing,robot_task_zs,robot_progress_zs,robospatial_home,humaneval_exec,cruxeval_output,probe_state_facts"
ROB="aokvqa,vqav2_yesno,doclaynet_type,doclaynet_layout,laya_typed,laya_typed_img,synth_ui,banking77,boolq,code_contests,oi_presence"
HOF=$(ls "$WS/data/vqa" | grep '_hof$' | paste -sd, -)

case "$MODE" in
  smoke) CMD="$PY scripts/visionjev_finetune.py --run-name $RUN-smoke --mix-config configs/mix_v1.json --steps 60 \
              --max-val 32 --n-calib 32 --eval-every 1000 --no-keep-best --out $WS/checkpoints $*" ;;
  # every stage runs even if an earlier eval fails (training failing makes the later ones fail fast)
  train) CMD="$PY scripts/visionjev_finetune.py --run-name $RUN --out $WS/checkpoints $* ; T=\$? ; \
              $PY scripts/visionjev_eval.py $WS/checkpoints/$RUN --kind zero_shot --datasets $ZS --tag zeroshot ; \
              $PY scripts/visionjev_eval.py $WS/checkpoints/$RUN --kind heldout_formulation --datasets $HOF --tag hof ; \
              $PY scripts/visionjev_robustness.py $WS/checkpoints/$RUN --datasets $ROB --n 100 --tag robust ; \
              $PY scripts/visionjev_latency.py $WS/checkpoints/$RUN --datasets aokvqa,vqav2_yesno,doclaynet_type,synth_ui --n 100 ; \
              $PY scripts/visionjev_report.py ; exit \$T" ;;
  eval)  CMD="$PY scripts/visionjev_eval.py $WS/checkpoints/$RUN --kind zero_shot --datasets $ZS --tag zeroshot && \
              $PY scripts/visionjev_eval.py $WS/checkpoints/$RUN --kind heldout_formulation --datasets $HOF --tag hof" ;;
  *) echo "mode must be smoke|train|eval"; exit 2 ;;
esac
cd "$WS/laya-vision"
setsid bash -c "$CMD; echo EXIT_CODE=\$?" > "$LOG" 2>&1 < /dev/null &
echo "LAUNCHED $RUN $MODE -> $LOG"
