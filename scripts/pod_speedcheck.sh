#!/usr/bin/env bash
# Verify the trainer speed options and think-if-unsure on a 2-GPU pod before B3 (Hypothesis/V6.3.md sections 4 and 7):
#   1. visionjev_speedcheck.py: grouped / multi-device forward equals the padded one on the real 4B; seconds per step,
#      padded tokens and peak memory for: B2 setting (padded + checkpointing), length groups, length groups + 2 GPUs.
#   2. visionjev_notes.py on two datasets x 16 rows: the cached generation self-check on the Qwen backbone.
#   3. JevBench public with --thinking auto on the B2 checkpoint (not trained for it: the zero-shot baseline of H14).
# Needs checkpoints/qwen4b-v6b2, data/vqa (the mix_v6b2 datasets), /workspace/jevbench. Last line of $DONE is
# always EXIT_CODE=<rc> (PROBLEMS I5).
#
#   setsid nohup bash scripts/pod_speedcheck.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
DONE=$LOGS/speedcheck.log
CK=$WS/checkpoints/qwen4b-v6b2
mkdir -p "$LOGS"
rc=0
trap 'echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/speedcheck-$name.log" 2>&1
  local r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export VISIONJEV_DATA=$WS/data/vqa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venv/bin/python
[ -f $CK/vlm_agent_config.json ] || { note "missing $CK"; rc=2; exit; }
REV=$(python3 -c "import json; print(json.load(open('$CK/vlm_agent_config.json'))['backbone_revision'])")
BB=$(python3 -c "import json; print(json.load(open('$CK/vlm_agent_config.json'))['backbone'])")
step setup bash $WS/laya-vision/scripts/setup_runpod.sh || exit
step download /workspace/venv/bin/hf download "$BB" --revision "$REV" || exit
cd $WS/laya-vision
NGPU=$(nvidia-smi -L | wc -l)
DEV=$(python3 -c "print(','.join('cuda:%d' % i for i in range($NGPU)) if $NGPU > 1 else '')")
note "gpus: $NGPU ($DEV)"
step speed $PY scripts/visionjev_speedcheck.py $CK --mix-config configs/mix_v6b2.json --steps ${STEPS:-40} \
  ${DEV:+--devices $DEV} --num-workers 8
grep -E "^exactness|s_per_step" $LOGS/speedcheck-speed.log | cut -c1-260 >> "$DONE"
step notes $PY scripts/visionjev_notes.py $CK --datasets gen_rules,gen_temporal2,aokvqa --per-dataset 16 --max-tokens 96 \
  --tag speedcheck --max-len 4096
grep -E "^selfcheck|notes, mean" $LOGS/speedcheck-notes.log | cut -c1-260 >> "$DONE"
step jevbench-think $PY scripts/visionjev_jevbench.py $CK --jevbench /workspace/jevbench --device cuda --max-len 4096 \
  --thinking auto --think-threshold 0.7 --think-max-tokens 96 --out $WS/results/jevbench --limit 60
