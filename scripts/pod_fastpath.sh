#!/usr/bin/env bash
# Fast serving path (laya/qwen_fast.py) against the eager model on one H100: same decisions? how much faster?
# scripts/visionjev_fast_check.py on the demo samples (text + image) and all JevBench public tiers, then the stage profile
# with and without the fast path. Run through pod_autopilot.sh (jobs/fastpath.env). No training.
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
DONE=$LOGS/fastpath.log
CK=$WS/checkpoints/qwen4b-v6b4
mkdir -p "$LOGS"
rc=0
trap 'echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/fastpath-$name.log" 2>&1
  local r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false VISIONJEV_DATA=$WS/data/vqa
PY=/workspace/venv/bin/python
REV=$(python3 -c "import json; print(json.load(open('$CK/vlm_agent_config.json'))['backbone_revision'])")
BB=$(python3 -c "import json; print(json.load(open('$CK/vlm_agent_config.json'))['backbone'])")
step setup bash $WS/laya-vision/scripts/setup_runpod.sh || exit
step download /workspace/venv/bin/hf download "$BB" --revision "$REV" || exit
cd $WS/laya-vision
step check $PY scripts/visionjev_fast_check.py $CK --jevbench /workspace/jevbench --reps 3
grep -E '^(text|image|jevbench_|fast path|gpu)' $LOGS/fastpath-check.log | cut -c1-400 >> "$DONE"
step check-unmerged $PY scripts/visionjev_fast_check.py $CK --jevbench /workspace/jevbench --reps 3 --no-merge
grep -E '^(text|image|jevbench_)' $LOGS/fastpath-check-unmerged.log | cut -c1-400 >> "$DONE"
step profile-eager $PY scripts/visionjev_profile.py $CK
step profile-fast $PY scripts/visionjev_profile.py $CK --fast
grep -E '^(text|image)' $LOGS/fastpath-profile-*.log | cut -c1-400 >> "$DONE"
