#!/usr/bin/env bash
# launch_when_validated.sh <mix config> <job env> [extra eval sets]: validate the mix on the workstation's CPUs (writes
# the marker the pod trusts), build the pod's data list, then start the autopilot. Detached; no chat session needed:
#   setsid nohup bash scripts/launch_when_validated.sh configs/mix_v6b4.json jobs/b4.env teacher_hard_eval2 &
# A failed validation launches nothing. After the launch, cron's pod_rescue.sh covers a reboot.
set -uo pipefail
cd "$(dirname "$0")/.."
MIX=$1; JOB=$2; EXTRA=${3:-}
TAG=$(basename "$MIX" .json)
LOG=../data/logs/launch-$TAG.log
say() { echo "[$(date -u '+%F %H:%M')] $*" >> "$LOG"; }
say "validating $MIX"
CUDA_VISIBLE_DEVICES="" env -u PYTHONPATH .venv/bin/python scripts/visionjev_finetune.py --mix-config "$MIX" --validate-only \
  > ../data/logs/validate-$TAG.log 2>&1 || { say "VALIDATION FAILED (see data/logs/validate-$TAG.log): nothing launched"; exit 1; }
say "validation passed: $(tail -1 ../data/logs/validate-$TAG.log | cut -c1-160)"
CUDA_VISIBLE_DEVICES="" env -u PYTHONPATH .venv/bin/python scripts/mix_data_dirs.py "$MIX" --out "jobs/data_$TAG.txt" \
  ${EXTRA:+--extra "$EXTRA"} >> "$LOG" 2>&1 || { say "data list failed: nothing launched"; exit 1; }
grep -q "validated-" "jobs/data_$TAG.txt" || { say "the validation marker is not in the data list: nothing launched"; exit 1; }
say "data list: $(wc -l < "jobs/data_$TAG.txt") entries; starting the autopilot for $JOB"
exec bash scripts/pod_autopilot.sh "$JOB"
