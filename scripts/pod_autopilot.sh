#!/usr/bin/env bash
# One pod run, start to finish, with no chat session and no single ssh connection needed (PROBLEMS I9 / I10):
#   price check -> create the pod (runpodctl, local API key) -> send code, validator, JevBench clone, checkpoints and
#   data -> set the environment up -> start the pod script detached -> wait for its EXIT_CODE (hard time cap) ->
#   fetch results, logs and named checkpoints, verify by sha256 -> merge the pod's experiments.jsonl -> DELETE THE POD
#   and confirm it is gone -> log a "pod" record. Any failure still fetches the logs and deletes the pod.
# Run it detached:   tmux new-session -d -s vj-<tag> "bash scripts/pod_autopilot.sh jobs/<tag>.env"
# The job file is a bash fragment, see jobs/*.env. Needs RUNPOD_API_KEY (or ~/.runpod/config.toml) and runpodctl.
# After a workstation reboot: `pod_autopilot.sh <job> --resume` picks the run up at the wait loop from $OUT/state
# (scripts/pod_rescue.sh, run from cron, does that for every stranded pod).
set -uo pipefail
JOB=$(readlink -f "$1")
RESUME=0; [ "${2:-}" = "--resume" ] && RESUME=1
HERE=$(cd "$(dirname "$0")" && pwd)
LOCAL=/home/kneepolean/shubhj/VisionJev
S_KEY=$(cat ~/.ssh/id_ed25519.pub)
# ---- job defaults, overridable by the job file ---------------------------------------------------------------
CLOUD=SECURE; MIN_CUDA=13.0; IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404; DISK=80; VOLUME=60
GPU_COUNT=1; CKPTS_SEND=""; CKPTS_FETCH=""; DATA_LIST=""; JEVBENCH=""; PRESETUP=1; POD_ENV=""; EVAL_RESERVE_MIN=0
MAX_HOURS=6; WAIT_TIMEOUT=20m; DELETE_ON_FAIL=1
# shellcheck disable=SC1090
source "$JOB"
: "${TAG:?}" "${GPU_ID:?}" "${POD_SCRIPT:?}" "${DONE_LOG:?}" "${POD_NAME:=visionjev-$TAG}"
OUT=$LOCAL/data/logs/autopilot/$TAG
mkdir -p "$OUT" "$LOCAL/results/pod/$TAG" "$LOCAL/data/logs/pod/$TAG"
LOG=$OUT/autopilot.log
say() { echo "[$(date -u "+%Y-%m-%d %H:%M")] $*" | tee -a "$LOG"; }
POD_ID=""; HOST=""; PORT=""; DELETED=0; RC=0; T_START=$(date +%s)
WS=/workspace/VisionJev

delete_pod() {  # delete and confirm; retried, never silently skipped
  [ -z "$POD_ID" ] || [ $DELETED -eq 1 ] && return 0
  for i in 1 2 3 4 5 6; do
    runpodctl pod delete "$POD_ID" >> "$LOG" 2>&1
    sleep 10
    st=$(runpodctl pod get "$POD_ID" 2>&1)
    if echo "$st" | grep -qE '"code": *"not_found"' || echo "$st" | grep -qiE '"(runtimeStatus|desiredStatus)": *"(terminated|TERMINATED|EXITED)"'; then
      DELETED=1; say "pod $POD_ID deleted and confirmed gone"; rm -f "$OUT/pod.id" "$OUT/state"; return 0
    fi
    say "delete attempt $i: pod still reported: $(echo "$st" | head -c 200)"; sleep $((30 * i))
  done
  say "COULD NOT CONFIRM DELETION of pod $POD_ID: delete it in the RunPod console"
  return 1
}
finish() {
  local rc=$?
  [ $RC -eq 0 ] && RC=$rc
  if [ -n "$POD_ID" ] && [ $DELETED -eq 0 ]; then
    if [ $RC -ne 0 ] && [ "$DELETE_ON_FAIL" != 1 ]; then say "failure and DELETE_ON_FAIL=0: pod $POD_ID LEFT RUNNING"; else delete_pod; fi
  fi
  hours=$(python3 -c "print(round(($(date +%s) - $T_START) / 3600, 2))")
  say "AUTOPILOT_DONE tag=$TAG rc=$RC pod=$POD_ID deleted=$DELETED hours=$hours est_cost=\$$(python3 -c "print(round($hours * ${PRICE:-0}, 2))")"
  cd "$LOCAL/laya-vision" && CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<PY >> "$LOG" 2>&1
from laya.visionjev_log import log_experiment
log_experiment("pod", "pod:$TAG", config={"job": open("$JOB").read(), "pod_id": "$POD_ID", "gpu": "$GPU_ID x $GPU_COUNT", "price_per_h": ${PRICE:-0}},
               metrics={"rc": $RC, "deleted": $DELETED, "hours": $hours, "est_cost_usd": round($hours * ${PRICE:-0}, 2)},
               notes="scripts/pod_autopilot.sh; log $LOG")
PY
}
trap finish EXIT
ssh_() { ssh -p "$PORT" -o ConnectTimeout=20 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -o BatchMode=yes "root@$HOST" "$@"; }
retry() { local n=$1; shift; for i in $(seq 1 "$n"); do "$@" && return 0; sleep $((15 * i)); done; return 1; }
pull() { rsync -a --no-owner --no-group --partial -e "ssh -p $PORT -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new" "$@" 2>&1 | grep -v setlocale; }

if [ $RESUME -eq 1 ]; then
  # shellcheck disable=SC1091
  source "$OUT/state" || { echo "no $OUT/state to resume from" >&2; trap - EXIT; exit 2; }
  say "RESUMED after an interruption: pod $POD_ID at $HOST:$PORT, started $(date -u -d @$T_START '+%H:%M UTC'), phase ${PHASE:-?}"
  if [ "${PHASE:-}" != launched ]; then say "interrupted before the pod script started: nothing to wait for, deleting"; RC=10; exit 10; fi
else
# ---- 1. price -----------------------------------------------------------------------------------------------
PRICE=$(runpodctl gpu list --include-unavailable 2>/dev/null | python3 -c "
import json,sys
for g in json.load(sys.stdin):
    if g.get('gpuId')=='$GPU_ID' or g.get('id')=='$GPU_ID' or g.get('displayName')=='$GPU_ID':
        p=g.get('securePricePerHr' if '$CLOUD'=='SECURE' else 'communityPricePerHr'); print((p or 0)*$GPU_COUNT); break
else: print(0)")
say "job $TAG: $GPU_COUNT x $GPU_ID ($CLOUD) at \$$PRICE/h, hard cap $MAX_HOURS h (max \$$(python3 -c "print(round($PRICE*$MAX_HOURS,2))"))"

# ---- 2. create ----------------------------------------------------------------------------------------------
ENVJ=$(python3 -c "import json; d={'PUBLIC_KEY': '''$S_KEY'''}; d.update(json.loads('''${POD_ENV:-{\}}''')); print(json.dumps(d))")
CREATE=$(runpodctl pod create --name "$POD_NAME" --image "$IMAGE" --gpu-id "$GPU_ID" --gpu-count "$GPU_COUNT" --cloud-type "$CLOUD" \
  --min-cuda-version "$MIN_CUDA" --container-disk-in-gb "$DISK" --volume-in-gb "$VOLUME" --ports "22/tcp" --env "$ENVJ" \
  $([ "$CLOUD" = COMMUNITY ] && echo --public-ip) --wait --wait-timeout "$WAIT_TIMEOUT" 2>&1)
echo "$CREATE" >> "$LOG"
POD_ID=$(echo "$CREATE" | python3 -c "
import json,sys,re
t=sys.stdin.read()
for line in t.splitlines():
    try: d=json.loads(line)
    except Exception: continue
    if isinstance(d,dict) and d.get('id'): print(d['id']); break
else:
    m=re.search(r'\"id\": *\"([a-z0-9]+)\"', t); print(m.group(1) if m else '')")
[ -n "$POD_ID" ] || { say "pod create failed: $(echo "$CREATE" | head -c 300)"; RC=2; exit 2; }
echo "$POD_ID" > "$OUT/pod.id"
printf 'POD_ID=%s\nT_START=%s\nPRICE=%s\nPHASE=created\n' "$POD_ID" "$T_START" "$PRICE" > "$OUT/state"
say "pod $POD_ID created (delete it in the console if this script dies: runpodctl pod delete $POD_ID)"
for i in $(seq 1 40); do
  INFO=$(runpodctl pod get "$POD_ID" 2>/dev/null)
  read -r HOST PORT <<< "$(echo "$INFO" | python3 -c "
import json,sys
d=json.load(sys.stdin)
s=d.get('ssh') or {}
if s.get('ip') and s.get('port'):  # runpodctl: {'ssh': {'ip': ..., 'port': ..., 'ssh_command': ...}} once the host mapped port 22
    print(s['ip'], s['port'])
else:
    for p in ((d.get('runtime') or {}).get('ports') or []):
        if p.get('privatePort')==22 and p.get('type')=='tcp': print(p.get('ip',''), p.get('publicPort','')); break
    else: print('', '')")"
  [ -n "$HOST" ] && [ -n "$PORT" ] && break
  sleep 15
done
[ -n "$HOST" ] || { say "no public ssh port after 10 min"; RC=3; exit 3; }
retry 20 ssh_ "echo ssh-ok" > /dev/null 2>&1 || { say "ssh never answered"; RC=3; exit 3; }
say "ssh root@$HOST -p $PORT"
printf 'HOST=%s\nPORT=%s\n' "$HOST" "$PORT" >> "$OUT/state"

# ---- 3. transfer ---------------------------------------------------------------------------------------------
ssh_ "mkdir -p $WS/laya-vision $WS/checkpoints $WS/data/logs $WS/results /workspace/jevbench" 2>&1 | grep -v setlocale
git -C "$LOCAL/laya-vision" archive --format=tar.gz HEAD | ssh_ "tar --no-same-owner -xzf - -C $WS/laya-vision" 2>&1 | grep -v setlocale
say "code: $(git -C "$LOCAL/laya-vision" rev-parse --short HEAD)"
scp -P "$PORT" -q -o StrictHostKeyChecking=accept-new "$LOCAL/data_validator.py" "root@$HOST:$WS/data_validator.py"
[ -n "$JEVBENCH" ] && tar -C "$(dirname "$JEVBENCH")" -czf - "$(basename "$JEVBENCH")" | ssh_ "tar --no-same-owner -xzf - -C /workspace && mv -T /workspace/$(basename "$JEVBENCH") /workspace/jevbench 2>/dev/null; true" 2>&1 | grep -v setlocale
for R in $CKPTS_SEND; do
  pull "$LOCAL/checkpoints/$R/" "root@$HOST:$WS/checkpoints/$R/"
  a=$(cd "$LOCAL/checkpoints/$R" && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k2 | sha256sum | cut -c1-16)
  b=$(ssh_ "cd $WS/checkpoints/$R && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k2 | sha256sum | cut -c1-16" 2>/dev/null)
  [ "$a" = "$b" ] && say "checkpoint $R sent and verified" || { say "checkpoint $R MISMATCH"; RC=4; exit 4; }
done
if [ -n "$DATA_LIST" ]; then
  n=$(wc -l < "$DATA_LIST")
  tar -C "$LOCAL/data" -cf - -T "$DATA_LIST" | ssh_ "mkdir -p $WS/data && tar --no-same-owner -xf - -C /root && ln -sfn /root/vqa $WS/data/vqa" 2>&1 | grep -v setlocale
  say "data: $n dirs, $(ssh_ "du -sh /root/vqa | cut -f1" 2>/dev/null) on the container disk"
fi

# ---- 4. environment (detached on the pod) --------------------------------------------------------------------
if [ "$PRESETUP" = 1 ]; then
  ssh_ "cd $WS/laya-vision && (setsid nohup bash -c 'export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_PROGRESS_BARS=1; bash scripts/setup_runpod.sh > $WS/data/logs/presetup.log 2>&1 && echo PRESETUP_DONE >> $WS/data/logs/presetup.log || echo PRESETUP_FAILED >> $WS/data/logs/presetup.log' > /dev/null 2>&1 < /dev/null &)" 2>&1 | grep -v setlocale
  until ssh_ "grep -qE 'PRESETUP_(DONE|FAILED)' $WS/data/logs/presetup.log" 2>/dev/null; do sleep 20; done
  ssh_ "grep -q PRESETUP_DONE $WS/data/logs/presetup.log" 2>/dev/null || { say "environment setup failed"; RC=5; exit 5; }
  say "environment ready"
fi

# ---- 5. run --------------------------------------------------------------------------------------------------
# The trainer ends its loop by this time (scripts/visionjev_finetune.py reads the file), so the checkpoint and
# EVAL_RESERVE_MIN minutes of evals fit under the hard cap instead of being cut by it.
if [ "${EVAL_RESERVE_MIN:-0}" -gt 0 ]; then
  DL=$(python3 -c "print(int($T_START + $MAX_HOURS * 3600 - (${EVAL_RESERVE_MIN} + 25) * 60))")
  ssh_ "echo $DL > $WS/data/logs/train_deadline" 2>&1 | grep -v setlocale
  say "training deadline: $(date -u -d @$DL '+%H:%M UTC') (cap minus $EVAL_RESERVE_MIN min of evals and 25 min to save)"
fi
ssh_ "cd $WS/laya-vision && (setsid nohup bash scripts/$POD_SCRIPT > /dev/null 2>&1 < /dev/null &)" 2>&1 | grep -v setlocale
say "launched scripts/$POD_SCRIPT; waiting for EXIT_CODE in $DONE_LOG (cap $MAX_HOURS h)"
echo "PHASE=launched" >> "$OUT/state"
fi  # not a resume
DEADLINE=$(( T_START + MAX_HOURS * 3600 ))
last=""
until ssh_ "grep -q EXIT_CODE= $WS/data/logs/$DONE_LOG" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then say "HARD CAP reached: fetching what exists and deleting"; RC=6; break; fi
  cur=$(ssh_ "tail -1 $WS/data/logs/$DONE_LOG 2>/dev/null" 2>/dev/null); [ "$cur" != "$last" ] && [ -n "$cur" ] && { say "pod: $cur"; last=$cur; }
  sleep 120
done
say "pod log tail: $(ssh_ "tail -3 $WS/data/logs/$DONE_LOG" 2>/dev/null | tr '\n' ' | ')"
ssh_ "grep -q 'EXIT_CODE=0' $WS/data/logs/$DONE_LOG" 2>/dev/null || { [ $RC -eq 0 ] && RC=7; }

# ---- 6. fetch + verify ---------------------------------------------------------------------------------------
check() { local r l; r=$(ssh_ "cd $1 && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k2" 2>/dev/null)
  l=$(cd "$2" && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k2)
  [ -n "$r" ] && [ "$r" = "$l" ] && { say "verified $1 ($(echo "$l" | wc -l) files)"; return 0; }; say "MISMATCH $1"; return 1; }
ok=0
for attempt in 1 2 3; do
  ok=1
  pull "root@$HOST:$WS/results/" "$LOCAL/results/pod/$TAG/"; check $WS/results "$LOCAL/results/pod/$TAG" || ok=0
  pull "root@$HOST:$WS/data/logs/" "$LOCAL/data/logs/pod/$TAG/"; check $WS/data/logs "$LOCAL/data/logs/pod/$TAG" || ok=0
  for R in $CKPTS_FETCH; do
    ssh_ "test -f $WS/checkpoints/$R/vlm_agent_config.json" 2>/dev/null || { say "checkpoint $R not on the pod"; continue; }
    mkdir -p "$LOCAL/checkpoints/$R"; pull "root@$HOST:$WS/checkpoints/$R/" "$LOCAL/checkpoints/$R/"; check $WS/checkpoints/$R "$LOCAL/checkpoints/$R" || ok=0
  done
  [ $ok -eq 1 ] && break; sleep 60
done
[ $ok -eq 1 ] && say "FETCH_OK" || { say "FETCH_FAILED after 3 attempts"; [ $RC -eq 0 ] && RC=8; }
if [ -f "$LOCAL/results/pod/$TAG/experiments.jsonl" ]; then
  python3 - <<PY >> "$LOG"
import json
loc='$LOCAL/results/experiments.jsonl'; have={json.loads(l)['id'] for l in open(loc)}
new=[l for l in open('$LOCAL/results/pod/$TAG/experiments.jsonl') if json.loads(l)['id'] not in have]
open(loc,'a').write(''.join(l if l.endswith('\n') else l+'\n' for l in new)); print('merged', len(new), 'pod records')
PY
fi
if [ -n "${VERDICT_NEW:-}" ] && [ -n "${VERDICT_BASE:-}" ]; then  # autonomous keep / discard, written next to the results
  # shellcheck disable=SC2086
  (cd "$LOCAL/laya-vision" && CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/visionjev_verdict.py --base "$VERDICT_BASE" --new $VERDICT_NEW \
    --out "$LOCAL/results/pod/$TAG") >> "$LOG" 2>&1 && say "verdict: KEEP (see results/pod/$TAG/VERDICT.md)" || say "verdict: nothing kept (see results/pod/$TAG/VERDICT.md)"
fi
# ---- 7. delete (also in the EXIT trap) -------------------------------------------------------------------------
delete_pod || RC=9
exit $RC
