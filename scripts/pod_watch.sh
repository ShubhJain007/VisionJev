#!/usr/bin/env bash
# Detached local watcher for a pod run (PROBLEMS I9: a watcher inside the chat session dies with it). Waits for the
# run's EXIT_CODE line, copies results, logs and the named checkpoints back, verifies them by sha256 and writes
# FETCH_OK / FETCH_FAILED to <local-log>. It cannot delete the pod (that needs the RunPod API): the pod id and the
# expected finish time are given to the user at launch.
#
#   setsid nohup bash scripts/pod_watch.sh <host> <port> <pod-done-log> <tag> [checkpoint-name ...] > /dev/null 2>&1 &
set -uo pipefail
HOST="$1"; PORT="$2"; LOG="$3"; TAG="$4"; shift 4
WS=/workspace/VisionJev
LOCAL=/home/kneepolean/shubhj/VisionJev
OUT=$LOCAL/data/logs/pod/$TAG
mkdir -p "$OUT" "$LOCAL/results/pod/$TAG"
SSH="ssh -p $PORT -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes root@$HOST"
say() { echo "[$(date -u +%H:%M)] $*" >> "$OUT/watch.log"; }
say "watching $HOST:$PORT $LOG"
until $SSH "grep -q EXIT_CODE $LOG" 2>/dev/null; do sleep 120; done
say "run finished: $($SSH "tail -1 $LOG" 2>/dev/null)"
pull() { rsync -a --no-owner --no-group --partial -e "ssh -p $PORT -o ServerAliveInterval=30" "$@" 2>&1 | grep -v setlocale; }
ok=1
check() {  # check <remote dir> <local dir>
  local r l
  r=$($SSH "cd $1 && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k2" 2>/dev/null)
  l=$(cd "$2" && find . -type f ! -name watch.log -exec sha256sum {} + | LC_ALL=C sort -k2)
  if [ -n "$r" ] && [ "$r" = "$l" ]; then say "OK $1 ($(echo "$l" | wc -l) files)"; else say "MISMATCH $1"; ok=0; fi
}
for attempt in 1 2 3; do
  ok=1
  pull "root@$HOST:$WS/results/" "$LOCAL/results/pod/$TAG/"
  mkdir -p "$OUT/pod-logs"; pull "root@$HOST:$WS/data/logs/" "$OUT/pod-logs/"
  check $WS/results "$LOCAL/results/pod/$TAG"
  check $WS/data/logs "$OUT/pod-logs"
  for R in "$@"; do
    mkdir -p "$LOCAL/checkpoints/$R"; pull "root@$HOST:$WS/checkpoints/$R/" "$LOCAL/checkpoints/$R/"
    check $WS/checkpoints/$R "$LOCAL/checkpoints/$R"
  done
  [ $ok -eq 1 ] && break
  sleep 60
done
[ $ok -eq 1 ] && say "FETCH_OK: everything is local and verified; the pod can be deleted" || say "FETCH_FAILED"
