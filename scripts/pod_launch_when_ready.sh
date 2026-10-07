#!/usr/bin/env bash
# Local, detached (run inside tmux): wait until a pod has its environment (PRESETUP_DONE in presetup.log) and data
# (the data/vqa link), start a pod script there exactly once, then become the fetch watcher (pod_watch.sh).
# Nothing here depends on the chat session or on one ssh connection staying up: every ssh call is retried.
#
#   tmux new-session -d -s visionjev-pod "bash scripts/pod_launch_when_ready.sh <host> <port> <pod-script> <done-log-name> <tag> [ckpt ...]"
set -uo pipefail
HOST="$1"; PORT="$2"; SCRIPT="$3"; DONE="$4"; TAG="$5"; shift 5
WS=/workspace/VisionJev
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/home/kneepolean/shubhj/VisionJev/data/logs/pod/$TAG
mkdir -p "$OUT"
SSH="ssh -p $PORT -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes root@$HOST"
say() { echo "[$(date -u +%H:%M)] $*" | tee -a "$OUT/watch.log"; }
say "launcher: waiting for the pod's environment and data ($HOST:$PORT)"
until $SSH "grep -q PRESETUP_DONE $WS/data/logs/presetup.log && test -L $WS/data/vqa" 2>/dev/null; do sleep 20; done
say "pod ready: $($SSH "ls /root/vqa | wc -l" 2>/dev/null) dataset dirs"
# start once: a second launcher (or a re-run) sees the log and does not start it again
until $SSH "cd $WS/laya-vision && if [ -f $WS/data/logs/$DONE ]; then echo already-started; else (setsid nohup bash scripts/$SCRIPT > /dev/null 2>&1 < /dev/null &); sleep 3; echo started; fi" 2>/dev/null | tee -a "$OUT/watch.log" | grep -qE "started"; do sleep 20; done
say "launched scripts/$SCRIPT"
exec bash "$HERE/pod_watch.sh" "$HOST" "$PORT" "$WS/data/logs/$DONE" "$TAG" "$@"
