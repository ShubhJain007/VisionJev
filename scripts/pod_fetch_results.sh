#!/usr/bin/env bash
# Wait for a pod run to finish, then copy its results, logs and trained weights back and verify them, so the pod
# can be deleted. Base-model weights are NOT copied: they are public on Hugging Face (the run records the revision).
#
#   bash scripts/pod_fetch_results.sh <ssh-host> <ssh-port> <run-name> [log-to-wait-for]
#
# [log-to-wait-for] (default <run>.train.log on the pod) lets a follow-up stage (extra evals) finish first.
#
# Local layout: checkpoints/<run>/ (LoRA + heads, processor, config, metrics), results/pod/<run>/ (the pod's
# experiments.jsonl, evals, figures), data/logs/pod/ (training / eval logs). Prints FETCH_OK when every file is
# verified by sha256, FETCH_FAILED otherwise.
set -uo pipefail
HOST="$1"; PORT="$2"; RUN="$3"
WS=/workspace/VisionJev
LOCAL=/home/kneepolean/shubhj/VisionJev
SSH="ssh -p $PORT -o ConnectTimeout=20 -o ServerAliveInterval=30 root@$HOST"
LOG="${4:-$WS/data/logs/$RUN.train.log}"

until $SSH "grep -qE 'EXIT_CODE|Traceback|Killed|OutOfMemory' $LOG" 2>/dev/null; do sleep 300; done
$SSH "grep -E 'EXIT_CODE|Traceback|Error:|Killed|OutOfMemory|^saved|^logged' $LOG | tail -8 | cut -c1-250" 2>/dev/null

mkdir -p "$LOCAL/checkpoints/$RUN" "$LOCAL/results/pod/$RUN" "$LOCAL/data/logs/pod"
pull() { rsync -a --no-owner --no-group --partial -e "ssh -p $PORT -o ServerAliveInterval=30" "$@" 2>&1 | grep -v setlocale; }
pull "root@$HOST:$WS/checkpoints/$RUN/" "$LOCAL/checkpoints/$RUN/"
pull "root@$HOST:$WS/results/" "$LOCAL/results/pod/$RUN/"
pull "root@$HOST:$WS/data/logs/" "$LOCAL/data/logs/pod/"
# other runs' checkpoint metadata (metrics, configs) without their weights, e.g. the smoke run
pull --exclude '*.safetensors' "root@$HOST:$WS/checkpoints/" "$LOCAL/results/pod/$RUN/checkpoint-metadata/"

remote=$($SSH "cd $WS/checkpoints/$RUN && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k2" 2>/dev/null)
local_=$(cd "$LOCAL/checkpoints/$RUN" && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k2)
rres=$($SSH "cd $WS/results && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k2" 2>/dev/null)
lres=$(cd "$LOCAL/results/pod/$RUN" && find . -path ./checkpoint-metadata -prune -o -type f -exec sha256sum {} + | LC_ALL=C sort -k2)
if [ -n "$remote" ] && [ "$remote" = "$local_" ] && [ -n "$rres" ] && [ "$rres" = "$lres" ]; then
  echo "FETCH_OK $(echo "$local_" | wc -l) checkpoint files, $(echo "$lres" | wc -l) result files, $(du -sh "$LOCAL/checkpoints/$RUN" | cut -f1)"
else
  echo "FETCH_FAILED (checksums differ or remote unreadable)"
fi
