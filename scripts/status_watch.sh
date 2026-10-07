#!/usr/bin/env bash
# Persistent status board, independent of any chat session or ssh login. Run under tmux:
#   tmux new-session -d -s vj-status "bash scripts/status_watch.sh"
# Every 2 minutes it rewrites data/logs/STATUS.md (and appends changes to data/logs/status-history.log): live pods and
# cost so far, each autopilot job's last lines, and for a job whose pod is alive the pod's stage log, health verdicts,
# latest training step / eval lines and GPU use. Read it any time:  cat ~/shubhj/VisionJev/data/logs/STATUS.md
set -uo pipefail
LOCAL=/home/kneepolean/shubhj/VisionJev
AP=$LOCAL/data/logs/autopilot
OUT=$LOCAL/data/logs/STATUS.md
HIST=$LOCAL/data/logs/status-history.log
last=""
while true; do
  {
    echo "# VisionJev status: $(date -u '+%Y-%m-%d %H:%M UTC')  (local $(date '+%H:%M'))"
    echo
    echo "## Live pods"
    runpodctl pod list 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: d=None
if d is None: print('(could not query RunPod)')
elif not d: print('none')
else:
    for p in d: print('- %s  %s  %s  \$%s/h  up %.1f h' % (p.get('id'), p.get('name'), p.get('runtimeStatus'), p.get('costPerHr'), (p.get('uptimeSeconds') or 0)/3600))"
    echo
    for d in "$AP"/*/; do
      [ -f "$d/autopilot.log" ] || continue
      tag=$(basename "$d")
      echo "## job $tag  $( [ -f "$d/pod.id" ] && echo "(pod $(cat "$d/pod.id") ALIVE)" || echo "(no pod)")"
      grep -E '^\[20' "$d/autopilot.log" | tail -6 | cut -c1-230 | sed 's/^/    /'
      if [ -f "$d/pod.id" ]; then
        hp=$(grep -oE 'ssh root@[0-9.]+ -p [0-9]+' "$d/autopilot.log" | tail -1)
        host=$(echo "$hp" | sed -E 's/ssh root@([0-9.]+) -p ([0-9]+)/\1/'); port=$(echo "$hp" | sed -E 's/.* -p ([0-9]+)/\1/')
        if [ -n "$host" ]; then
          echo "    --- on the pod ($host:$port):"
          timeout 40 ssh -p "$port" -o ConnectTimeout=15 -o BatchMode=yes "root@$host" '
            L=/workspace/VisionJev/data/logs
            for f in $L/b3.log $L/speedcheck.log $L/night4b.log; do [ -f $f ] && tail -8 $f | cut -c1-220; done
            t=$(ls -t $L/*-b3.log $L/*-smoke-*.log 2>/dev/null | head -1)
            [ -n "$t" ] && { echo "latest training log: $(basename $t)"; grep -E "^step |^\[eval step|^train stats|Traceback|OutOfMemory" $t | tail -4 | cut -c1-200; }
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr "\n" ";"' 2>/dev/null | grep -v setlocale | sed 's/^/    /' || echo "    (ssh to the pod failed)"
          echo
        fi
      fi
      [ -f "$LOCAL/results/pod/$tag/VERDICT.md" ] && { echo "    verdict:"; sed 's/^/    /' "$LOCAL/results/pod/$tag/VERDICT.md" | head -12; }
      echo
    done
  } > "$OUT.tmp" 2>&1
  mv "$OUT.tmp" "$OUT"
  cur=$(grep -vE "^# VisionJev status|up [0-9.]+ h|^    [0-9], " "$OUT" | md5sum)
  if [ "$cur" != "$last" ]; then { echo "===== $(date -u '+%Y-%m-%d %H:%M UTC')"; cat "$OUT"; } >> "$HIST"; last=$cur; fi
  sleep 120
done
