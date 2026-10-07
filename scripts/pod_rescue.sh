#!/usr/bin/env bash
# Cron watchdog (every 5 min and @reboot): a pod whose autopilot died (reboot, crash) must not sit idle and billing.
# For every data/logs/autopilot/<tag>/state with no live autopilot for that job, restart the autopilot in resume mode
# (it waits for the pod script, fetches, verifies and deletes as usual; past the hard cap it fetches and deletes at once).
#   */5 * * * * /home/kneepolean/shubhj/VisionJev/laya-vision/scripts/pod_rescue.sh
#   @reboot sleep 60 && /home/kneepolean/shubhj/VisionJev/laya-vision/scripts/pod_rescue.sh
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH
LOCAL=/home/kneepolean/shubhj/VisionJev
HERE=$LOCAL/laya-vision
for st in "$LOCAL"/data/logs/autopilot/*/state; do
  [ -f "$st" ] || continue
  tag=$(basename "$(dirname "$st")")
  job=$(grep -lE "^TAG=$tag\$" "$HERE"/jobs/*.env | head -1)
  [ -n "$job" ] || { echo "[$(date -u '+%F %H:%M')] rescue: no job file for $tag" >> "$LOCAL/data/logs/autopilot/rescue.log"; continue; }
  pgrep -f "pod_autopilot.sh $job" > /dev/null && continue
  pgrep -f "pod_autopilot.sh jobs/$(basename "$job")" > /dev/null && continue
  echo "[$(date -u '+%F %H:%M')] rescue: autopilot for $tag is not running, resuming" >> "$LOCAL/data/logs/autopilot/rescue.log"
  (cd "$HERE" && setsid nohup bash scripts/pod_autopilot.sh "$job" --resume > /dev/null 2>&1 < /dev/null &)
done
