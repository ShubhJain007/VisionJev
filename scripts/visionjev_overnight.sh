#!/usr/bin/env bash
# VisionJev overnight queue: runs the experiments below one after another on the local GPU, unattended.
#
#   nohup laya-vision/scripts/visionjev_overnight.sh > data/logs/overnight/queue.log 2>&1 &
#
# Safe to re-run: a run whose checkpoint already has metrics.json is skipped (results are create-only). A failed run
# is logged in the status file and the queue moves on; an out-of-memory failure (GPU, or the OS killing the process
# for RAM) is retried once at batch 4 with 2 loader workers.
# At the end: latency of every checkpoint (bf16, real val pages) and a Markdown summary.
#   status:  results/overnight/status.tsv       per-run logs: data/logs/overnight/<run>.log
#   report:  results/overnight/summary.md       latency:      results/overnight/latency-*.json
set -u
WS="$(cd "$(dirname "$0")/../.." && pwd)"
REPO="$WS/laya-vision"
PY="$REPO/.venv/bin/python"
LOGS="$WS/data/logs/overnight"
RES="$WS/results/overnight"
CKPT="$WS/checkpoints"
mkdir -p "$LOGS" "$RES" "$CKPT"
STATUS="$RES/status.tsv"
[ -f "$STATUS" ] || printf "time\trun\tstatus\tminutes\n" > "$STATUS"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH  # ROS humble on PYTHONPATH breaks the venv

DOC="doclaynet_type,doclaynet_layout,doclaynet_read_small,doclaynet_read_large"
VQA="aokvqa,vqav2_yesno"
COMMON="--eval-every 500 --n-calib 200 --num-workers 4 --train-act"  # every run trains Laya's act/escalate head
TEXT="laya_typed"
# Document arms: batch 8 for all of them (split 1536 does not fit 16 GB at 16) and one pass over the data, so the arms
# differ only in tiling and the queue fits a night (split 2048 runs ~2.3 s/step on the 4070 Ti).
DOCARGS="--batch-size 8 --max-passes 1"

log() { echo "[$(date '+%F %T')] $*"; }

run() {  # run <name> <minutes cap> <args...>
  local name="$1" minutes="$2"; shift 2
  if [ -f "$CKPT/$name/metrics.json" ]; then log "skip $name (done)"; return 0; fi
  log "start $name: $*"
  local t0=$SECONDS
  (cd "$REPO" && "$PY" scripts/visionjev_finetune.py --run-name "$name" --minutes "$minutes" $COMMON "$@") \
    > "$LOGS/$name.log" 2>&1
  local rc=$? mins=$(( (SECONDS - t0) / 60 ))
  if [ $rc -eq 0 ] && [ -f "$CKPT/$name/metrics.json" ]; then
    printf "%s\t%s\tok\t%s\n" "$(date '+%F %T')" "$name" "$mins" >> "$STATUS"; log "done $name (${mins} min)"
    return 0
  fi
  if { [ $rc -eq 137 ] || grep -q "out of memory\|OutOfMemoryError" "$LOGS/$name.log"; } && [[ "$name" != *-b4 ]]; then
    printf "%s\t%s\toom-retry-b4\t%s\n" "$(date '+%F %T')" "$name" "$mins" >> "$STATUS"; log "OOM in $name; retrying at batch 4"
    rm -rf "$CKPT/$name"
    run "$name-b4" "$minutes" "$@" --batch-size 4 --num-workers 2
    return $?
  fi
  printf "%s\t%s\tfailed(rc=%s)\t%s\n" "$(date '+%F %T')" "$name" "$rc" "$mins" >> "$STATUS"; log "FAILED $name (rc=$rc), see $LOGS/$name.log"
  return 1
}

log "queue start on $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 0. Wait for the DocLayNet build (up to 2 h), then validate it once; document runs only go if it passes.
for i in $(seq 1 720); do
  ready=1; for d in ${DOC//,/ }; do [ -f "$WS/data/vqa/$d/_READY" ] || ready=0; done
  [ $ready -eq 1 ] && break; sleep 10
done
DOC_OK=0
if [ $ready -eq 1 ] && "$PY" "$WS/data_validator.py" $(for d in ${DOC//,/ }; do echo "$WS/data/vqa/$d"; done) \
     > "$LOGS/validate-doclaynet.log" 2>&1; then
  DOC_OK=1; log "DocLayNet ready and valid"
else
  log "DocLayNet missing or INVALID (see $LOGS/validate-doclaynet.log): document runs skipped"
  printf "%s\tdoclaynet-validate\tfailed\t0\n" "$(date '+%F %T')" >> "$STATUS"
fi

if [ $DOC_OK -eq 1 ]; then
  # 1-3. Tiling on documents: same data, same recipe, only the image views differ (the core experiment).
  run doc-split0    60  --datasets "$DOC" $DOCARGS --image-split-edge 0
  run doc-split1024 120 --datasets "$DOC" $DOCARGS --image-split-edge 1024
  run doc-split1536 200 --datasets "$DOC" $DOCARGS --image-split-edge 1536
fi
# 4. Photo baseline recipe (batch 16, 2 passes) + act head: vs baseline-v2 isolates the act head's effect.
run vqa-act       40  --datasets "$VQA"
# 5. Same at half the backbone LR (baseline-v2's calib NLL swung 0.62 -> 1.13): vs vqa-act isolates the LR.
run vqa-lr1e5     40  --datasets "$VQA" --lr-backbone 1e-5
if [ $DOC_OK -eq 1 ]; then
  # 6. Generalist: photos + documents + Laya's text-only decisions in one model at native document resolution.
  #    Long text states need gradient checkpointing (batch 16 ran out of GPU memory with them).
  run mix-split1024 180 --datasets "$VQA,$DOC,$TEXT" $DOCARGS --image-split-edge 1024 --grad-checkpointing
  # 7. Most tiles last (largest memory and time; nothing above depends on it).
  run doc-split2048 300 --datasets "$DOC" $DOCARGS --image-split-edge 2048
fi

log "latency"
DOCS=$(ls -d "$CKPT"/doc-* "$CKPT"/mix-* 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done)
PHOTOS=$(ls -d "$CKPT"/baseline-v2-* "$CKPT"/vqa-* "$CKPT"/mix-* 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done)
[ -n "$DOCS" ] && (cd "$REPO" && "$PY" scripts/visionjev_latency.py $DOCS --n 100 --datasets "$DOC" 2>>"$LOGS/latency.err" \
  | tee "$LOGS/latency-docs.log" | sed -n '/^{/,$p' > "$RES/latency-docs.json")
[ -n "$PHOTOS" ] && (cd "$REPO" && "$PY" scripts/visionjev_latency.py $PHOTOS --n 100 --datasets "$VQA" 2>>"$LOGS/latency.err" \
  | tee "$LOGS/latency-photos.log" | sed -n '/^{/,$p' > "$RES/latency-photos.json")

log "summary"
(cd "$REPO" && "$PY" scripts/visionjev_summarize.py $DOCS --latency "$RES/latency-docs.json" --out "$RES/summary-docs.md" > /dev/null)
(cd "$REPO" && "$PY" scripts/visionjev_summarize.py $PHOTOS --latency "$RES/latency-photos.json" --out "$RES/summary-photos.md" > /dev/null)
cat "$RES/summary-docs.md" "$RES/summary-photos.md" > "$RES/summary.md" 2>/dev/null
log "queue done"
