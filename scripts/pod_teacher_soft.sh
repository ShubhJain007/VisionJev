#!/usr/bin/env bash
# Soft labels for teacher-lite (after pod_teacher.sh), laya/visionjev_teacher.py:
#   reasoned  Qwen3.6-35B-A3B thinking, Imajev-style honest distribution over the options + unknown, and a worked
#             2-3 sentence rationale (the rationale loss teaches the reasoning; github.com/mohit67890/imajev gen_answer.py)
#   direct    thinking OFF, one option number, distribution from its token log-probs, from Qwen3.6-35B-A3B and
#             Qwen3.6-27B: how plausible each option looks in one pass, as the student answers (thinking answers are
#             certain: final-token top prob 1.000 on a 20-question test)
# assemble later: target = 0.5 reasoned + 0.5 direct on rows every thinking teacher verified. Each pass runs a
# 20-question smoke first and is skipped below 15/20 parsed. Last line of $DONE is always EXIT_CODE (PROBLEMS I5).
#
#   setsid nohup bash scripts/pod_teacher_soft.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
OUT=$WS/teacher
DONE=$LOGS/teacher-soft.log
rc=0
SERVER=
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null; echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0
VLLM=/root/vllm-venv/bin
export PATH=$VLLM:/usr/local/cuda/bin:$PATH
URL=http://127.0.0.1:8000/v1
NOTHINK='{"chat_template_kwargs": {"enable_thinking": false}}'
cd $WS/laya-vision

serve() {
  local model=$1 rev=$2
  $VLLM/vllm serve "$model" --revision "$rev" --served-model-name "$model" --port 8000 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-seqs 256 --max-logprobs 20 --reasoning-parser qwen3 \
    > "$LOGS/teacher-soft-vllm-$(basename $model).log" 2>&1 &
  SERVER=$!
  for i in $(seq 1 120); do
    curl -sf $URL/models > /dev/null && return 0
    kill -0 $SERVER 2>/dev/null || return 1
    sleep 15
  done
  return 1
}
stop() { [ -n "$SERVER" ] && kill $SERVER; wait $SERVER 2>/dev/null; SERVER=; sleep 5; }

pass() {  # pass <model> <name> <mode> <workers> <max_tokens> [extra json]
  local model=$1 name=$2 mode=$3 workers=$4 mt=$5 extra=${6:-}
  local args=(--writer $OUT/writer.jsonl --url $URL --model "$model" --name "$name" --max-tokens $mt --mode $mode)
  [ -n "$extra" ] && args+=(--extra "$extra")
  python3 -m laya.visionjev_teacher answer "${args[@]}" --out $OUT/smoke-$name.jsonl --workers 20 --limit 20 \
    > $LOGS/teacher-soft-smoke-$name.log 2>&1
  local ok
  ok=$(grep -vc '"probs": null' $OUT/smoke-$name.jsonl)
  note "$name smoke: $ok/20 parsed; mean top prob $(python3 -c "
import json; r=[json.loads(l) for l in open('$OUT/smoke-$name.jsonl')]; r=[x for x in r if x['probs']]
print(round(sum(max(x['probs']) for x in r)/max(1,len(r)),3))")"
  [ "$ok" -ge 15 ] || { note "$name: smoke below 15/20, pass skipped"; rc=3; return 1; }
  python3 -m laya.visionjev_teacher answer "${args[@]}" --out $OUT/$name.jsonl --workers $workers \
    > $LOGS/teacher-soft-$name.log 2>&1 || rc=1
  note "$name: $(grep -vc '"probs": null' $OUT/$name.jsonl) parsed of $(wc -l < $OUT/$name.jsonl)"
}

A=Qwen/Qwen3.6-35B-A3B-FP8
note "serve $A"
if serve $A 95a723d08a9490559dae23d0cff1d9466213d989; then
  pass $A reasoned-qwen35a3b reasoned 96 8000
  pass $A direct-qwen35a3b direct 128 8 "$NOTHINK"
else
  note "serve $A failed"; rc=1
fi
stop
B=Qwen/Qwen3.6-27B-FP8
note "serve $B"
if serve $B e89b16ebf1988b3d6befa7de50abc2d76f26eb09; then
  pass $B direct-qwen27b direct 128 8 "$NOTHINK"
else
  note "serve $B failed"; rc=1
fi
stop
