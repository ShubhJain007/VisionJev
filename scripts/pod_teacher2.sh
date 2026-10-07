#!/usr/bin/env bash
# Teacher round 2 on one H100 (laya/visionjev_teacher.py, plan --round 2): 5 questions per state, families weighted to
# what B3 misses, six kinds of state (documents, workflow JSON like Laya's, agent traces, threads, tables, code changes).
# One pod does everything; the models load in an order that keeps every stage useful if the cap cuts the run:
#   Qwen3.6-27B-FP8 (thinking)      write, for WRITE_MIN minutes at most
#   Qwen3.6-35B-A3B-FP8             blind thinking answers -> reasoned distributions + rationale -> direct one-pass
#   gpt-oss-20b                     blind answers (second verifier)
#   Qwen3.6-27B-FP8 (no thinking)   direct one-pass distributions, only if time is left
# Outputs go to $WS/results/teacher2 (fetched by pod_autopilot.sh); every stage resumes from its file. Assembly runs
# locally. All three models are Apache-2.0.
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
OUT=$WS/results/teacher2
DONE=$LOGS/teacher2.log
DOCS=${DOCS:-3200}
WRITE_MIN=${WRITE_MIN:-270}
LAST_PASS_BY_MIN=${LAST_PASS_BY_MIN:-640}  # the optional last pass starts only before this many minutes
T0=$(date +%s)
mkdir -p "$LOGS" "$OUT"
rc=0
SERVER=
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null; echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/teacher2-$name.log" 2>&1
  local r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0  # PROBLEMS I8: the image's nvcc is too old for the JIT kernels
VENV=/root/vllm-venv
VLLM=$VENV/bin
export PATH=$VLLM:/usr/local/cuda/bin:$PATH
URL=http://127.0.0.1:8000/v1
NOTHINK='{"chat_template_kwargs": {"enable_thinking": false}}'

setup() {
  command -v uv >/dev/null || pip install -q uv
  [ -x $VLLM/python ] || uv venv --python 3.12 $VENV
  uv pip install --python $VLLM/python -q "vllm==0.30.0" ninja
  $VLLM/python -c "import vllm, torch; print('vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"
}
serve() {  # serve <model> <revision> <extra vllm args...>
  local model=$1 rev=$2; shift 2
  $VLLM/hf download "$model" --revision "$rev" > /dev/null || return 1
  $VLLM/vllm serve "$model" --revision "$rev" --served-model-name "$model" --port 8000 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-seqs 256 --max-logprobs 20 "$@" \
    > "$LOGS/teacher2-vllm-$(basename $model)-$(date +%H%M).log" 2>&1 &
  SERVER=$!
  for i in $(seq 1 120); do
    curl -sf $URL/models > /dev/null && return 0
    kill -0 $SERVER 2>/dev/null || return 1
    sleep 15
  done
  return 1
}
stop() { [ -n "$SERVER" ] && kill $SERVER && wait $SERVER 2>/dev/null; SERVER=; sleep 5; }
pass() {  # pass <model> <name> <mode> <workers> <max_tokens> [extra json]: 20-question smoke, then everything
  local model=$1 name=$2 mode=$3 workers=$4 mt=$5 extra=${6:-}
  local args=(--writer $OUT/writer.jsonl --url $URL --model "$model" --name "${name#*-}" --max-tokens $mt --mode $mode)
  [ -n "$extra" ] && args+=(--extra "$extra")
  python3 -m laya.visionjev_teacher answer "${args[@]}" --out $OUT/smoke-$name.jsonl --workers 20 --limit 20 \
    > $LOGS/teacher2-smoke-$name.log 2>&1
  local ok
  ok=$(grep -vc '"probs": null' $OUT/smoke-$name.jsonl)
  note "$name smoke: $ok/20 parsed"
  [ "$ok" -ge 14 ] || { note "$name: smoke below 14/20, pass skipped"; [ $rc -eq 0 ] && rc=3; return 1; }
  note "start $name"
  python3 -m laya.visionjev_teacher answer "${args[@]}" --out $OUT/$name.jsonl --workers $workers \
    > $LOGS/teacher2-$name.log 2>&1 || { [ $rc -eq 0 ] && rc=1; }
  note "$name: $(grep -vc '"probs": null' $OUT/$name.jsonl) parsed of $(wc -l < $OUT/$name.jsonl)"
}

step setup setup || exit
cd $WS/laya-vision
python3 -m laya.visionjev_teacher plan --round 2 --docs $DOCS --seed 1 --out $OUT/plan.jsonl >> "$DONE" || { rc=1; exit; }

W=Qwen/Qwen3.6-27B-FP8; WREV=e89b16ebf1988b3d6befa7de50abc2d76f26eb09
step serve-writer serve $W $WREV --reasoning-parser qwen3 || exit
# pilot: 60 states first; stop the pod if most writer outputs are unusable (prompt problem, not worth hours of GPU)
head -60 $OUT/plan.jsonl > $OUT/plan-pilot.jsonl
step write-pilot python3 -m laya.visionjev_teacher write --plan $OUT/plan-pilot.jsonl --url $URL --model $W \
  --out $OUT/writer.jsonl --workers 60 --max-tokens 16000
OK=$(grep -c '"ok": true' $OUT/writer.jsonl)
note "pilot: $OK of 60 usable; reasons: $(grep -o '"reject": "[^"]*"' $OUT/writer.jsonl | sort | uniq -c | tr '\n' ';')"
[ "$OK" -ge 27 ] || { note "pilot below 45%: stopping"; rc=3; exit; }
step write python3 -m laya.visionjev_teacher write --plan $OUT/plan.jsonl --url $URL --model $W --out $OUT/writer.jsonl \
  --workers 96 --max-tokens 16000 --minutes $WRITE_MIN
stop
note "writer ok: $(grep -c '"ok": true' $OUT/writer.jsonl) of $(wc -l < $OUT/writer.jsonl) written ($DOCS planned)"

A1=Qwen/Qwen3.6-35B-A3B-FP8
if step serve-a1 serve $A1 95a723d08a9490559dae23d0cff1d9466213d989 --reasoning-parser qwen3; then
  pass $A1 answers-qwen35a3b json 96 8000
  pass $A1 reasoned-qwen35a3b reasoned 96 8000
  pass $A1 direct-qwen35a3b direct 128 8 "$NOTHINK"
fi
stop
A2=openai/gpt-oss-20b
if step serve-a2 serve $A2 6cee5e81ee83917806bbde320786a8fb61efebee; then
  pass $A2 answers-gptoss20b json 128 8000 '{"reasoning_effort": "medium"}'
fi
stop
if [ $(( ($(date +%s) - T0) / 60 )) -lt "$LAST_PASS_BY_MIN" ]; then
  if step serve-direct27 serve $W $WREV --reasoning-parser qwen3; then pass $W direct-qwen27b direct 128 8 "$NOTHINK"; fi
  stop
else
  note "no time for the direct Qwen3.6-27B pass (optional)"
fi
note "done after $(( ($(date +%s) - T0) / 60 )) min"
