#!/usr/bin/env bash
# Teacher-lite generation on one H100 80GB (Hypothesis/V6.2.md section 2 step 4; laya/visionjev_teacher.py): vLLM serves
# one model at a time (FP8 Qwen checkpoints so each fits with KV room):
#   writer    Qwen/Qwen3.6-27B-FP8 (thinking)        -> teacher/writer.jsonl
#   answerer  Qwen/Qwen3.6-35B-A3B-FP8 (thinking)    -> teacher/answers-qwen35a3b.jsonl
#   answerer  openai/gpt-oss-20b (reasoning medium)  -> teacher/answers-gptoss20b.jsonl
# All three are Apache-2.0 (HF API, 2026-09-29). Every stage resumes from its output file. Assembly (JevBench lint, splits)
# runs locally after the fetch. The last line of $DONE is always "EXIT_CODE=<rc>" (trap; PROBLEMS I5).
#
#   setsid nohup bash scripts/pod_teacher.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
WS=/workspace/VisionJev
LOGS=$WS/data/logs
OUT=$WS/teacher
DONE=$LOGS/teacher.log
DOCS=${DOCS:-3000}
mkdir -p "$LOGS" "$OUT"
rc=0
SERVER=
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null; echo "EXIT_CODE=$rc" >> "$DONE"' EXIT
note() { echo "[$(date -u +%H:%M)] $*" >> "$DONE"; }
step() {
  local name=$1; shift
  note "start $name"
  "$@" > "$LOGS/teacher-$name.log" 2>&1
  local r=$?
  note "end $name rc=$r"
  [ $r -ne 0 ] && [ $rc -eq 0 ] && rc=$r
  return $r
}
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
# DeepGEMM JIT-compiles FP8 kernels and needs nvcc >= 12.9; the runpod image has CUDA 12.8 -> CUTLASS FP8 kernels instead
export VLLM_USE_DEEP_GEMM=0
# FlashInfer JIT-compiles its top-k sampler with ninja + nvcc on first use: use vLLM's PyTorch sampler instead
export VLLM_USE_FLASHINFER_SAMPLER=0
VENV=${VLLM_VENV:-/root/vllm-venv}  # container disk: the /workspace network FS is slow for many small files
VLLM=$VENV/bin
export PATH=$VLLM:/usr/local/cuda/bin:$PATH  # ninja (installed below) and nvcc for any other JIT step
URL=http://127.0.0.1:8000/v1

setup() {
  command -v uv >/dev/null || pip install -q uv
  [ -x $VLLM/python ] || uv venv --python 3.12 $VENV
  uv pip install --python $VLLM/python -q "vllm==0.30.0" ninja
  $VLLM/python -c "import vllm, torch; print('vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"
}
serve() {  # serve <model> <revision> <extra vllm args...>: start, wait until it answers, leave running
  local model=$1 rev=$2; shift 2
  $VLLM/hf download "$model" --revision "$rev" > /dev/null || return 1
  $VLLM/vllm serve "$model" --revision "$rev" --served-model-name "$model" --port 8000 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-seqs 256 "$@" > "$LOGS/teacher-vllm-$(basename $model).log" 2>&1 &
  SERVER=$!
  for i in $(seq 1 120); do
    curl -sf $URL/models > /dev/null && return 0
    kill -0 $SERVER 2>/dev/null || return 1
    sleep 15
  done
  return 1
}
stop() { [ -n "$SERVER" ] && kill $SERVER && wait $SERVER 2>/dev/null; SERVER=; sleep 5; }

step setup setup || exit
cd $WS/laya-vision
python3 -m laya.visionjev_teacher plan --docs $DOCS --out $OUT/plan.jsonl >> "$DONE" || { rc=1; exit; }

W=Qwen/Qwen3.6-27B-FP8
step serve-writer serve $W e89b16ebf1988b3d6befa7de50abc2d76f26eb09 --reasoning-parser qwen3 || exit
# pilot: 60 documents first; stop the pod if most writer outputs are unusable (prompt / parser problem, not worth 3 h)
head -60 $OUT/plan.jsonl > $OUT/plan-pilot.jsonl
step write-pilot python3 -m laya.visionjev_teacher write --plan $OUT/plan-pilot.jsonl --url $URL --model $W \
  --out $OUT/writer.jsonl --workers 48 --max-tokens 12000
OK=$(grep -c '"ok": true' $OUT/writer.jsonl)
note "pilot: $OK of 60 usable; reasons: $(grep -o '"reject": "[^"]*"' $OUT/writer.jsonl | sort | uniq -c | tr '\n' ';')"
[ "$OK" -ge 30 ] || { note "pilot below 50%: stopping"; rc=3; exit; }
step write python3 -m laya.visionjev_teacher write --plan $OUT/plan.jsonl --url $URL --model $W --out $OUT/writer.jsonl \
  --workers 48 --max-tokens 12000
stop
note "writer ok: $(grep -c '"ok": true' $OUT/writer.jsonl) of $(wc -l < $OUT/writer.jsonl)"

A1=Qwen/Qwen3.6-35B-A3B-FP8
step serve-a1 serve $A1 95a723d08a9490559dae23d0cff1d9466213d989 --reasoning-parser qwen3 || exit
step answer-a1 python3 -m laya.visionjev_teacher answer --writer $OUT/writer.jsonl --url $URL --model $A1 \
  --name qwen35a3b --out $OUT/answers-qwen35a3b.jsonl --workers 96 --max-tokens 8000
stop

A2=openai/gpt-oss-20b
step serve-a2 serve $A2 6cee5e81ee83917806bbde320786a8fb61efebee || exit
step answer-a2 python3 -m laya.visionjev_teacher answer --writer $OUT/writer.jsonl --url $URL --model $A2 \
  --name gptoss20b --out $OUT/answers-gptoss20b.jsonl --workers 128 --max-tokens 8000 --extra '{"reasoning_effort": "medium"}'
stop
for f in $OUT/answers-*.jsonl; do note "$(basename $f): $(grep -vc '"probs": null' $f) parsed of $(wc -l < $f)"; done
