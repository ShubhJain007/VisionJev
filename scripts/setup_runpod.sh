#!/usr/bin/env bash
# Set up a Runpod pod for VisionJev training. Run ON THE POD after the code and data are on the network volume:
#   /workspace/VisionJev/{laya-vision, data/vqa, data_validator.py, results}
#
#   bash /workspace/VisionJev/laya-vision/scripts/setup_runpod.sh
#
# Creates /workspace/venv (persists on the volume, so a new pod reuses it), with the same pins as the local venv,
# puts the Hugging Face cache on the volume and checks CUDA. Then launch runs with scripts/visionjev_pod_run.sh.
set -euo pipefail
WS=/workspace/VisionJev
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" "$WS/results" "$WS/checkpoints" "$WS/data/logs"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
# torch 2.14 cu130 needs driver >= 580; otherwise use the cu128 build of the same torch
if [ "$DRIVER_MAJOR" -ge 580 ]; then IDX=https://download.pytorch.org/whl/cu130; else IDX=https://download.pytorch.org/whl/cu128; fi

command -v uv >/dev/null || pip install -q uv
# The venv lives on the container disk (/root: a 25-minute install on the /workspace network volume took ~1 minute
# there, 2026-09-30) and /workspace/venv points at it, so every script's /workspace/venv/bin/python keeps working.
# A container-disk venv does not survive a pod restart: this script is idempotent and simply rebuilds it.
if [ ! -x /workspace/venv/bin/python ]; then
  [ -d /workspace/venv ] && [ ! -L /workspace/venv ] && rm -rf /workspace/venv
  uv venv --python 3.12 /root/venv
  ln -sfn /root/venv /workspace/venv
fi
uv pip install --python /workspace/venv/bin/python -q --index-url "$IDX" --extra-index-url https://pypi.org/simple \
  "torch==2.14.0" "torchvision==0.29.0"
uv pip install --python /workspace/venv/bin/python -q "transformers==5.17.0" safetensors huggingface_hub hf_transfer \
  numpy pillow datasets peft==0.21.0 matplotlib num2words
uv pip install --python /workspace/venv/bin/python -q -e "$WS/laya-vision"

/workspace/venv/bin/python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "peft", peft.__version__)
PY
echo "SETUP_DONE"
