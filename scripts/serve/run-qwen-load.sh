#!/bin/bash

set -e

source .venv/bin/activate

# ===== 参数 =====
PORTS=${1:-"6758 6759 6760 6761 6762 6763 6764 6765"}
MODEL_NAME=${2:-"Qwen3-30B-A3B-Instruct-2507"}
CONCURRENCY=${3:-128}
DURATION=${4:-0}
PROMPT_CHARS=${5:-12000}
MAX_TOKENS=${6:-2048}

# 可选参数（压 GPU 建议）
TEMPERATURE=${7:-0}
TOP_P=${8:-1}

mkdir -p logs

LOG_FILE="logs/load_qwen_async_$(date +%Y%m%d_%H%M%S).log"

echo "Starting async load test..."
echo "Ports: ${PORTS}"
echo "Model: ${MODEL_NAME}"
echo "Concurrency: ${CONCURRENCY}"
echo "Prompt chars: ${PROMPT_CHARS}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Duration: ${DURATION}"
echo "Log: ${LOG_FILE}"

echo "----------------------------------------"

.venv/bin/python scripts/serve/load_qwen_async.py \
  --ports ${PORTS} \
  --model-name "${MODEL_NAME}" \
  --concurrency "${CONCURRENCY}" \
  --duration "${DURATION}" \
  --prompt-chars "${PROMPT_CHARS}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --timeout 1800 \
  --report-interval 5 \
  2>&1 | tee "${LOG_FILE}"
