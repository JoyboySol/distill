#!/bin/bash

set -euo pipefail

source .venv/bin/activate

DEFAULT_BASE_PORT=6758
DEFAULT_MODEL_PATH="/mnt/hdd/Nanbeige4.1-3B"
DEFAULT_CACHE_ROOT="/mnt/ssd/cache_tmp"
DEFAULT_GPUS_STR="0 1 2 3 4 5 6 7"
DEFAULT_MODEL_NAME="Nanbeige4.1-3B"
DEFAULT_MAX_MODEL_LEN=65536

BASE_PORT="${BASE_PORT:-$DEFAULT_BASE_PORT}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
CACHE_ROOT="${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
GPUS_STR="${GPUS_STR:-$DEFAULT_GPUS_STR}"
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0")
  $(basename "$0") BASE_PORT MODEL_PATH GPUS_STR MODEL_NAME MAX_MODEL_LEN [CACHE_ROOT]
  $(basename "$0") --base-port 6758 --model-path /mnt/hdd/Qwen3-30B-A3B-Instruct-2507 --gpus "0 1 2 3 4 5 6 7" --model-name Qwen3-30B-A3B-Instruct-2507 --max-model-len 16384 --cache-root /mnt/ssd/cache_tmp
  $(basename "$0") base_port=6758 model_path=/mnt/hdd/Qwen3-30B-A3B-Instruct-2507 gpus="0 1 2 3 4 5 6 7" model_name=Qwen3-30B-A3B-Instruct-2507 max_model_len=16384 cache_root=/mnt/ssd/cache_tmp

Supported keys:
  base_port / BASE_PORT
  model_path / MODEL_PATH
  cache_root / CACHE_ROOT
  gpus / GPUS_STR
  model_name / MODEL_NAME
  max_model_len / MAX_MODEL_LEN
EOF
}

set_config_value() {
    local key="$1"
    local value="$2"

    case "$key" in
        base_port|BASE_PORT)
            BASE_PORT="$value"
            ;;
        model_path|MODEL_PATH)
            MODEL_PATH="$value"
            ;;
        cache_root|CACHE_ROOT)
            CACHE_ROOT="$value"
            ;;
        gpus|GPUS_STR)
            GPUS_STR="$value"
            ;;
        model_name|MODEL_NAME)
            MODEL_NAME="$value"
            ;;
        max_model_len|MAX_MODEL_LEN)
            MAX_MODEL_LEN="$value"
            ;;
        *)
            echo "Unknown argument key: $key" >&2
            usage >&2
            exit 1
            ;;
    esac
}

parse_args() {
    local positional=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --base-port)
                BASE_PORT="$2"
                shift 2
                ;;
            --model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --cache-root)
                CACHE_ROOT="$2"
                shift 2
                ;;
            --gpus)
                GPUS_STR="$2"
                shift 2
                ;;
            --model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --max-model-len)
                MAX_MODEL_LEN="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *=*)
                local key="${1%%=*}"
                local value="${1#*=}"
                set_config_value "$key" "$value"
                shift
                ;;
            *)
                positional+=("$1")
                shift
                ;;
        esac
    done

    if [[ ${#positional[@]} -gt 6 ]]; then
        echo "Too many positional arguments: ${#positional[@]}" >&2
        usage >&2
        exit 1
    fi

    if [[ ${#positional[@]} -ge 1 ]]; then
        BASE_PORT="${positional[0]}"
    fi
    if [[ ${#positional[@]} -ge 2 ]]; then
        MODEL_PATH="${positional[1]}"
    fi
    if [[ ${#positional[@]} -ge 3 ]]; then
        GPUS_STR="${positional[2]}"
    fi
    if [[ ${#positional[@]} -ge 4 ]]; then
        MODEL_NAME="${positional[3]}"
    fi
    if [[ ${#positional[@]} -ge 5 ]]; then
        MAX_MODEL_LEN="${positional[4]}"
    fi
    if [[ ${#positional[@]} -ge 6 ]]; then
        CACHE_ROOT="${positional[5]}"
    fi
}

parse_args "$@"

read -a GPUS <<< "$GPUS_STR"
mkdir -p logs

for ((i=0; i<${#GPUS[@]}; i++)); do
    GPU_ID="${GPUS[$i]}"
    PORT=$((BASE_PORT + i))

    INSTANCE_CACHE="${CACHE_ROOT%/}/vllm_cache_port_$PORT"
    mkdir -p "$INSTANCE_CACHE"

    echo "Starting vLLM on GPU $GPU_ID with Port $PORT for model $MODEL_NAME..."

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    VLLM_CACHE_ROOT="$INSTANCE_CACHE" \
    nohup vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.95 \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --reasoning-parser qwen3 \
        > "logs/serve_port_$PORT.log" 2>&1 &
done

echo "✅ All ${#GPUS[@]} vLLM instances started (one instance per GPU)."
echo "Model path: $MODEL_PATH"
echo "Cache root: $CACHE_ROOT"
echo "Served model name: $MODEL_NAME"
echo "Ports: ${BASE_PORT}-$((BASE_PORT + ${#GPUS[@]} - 1))"
