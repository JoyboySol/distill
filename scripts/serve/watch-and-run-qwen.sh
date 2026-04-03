#!/usr/bin/env bash

set -euo pipefail

BASE_PORT=6758
MODEL_PATH="/mnt/hdd/Qwen3-30B-A3B-Instruct-2507"
GPUS_STR="0 1 2 3 4 5 6 7"
MODEL_NAME="Qwen3-30B-A3B-Instruct-2507"
MAX_MODEL_LEN=16384
UTIL_THRESHOLD=10
MEM_THRESHOLD_MB=2048
POLL_INTERVAL=30
READY_TIMEOUT=1800
READY_CHECK_INTERVAL=5
SERVE_SCRIPT="/mnt/ssd/lvzhihao/data-pipeline/scripts/serve/serve-qwen.sh"
LOAD_SCRIPT="/mnt/ssd/lvzhihao/data-pipeline/scripts/serve/run-qwen-load.sh"
SERVE_ARGS=""
LOAD_ARGS=""
SCRIPT_NAME="$(basename "$0")"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]
Usage: $SCRIPT_NAME key=value [key=value ...]

Watch configured GPUs and start Qwen serving on the GPUs that are currently idle.

Options:
  --base-port PORT            Base port for serve-qwen.sh (default: $BASE_PORT)
  --model-path PATH           Model path for serve-qwen.sh (default: $MODEL_PATH)
  --gpus "0 1"                Space-separated GPU ids (default: "$GPUS_STR")
  --model-name NAME           Model name (default: $MODEL_NAME)
  --max-model-len LEN         Max model length (default: $MAX_MODEL_LEN)
  --util-threshold PCT        Idle util threshold percent (default: $UTIL_THRESHOLD)
  --mem-threshold-mb MB       Idle memory threshold in MB (default: $MEM_THRESHOLD_MB)
  --poll-interval SEC         GPU poll interval in seconds (default: $POLL_INTERVAL)
  --ready-timeout SEC         Max wait for service readiness (default: $READY_TIMEOUT)
  --ready-check-interval SEC  Port readiness poll interval (default: $READY_CHECK_INTERVAL)
  --serve-script PATH         Serve script path or command (default: $SERVE_SCRIPT)
  --load-script PATH          Load script path or command (default: $LOAD_SCRIPT)
  --serve-args "..."          Extra args appended to serve script
  --load-args "..."           Extra args appended to load script
  --help                      Show this help

Key=value forms:
  base_port=6758
  model_path=/mnt/hdd/Qwen3-30B-A3B-Instruct-2507
  gpus="0 1 2 3 4 5 6 7"
  model_name=Qwen3-30B-A3B-Instruct-2507
  max_model_len=16384
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

set_config_value() {
  local key=$1
  local value=$2

  case "$key" in
    base_port|BASE_PORT)
      BASE_PORT="$value"
      ;;
    model_path|MODEL_PATH)
      MODEL_PATH="$value"
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
    util_threshold|UTIL_THRESHOLD)
      UTIL_THRESHOLD="$value"
      ;;
    mem_threshold_mb|MEM_THRESHOLD_MB)
      MEM_THRESHOLD_MB="$value"
      ;;
    poll_interval|POLL_INTERVAL)
      POLL_INTERVAL="$value"
      ;;
    ready_timeout|READY_TIMEOUT)
      READY_TIMEOUT="$value"
      ;;
    ready_check_interval|READY_CHECK_INTERVAL)
      READY_CHECK_INTERVAL="$value"
      ;;
    serve_script|SERVE_SCRIPT)
      SERVE_SCRIPT="$value"
      ;;
    load_script|LOAD_SCRIPT)
      LOAD_SCRIPT="$value"
      ;;
    serve_args|SERVE_ARGS)
      SERVE_ARGS="$value"
      ;;
    load_args|LOAD_ARGS)
      LOAD_ARGS="$value"
      ;;
    *)
      printf 'Unknown argument key: %s\n\n' "$key" >&2
      usage >&2
      exit 1
      ;;
  esac
}

parse_args() {
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
      --util-threshold)
        UTIL_THRESHOLD="$2"
        shift 2
        ;;
      --mem-threshold-mb)
        MEM_THRESHOLD_MB="$2"
        shift 2
        ;;
      --poll-interval)
        POLL_INTERVAL="$2"
        shift 2
        ;;
      --ready-timeout)
        READY_TIMEOUT="$2"
        shift 2
        ;;
      --ready-check-interval)
        READY_CHECK_INTERVAL="$2"
        shift 2
        ;;
      --serve-script)
        SERVE_SCRIPT="$2"
        shift 2
        ;;
      --load-script)
        LOAD_SCRIPT="$2"
        shift 2
        ;;
      --serve-args)
        SERVE_ARGS="$2"
        shift 2
        ;;
      --load-args)
        LOAD_ARGS="$2"
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *=*)
        set_config_value "${1%%=*}" "${1#*=}"
        shift
        ;;
      *)
        printf 'Unknown argument: %s\n\n' "$1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
}

build_ports() {
  local gpu_count=$1
  local ports=()
  local i

  for ((i = 0; i < gpu_count; i++)); do
    ports+=("$((BASE_PORT + i))")
  done

  printf '%s\n' "${ports[@]}"
}

query_gpu_status() {
  local gpu_id=$1
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$gpu_id"
}

gpu_is_idle() {
  local gpu_id=$1
  local status util mem

  if ! status="$(query_gpu_status "$gpu_id" 2>/dev/null)"; then
    log "GPU $gpu_id query failed; will retry"
    return 1
  fi

  status="${status%%$'\n'*}"
  util="$(printf '%s' "$status" | cut -d',' -f1 | tr -d ' ')"
  mem="$(printf '%s' "$status" | cut -d',' -f2 | tr -d ' ')"

  if [[ -z "$util" || -z "$mem" ]]; then
    log "GPU $gpu_id returned malformed status: $status"
    return 1
  fi

  [[ "$util" =~ ^[0-9]+$ ]] || return 1
  [[ "$mem" =~ ^[0-9]+$ ]] || return 1

  (( util < UTIL_THRESHOLD && mem < MEM_THRESHOLD_MB ))
}

all_gpus_idle() {
  local gpu_id

  for gpu_id in "$@"; do
    if ! gpu_is_idle "$gpu_id"; then
      return 1
    fi
  done

  return 0
}

get_idle_gpus() {
  local gpu_id
  local idle_gpus=()

  for gpu_id in "$@"; do
    if gpu_is_idle "$gpu_id"; then
      idle_gpus+=("$gpu_id")
    fi
  done

  printf '%s\n' "${idle_gpus[@]}"
}

port_ready() {
  local port=$1
  curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null
}

all_ports_ready() {
  local port

  for port in "$@"; do
    if ! port_ready "$port"; then
      return 1
    fi
  done

  return 0
}

wait_for_idle_gpus() {
  local gpu_ids=("$@")
  local idle_gpus=()

  while true; do
    mapfile -t idle_gpus < <(get_idle_gpus "${gpu_ids[@]}")

    if [[ "${#idle_gpus[@]}" -gt 0 ]]; then
      log "Idle GPUs available: ${idle_gpus[*]}" >&2
      printf '%s\n' "${idle_gpus[@]}"
      return 0
    fi

    log "Waiting for at least one idle GPU among: ${gpu_ids[*]}" >&2
    sleep "$POLL_INTERVAL"
  done
}

wait_for_all_ports_ready() {
  local ports=("$@")
  local start_ts now

  start_ts="$(date +%s)"

  while true; do
    if all_ports_ready "${ports[@]}"; then
      log "All target ports are ready: ${ports[*]}"
      return 0
    fi

    now="$(date +%s)"
    if (( now - start_ts >= READY_TIMEOUT )); then
      log "Timed out waiting for ports: ${ports[*]}"
      return 1
    fi

    log "Waiting for ports to become ready: ${ports[*]}"
    sleep "$READY_CHECK_INTERVAL"
  done
}

run_serve() {
  local gpus=("$@")
  local serve_args=()
  local gpus_joined="${gpus[*]}"

  if [[ -n "$SERVE_ARGS" ]]; then
    read -r -a serve_args <<< "$SERVE_ARGS"
  fi

  log "Starting serve script: $SERVE_SCRIPT"
  "$SERVE_SCRIPT" \
    --base-port "$BASE_PORT" \
    --model-path "$MODEL_PATH" \
    --gpus "$gpus_joined" \
    --model-name "$MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${serve_args[@]}"
}

run_load() {
  local ports=("$@")
  local load_args=()
  local ports_joined="${ports[*]}"

  if [[ -n "$LOAD_ARGS" ]]; then
    read -r -a load_args <<< "$LOAD_ARGS"
  fi

  log "Starting load script: $LOAD_SCRIPT"
  "$LOAD_SCRIPT" \
    "$ports_joined" \
    "$MODEL_NAME" \
    "${load_args[@]}"
}

main() {
  local gpus idle_gpus ports

  parse_args "$@"
  read -r -a gpus <<< "$GPUS_STR"

  if [[ "${#gpus[@]}" -eq 0 ]]; then
    echo "No GPUs configured" >&2
    exit 1
  fi

  log "Watching GPUs: ${gpus[*]}"
  mapfile -t idle_gpus < <(wait_for_idle_gpus "${gpus[@]}")
  mapfile -t ports < <(build_ports "${#idle_gpus[@]}")
  log "Selected idle GPUs: ${idle_gpus[*]}"
  log "Target ports: ${ports[*]}"

  if all_ports_ready "${ports[@]}"; then
    log "Ports already ready; skipping serve startup"
  else
    run_serve "${idle_gpus[@]}"
    wait_for_all_ports_ready "${ports[@]}"
  fi

  run_load "${ports[@]}"
}

main "$@"
