#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_VENV_DIR="$ROOT_DIR/.venv"
DEFAULT_PYTHON_BIN="$DEFAULT_VENV_DIR/bin/python"
DEFAULT_VLLM_BIN="$DEFAULT_VENV_DIR/bin/vllm"
DEFAULT_CONFIG_PATH="$ROOT_DIR/manifest/opencode_reasoning_with_tests.yaml"
DEFAULT_BASE_PORT=1597
DEFAULT_RANDOMIZE_PORTS=1
DEFAULT_PORT_RANGE_START=20000
DEFAULT_PORT_RANGE_END=45000
DEFAULT_MODEL_PATH="/mnt/hdd/Nanbeige4.1-3B"
DEFAULT_CACHE_ROOT="/mnt/ssd/cache_tmp"
DEFAULT_GPUS_STR="0 1 2 3 4 5"
DEFAULT_MODEL_NAME="Nanbeige4.1-3B"
DEFAULT_MAX_MODEL_LEN=65536
DEFAULT_GPU_MEMORY_UTILIZATION=0.95
DEFAULT_TENSOR_PARALLEL_SIZE=1
DEFAULT_REASONING_PARSER="qwen3"
DEFAULT_WAIT_TIMEOUT_SEC=1800
DEFAULT_HEALTH_CHECK_INTERVAL_SEC=5
DEFAULT_API_KEY="EMPTY"
DEFAULT_LLM_TIMEOUT=3600
DEFAULT_VLLM_LS_COMMAND="${DISTILL_VLLM_LS_COMMAND:-${VLLM_LS_BIN:-/mnt/ssd/yulan/bin/vllm_ls}}"

PYTHON_BIN_EXPLICIT=0
VLLM_BIN_EXPLICIT=0
if [[ -n "${PYTHON_BIN+x}" ]]; then
    PYTHON_BIN_EXPLICIT=1
fi
if [[ -n "${VLLM_BIN+x}" ]]; then
    VLLM_BIN_EXPLICIT=1
fi

VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
VLLM_BIN="${VLLM_BIN:-$DEFAULT_VLLM_BIN}"
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"
TASK_NAME="${TASK_NAME:-}"
BASE_PORT="${BASE_PORT:-$DEFAULT_BASE_PORT}"
PORTS_SPEC="${PORTS_SPEC:-}"
RANDOMIZE_PORTS="${RANDOMIZE_PORTS:-$DEFAULT_RANDOMIZE_PORTS}"
PORT_RANGE_START="${PORT_RANGE_START:-$DEFAULT_PORT_RANGE_START}"
PORT_RANGE_END="${PORT_RANGE_END:-$DEFAULT_PORT_RANGE_END}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
CACHE_ROOT="${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
GPUS_STR="${GPUS_STR:-$DEFAULT_GPUS_STR}"
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TENSOR_PARALLEL_SIZE}"
REASONING_PARSER="${REASONING_PARSER:-$DEFAULT_REASONING_PARSER}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-$DEFAULT_WAIT_TIMEOUT_SEC}"
HEALTH_CHECK_INTERVAL_SEC="${HEALTH_CHECK_INTERVAL_SEC:-$DEFAULT_HEALTH_CHECK_INTERVAL_SEC}"
API_KEY="${API_KEY:-$DEFAULT_API_KEY}"
LLM_TIMEOUT="${LLM_TIMEOUT:-$DEFAULT_LLM_TIMEOUT}"
VLLM_LS_COMMAND="${VLLM_LS_COMMAND:-$DEFAULT_VLLM_LS_COMMAND}"
SKIP_SERVE="${SKIP_SERVE:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"

SCRIPT_NAME="$(basename "$0")"
STARTED_PIDS=()
FORWARD_ARGS=()
PORTS=()
GPUS=()
PORT_LOCK_FDS=()

usage() {
    cat <<EOF
Usage:
  $SCRIPT_NAME [options] [-- distill_extra_args...]
  $SCRIPT_NAME key=value [key=value ...] [-- distill_extra_args...]

One-command workflow:
  1. Start one vLLM instance per GPU.
  2. Wait until all ports are healthy.
  3. Run distill with the configured manifest and matched ports.

Options:
  --config PATH                     Manifest YAML to run (default: $CONFIG_PATH)
  --task NAME                       Optional task_name inside the manifest
  --base-port PORT                  Base port when deriving ports from GPUs if random ports are disabled (default: $BASE_PORT)
  --ports SPEC                      Explicit ports, e.g. 1597-1602 or 1597,1599,1601
  --randomize-ports                 Randomly choose free ports per run (default behavior)
  --no-randomize-ports              Disable random ports and derive from --base-port
  --port-range-start PORT           Start of random port range (default: $PORT_RANGE_START)
  --port-range-end PORT             End of random port range (default: $PORT_RANGE_END)
  --model-path PATH                 Model path for vLLM serve (default: $MODEL_PATH)
  --cache-root PATH                 Cache root for per-port vLLM cache (default: $CACHE_ROOT)
  --gpus "0 1 2"                    Space-separated GPU ids (default: "$GPUS_STR")
  --model-name NAME                 Served model name (default: $MODEL_NAME)
  --max-model-len LEN               Max model length (default: $MAX_MODEL_LEN)
  --gpu-memory-utilization FLOAT    vLLM GPU memory utilization (default: $GPU_MEMORY_UTILIZATION)
  --tensor-parallel-size N          vLLM tensor parallel size (default: $TENSOR_PARALLEL_SIZE)
  --reasoning-parser NAME           vLLM reasoning parser (default: $REASONING_PARSER)
  --serve-extra-args "..."          Extra arguments appended to vLLM serve
  --wait-timeout-sec SEC            Max wait for all ports to become healthy (default: $WAIT_TIMEOUT_SEC)
  --health-check-interval-sec SEC   Readiness poll interval (default: $HEALTH_CHECK_INTERVAL_SEC)
  --api-key KEY                     API key passed to distill (default: $API_KEY)
  --llm-timeout SEC                Distill per-request LLM timeout in seconds (default: $LLM_TIMEOUT)
  --vllm-ls-command CMD            Command used by distill to verify local backend processes (default: $VLLM_LS_COMMAND)
  --venv DIR                        Virtualenv dir (default: $VENV_DIR)
  --python-bin PATH                 Python executable for distill (default: $PYTHON_BIN)
  --vllm-bin PATH                   vLLM executable (default: $VLLM_BIN)
  --skip-serve                      Reuse existing services and only run readiness + distill
  --keep-servers                    Keep started vLLM processes alive after distill exits
  --help, -h                        Show this help

Key=value forms:
  config=/path/to/config.yaml
  task=opencode_reasoning_split_0
  base_port=1597
  ports=1597-1602
  randomize_ports=1
  port_range_start=20000
  port_range_end=45000
  model_path=/mnt/hdd/Nanbeige4.1-3B
  cache_root=/mnt/ssd/cache_tmp
  gpus="0 1 2 3 4 5"
  model_name=Nanbeige4.1-3B
  max_model_len=65536
  gpu_memory_utilization=0.95
  tensor_parallel_size=1
  reasoning_parser=qwen3
  wait_timeout_sec=1800
  health_check_interval_sec=5
  skip_serve=1
  keep_servers=1
  llm_timeout=3600
  vllm_ls_command='ps -eo args='

Forwarding extra args to distill:
  $SCRIPT_NAME --task opencode_reasoning_split_0 -- --range-start 0 --range-end 10

Examples:
  $SCRIPT_NAME
  $SCRIPT_NAME --task opencode_reasoning_split_0
  $SCRIPT_NAME --no-randomize-ports --base-port 1597
  $SCRIPT_NAME --skip-serve --ports 1597-1602 --task opencode_reasoning_split_0
EOF
}

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

set_config_value() {
    local key="$1"
    local value="$2"

    case "$key" in
        config|CONFIG_PATH)
            CONFIG_PATH="$value"
            ;;
        task|TASK_NAME)
            TASK_NAME="$value"
            ;;
        base_port|BASE_PORT)
            BASE_PORT="$value"
            ;;
        ports|PORTS_SPEC)
            PORTS_SPEC="$value"
            ;;
        randomize_ports|RANDOMIZE_PORTS)
            RANDOMIZE_PORTS="$value"
            ;;
        port_range_start|PORT_RANGE_START)
            PORT_RANGE_START="$value"
            ;;
        port_range_end|PORT_RANGE_END)
            PORT_RANGE_END="$value"
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
        gpu_memory_utilization|GPU_MEMORY_UTILIZATION)
            GPU_MEMORY_UTILIZATION="$value"
            ;;
        tensor_parallel_size|TENSOR_PARALLEL_SIZE)
            TENSOR_PARALLEL_SIZE="$value"
            ;;
        reasoning_parser|REASONING_PARSER)
            REASONING_PARSER="$value"
            ;;
        wait_timeout_sec|WAIT_TIMEOUT_SEC)
            WAIT_TIMEOUT_SEC="$value"
            ;;
        health_check_interval_sec|HEALTH_CHECK_INTERVAL_SEC)
            HEALTH_CHECK_INTERVAL_SEC="$value"
            ;;
        api_key|API_KEY)
            API_KEY="$value"
            ;;
        llm_timeout|LLM_TIMEOUT)
            LLM_TIMEOUT="$value"
            ;;
        vllm_ls_command|VLLM_LS_COMMAND)
            VLLM_LS_COMMAND="$value"
            ;;
        venv|VENV_DIR)
            VENV_DIR="$value"
            ;;
        python_bin|PYTHON_BIN)
            PYTHON_BIN="$value"
            PYTHON_BIN_EXPLICIT=1
            ;;
        vllm_bin|VLLM_BIN)
            VLLM_BIN="$value"
            VLLM_BIN_EXPLICIT=1
            ;;
        skip_serve|SKIP_SERVE)
            SKIP_SERVE="$value"
            ;;
        keep_servers|KEEP_SERVERS)
            KEEP_SERVERS="$value"
            ;;
        serve_extra_args|SERVE_EXTRA_ARGS)
            SERVE_EXTRA_ARGS="$value"
            ;;
        *)
            die "Unknown argument key: $key"
            ;;
    esac
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                CONFIG_PATH="$2"
                shift 2
                ;;
            --task)
                TASK_NAME="$2"
                shift 2
                ;;
            --base-port)
                BASE_PORT="$2"
                shift 2
                ;;
            --ports)
                PORTS_SPEC="$2"
                shift 2
                ;;
            --randomize-ports)
                RANDOMIZE_PORTS=1
                shift
                ;;
            --no-randomize-ports)
                RANDOMIZE_PORTS=0
                shift
                ;;
            --port-range-start)
                PORT_RANGE_START="$2"
                shift 2
                ;;
            --port-range-end)
                PORT_RANGE_END="$2"
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
            --gpu-memory-utilization)
                GPU_MEMORY_UTILIZATION="$2"
                shift 2
                ;;
            --tensor-parallel-size)
                TENSOR_PARALLEL_SIZE="$2"
                shift 2
                ;;
            --reasoning-parser)
                REASONING_PARSER="$2"
                shift 2
                ;;
            --serve-extra-args)
                SERVE_EXTRA_ARGS="$2"
                shift 2
                ;;
            --wait-timeout-sec)
                WAIT_TIMEOUT_SEC="$2"
                shift 2
                ;;
            --health-check-interval-sec)
                HEALTH_CHECK_INTERVAL_SEC="$2"
                shift 2
                ;;
            --api-key)
                API_KEY="$2"
                shift 2
                ;;
            --llm-timeout)
                LLM_TIMEOUT="$2"
                shift 2
                ;;
            --vllm-ls-command)
                VLLM_LS_COMMAND="$2"
                shift 2
                ;;
            --venv)
                VENV_DIR="$2"
                shift 2
                ;;
            --python-bin)
                PYTHON_BIN="$2"
                PYTHON_BIN_EXPLICIT=1
                shift 2
                ;;
            --vllm-bin)
                VLLM_BIN="$2"
                VLLM_BIN_EXPLICIT=1
                shift 2
                ;;
            --skip-serve)
                SKIP_SERVE=1
                shift
                ;;
            --keep-servers)
                KEEP_SERVERS=1
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            --)
                shift
                FORWARD_ARGS=("$@")
                break
                ;;
            *=*)
                set_config_value "${1%%=*}" "${1#*=}"
                shift
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
    done
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || die "File not found: $path"
}

require_executable() {
    local path="$1"
    [[ -x "$path" ]] || die "Executable not found: $path"
}

sanitize_proxy_env() {
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}"
    export no_proxy="$NO_PROXY"
}

expand_ports_spec() {
    local spec="${1//,/ }"
    local token
    local start
    local end
    local port

    for token in $spec; do
        [[ -n "$token" ]] || continue
        if [[ "$token" == *-* ]]; then
            start="${token%-*}"
            end="${token#*-}"
            [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]] || die "Invalid port range: $token"
            (( end >= start )) || die "Invalid port range: $token"
            for ((port = start; port <= end; port++)); do
                printf '%s\n' "$port"
            done
        else
            [[ "$token" =~ ^[0-9]+$ ]] || die "Invalid port: $token"
            printf '%s\n' "$token"
        fi
    done
}

build_ports() {
    local i
    if [[ -n "$PORTS_SPEC" ]]; then
        mapfile -t PORTS < <(expand_ports_spec "$PORTS_SPEC")
    elif [[ "$RANDOMIZE_PORTS" == "1" ]]; then
        mapfile -t PORTS < <(pick_random_ports "${#GPUS[@]}")
    else
        for ((i = 0; i < ${#GPUS[@]}; i++)); do
            PORTS+=("$((BASE_PORT + i))")
        done
    fi
    (( ${#PORTS[@]} > 0 )) || die "No ports resolved."
}

port_list_string() {
    local joined=""
    local port
    for port in "${PORTS[@]}"; do
        joined+="${joined:+,}$port"
    done
    printf '%s' "$joined"
}

is_port_listening() {
    local port="$1"
    ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$port$"
}

random_port_candidate() {
    local span=$((PORT_RANGE_END - PORT_RANGE_START + 1))
    printf '%s\n' "$((PORT_RANGE_START + RANDOM % span))"
}

pick_random_ports() {
    local count="$1"
    local attempts=0
    local max_attempts
    local candidate
    local chosen=()
    local existing

    [[ "$PORT_RANGE_START" =~ ^[0-9]+$ ]] || die "Invalid PORT_RANGE_START: $PORT_RANGE_START"
    [[ "$PORT_RANGE_END" =~ ^[0-9]+$ ]] || die "Invalid PORT_RANGE_END: $PORT_RANGE_END"
    (( PORT_RANGE_START >= 1024 )) || die "PORT_RANGE_START must be >= 1024"
    (( PORT_RANGE_END <= 65535 )) || die "PORT_RANGE_END must be <= 65535"
    (( PORT_RANGE_END >= PORT_RANGE_START )) || die "PORT_RANGE_END must be >= PORT_RANGE_START"
    (( count <= PORT_RANGE_END - PORT_RANGE_START + 1 )) || die "Random port range is too small for $count GPU(s)"

    max_attempts=$((count * 200))
    while (( ${#chosen[@]} < count && attempts < max_attempts )); do
        attempts=$((attempts + 1))
        candidate="$(random_port_candidate)"
        if is_port_listening "$candidate"; then
            continue
        fi

        for existing in "${chosen[@]}"; do
            if [[ "$existing" == "$candidate" ]]; then
                candidate=""
                break
            fi
        done

        [[ -n "$candidate" ]] || continue
        chosen+=("$candidate")
    done

    (( ${#chosen[@]} == count )) || die "Failed to find $count free random ports in range ${PORT_RANGE_START}-${PORT_RANGE_END}"
    printf '%s\n' "${chosen[@]}"
}

reserve_port_locks() {
    local port
    local fd
    local lock_path

    mkdir -p "$ROOT_DIR/.port_locks"

    for port in "${PORTS[@]}"; do
        lock_path="$ROOT_DIR/.port_locks/${port}.lock"
        exec {fd}>"$lock_path"
        if ! flock -n "$fd"; then
            die "Port $port is reserved by another run.sh instance. Retry, or pass --ports to choose ports explicitly."
        fi
        PORT_LOCK_FDS+=("$fd")
    done
}

ensure_ports_available() {
    local port
    for port in "${PORTS[@]}"; do
        if is_port_listening "$port"; then
            die "Port $port is already listening. Use --skip-serve to reuse existing services or change --ports/--base-port."
        fi
    done
}

port_ready() {
    local port="$1"
    curl --noproxy '*' -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null
}

wait_for_ports_ready() {
    local deadline=$((SECONDS + WAIT_TIMEOUT_SEC))
    local port
    local all_ready

    while (( SECONDS < deadline )); do
        all_ready=1
        for port in "${PORTS[@]}"; do
            if ! port_ready "$port"; then
                all_ready=0
                break
            fi
        done

        if (( all_ready == 1 )); then
            log "All ports are healthy: $(port_list_string)"
            return 0
        fi

        sleep "$HEALTH_CHECK_INTERVAL_SEC"
    done

    printf 'Timed out waiting for ports: %s\n' "$(port_list_string)" >&2
    for port in "${PORTS[@]}"; do
        if [[ -f "$ROOT_DIR/logs/serve_port_${port}.log" ]]; then
            printf '\n===== logs/serve_port_%s.log (tail) =====\n' "$port" >&2
            tail -n 20 "$ROOT_DIR/logs/serve_port_${port}.log" >&2 || true
        fi
    done
    return 1
}

start_servers() {
    local i
    local gpu_id
    local port
    local instance_cache
    local log_path
    local pid
    local serve_extra_args_arr=()
    local serve_cmd=()

    [[ -x "$VLLM_BIN" ]] || die "vLLM executable not found: $VLLM_BIN"
    (( ${#PORTS[@]} == ${#GPUS[@]} )) || die "When serving, the number of ports must equal the number of GPUs."

    mkdir -p "$ROOT_DIR/logs"

    if [[ -n "$SERVE_EXTRA_ARGS" ]]; then
        read -r -a serve_extra_args_arr <<< "$SERVE_EXTRA_ARGS"
    fi

    for ((i = 0; i < ${#GPUS[@]}; i++)); do
        gpu_id="${GPUS[$i]}"
        port="${PORTS[$i]}"
        instance_cache="${CACHE_ROOT%/}/vllm_cache_port_${port}"
        log_path="$ROOT_DIR/logs/serve_port_${port}.log"
        mkdir -p "$instance_cache"

        log "Starting vLLM on GPU $gpu_id -> port $port"
        serve_cmd=(
            "$VLLM_BIN" serve "$MODEL_PATH"
            --served-model-name "$MODEL_NAME"
            --port "$port"
            --max-model-len "$MAX_MODEL_LEN"
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
            --trust-remote-code
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        )
        if [[ -n "$REASONING_PARSER" ]]; then
            serve_cmd+=(--reasoning-parser "$REASONING_PARSER")
        fi
        serve_cmd+=("${serve_extra_args_arr[@]}")
        (
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            export VLLM_CACHE_ROOT="$instance_cache"
            exec "${serve_cmd[@]}"
        ) >"$log_path" 2>&1 &
        pid=$!
        STARTED_PIDS+=("$pid")
    done

    log "Started ${#STARTED_PIDS[@]} vLLM process(es)."
}

cleanup() {
    local exit_code="${1:-0}"
    local pid
    local fd

    trap - EXIT INT TERM

    if (( ${#STARTED_PIDS[@]} > 0 )) && [[ "$KEEP_SERVERS" != "1" ]]; then
        log "Stopping ${#STARTED_PIDS[@]} started vLLM process(es)..."
        for pid in "${STARTED_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        for pid in "${STARTED_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    elif (( ${#STARTED_PIDS[@]} > 0 )); then
        log "Keeping started vLLM process(es) alive as requested."
    fi

    for fd in "${PORT_LOCK_FDS[@]}"; do
        eval "exec ${fd}>&-"
    done

    exit "$exit_code"
}

run_distill() {
    local ports_arg
    local cmd=()
    local rendered=""
    local item

    ports_arg="$(port_list_string)"
    cmd=(
        "$PYTHON_BIN" -m distill
        --config "$CONFIG_PATH"
        --ports "$ports_arg"
        --api-key "$API_KEY"
        --llm-timeout "$LLM_TIMEOUT"
    )
    if [[ -n "$VLLM_LS_COMMAND" ]]; then
        cmd+=(--vllm-ls-command "$VLLM_LS_COMMAND")
    fi
    if [[ -n "$TASK_NAME" ]]; then
        cmd+=(--task "$TASK_NAME")
    fi
    cmd+=("${FORWARD_ARGS[@]}")

    for item in "${cmd[@]}"; do
        rendered+=" $(printf '%q' "$item")"
    done

    log "Running distill with ports: $(port_list_string)"
    log "Backend verification command: $VLLM_LS_COMMAND"
    log "Command:${rendered}"
    "${cmd[@]}"
}

parse_args "$@"

if [[ "$PYTHON_BIN_EXPLICIT" != "1" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
fi
if [[ "$VLLM_BIN_EXPLICIT" != "1" ]]; then
    VLLM_BIN="$VENV_DIR/bin/vllm"
fi
read -r -a GPUS <<< "$GPUS_STR"

[[ ${#GPUS[@]} -gt 0 ]] || die "No GPUs configured."
require_file "$CONFIG_PATH"
require_executable "$PYTHON_BIN"
sanitize_proxy_env
build_ports

trap 'cleanup $?' EXIT INT TERM

if [[ "$SKIP_SERVE" != "1" ]]; then
    reserve_port_locks
    ensure_ports_available
    start_servers
else
    log "Skipping serve stage and reusing existing ports: $(port_list_string)"
fi

log "Waiting for ports to become healthy..."
wait_for_ports_ready
run_distill
