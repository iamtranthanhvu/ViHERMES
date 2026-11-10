#!/usr/bin/env bash
# Start or stop the vLLM servers this project needs, in the background.
#
#   scripts/serve.sh base    generator (GEN_MODEL) on GEN_PORT
#   scripts/serve.sh lora    the same generator plus the Conductor LoRA adapter
#   scripts/serve.sh judge   judge (JUDGE_MODEL) on JUDGE_PORT
#   scripts/serve.sh stop    stop every server started by this script
#
# Settings come from configs/host.env; any variable already in the environment wins.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "usage: scripts/serve.sh base|lora|judge|stop" >&2
  exit 2
}

[ $# -eq 1 ] || usage
MODE="$1"
case "$MODE" in base | lora | judge | stop) ;; *) usage ;; esac

# Load defaults without overriding what the caller already exported.
while IFS='=' read -r key value; do
  case "$key" in '' | \#*) continue ;; esac
  [ -n "${!key+x}" ] || export "$key=$value"
done <"$ROOT/configs/host.env"

mkdir -p "$LOG_DIR"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-900}"

alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
ready() { curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }

stop_one() { # name
  local pidfile="$LOG_DIR/$1.pid" pid
  [ -f "$pidfile" ] || return 0
  pid="$(cat "$pidfile")"
  if alive "$pid"; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do alive "$pid" || break; sleep 1; done
    alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile" "$LOG_DIR/$1.mode"
}

if [ "$MODE" = stop ]; then
  stop_one gen
  stop_one judge
  exit 0
fi

if [ ! -x "$VLLM_BIN" ]; then
  echo "serve.sh: vllm not found at $VLLM_BIN (set VLLM_BIN in configs/host.env)" >&2
  exit 1
fi
command -v curl >/dev/null || { echo "serve.sh: curl is required" >&2; exit 1; }
"$VLLM_BIN" --version 2>/dev/null | tail -1 >"$LOG_DIR/vllm.version" || true

start() { # name mode port model revision gpu_util [extra args...]
  local name="$1" mode="$2" port="$3" model="$4" revision="$5" util="$6"
  shift 6
  local pidfile="$LOG_DIR/$name.pid" modefile="$LOG_DIR/$name.mode" log="$LOG_DIR/$name.log" pid
  if [ -f "$pidfile" ] && alive "$(cat "$pidfile")"; then
    if [ "$(cat "$modefile" 2>/dev/null)" = "$mode" ] && ready "$port"; then
      echo "serve.sh: $name already running in $mode mode on port $port"
      return 0
    fi
    echo "serve.sh: restarting $name for $mode mode"
    stop_one "$name"
  fi
  local args=(serve "$model" --port "$port" --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$util")
  [ -z "$revision" ] || args+=(--revision "$revision")
  args+=("$@")
  nohup "$VLLM_BIN" "${args[@]}" >"$log" 2>&1 &
  pid=$!
  echo "$pid" >"$pidfile"
  echo "$mode" >"$modefile"
  echo "serve.sh: starting $name ($model, $mode) on port $port; log: $log"
  local waited=0
  while ! ready "$port"; do
    if ! alive "$pid"; then
      echo "serve.sh: $name exited before it was ready; last log lines:" >&2
      tail -n 20 "$log" >&2 || true
      rm -f "$pidfile" "$modefile"
      return 1
    fi
    if [ "$waited" -ge "$SERVE_TIMEOUT" ]; then
      echo "serve.sh: $name timed out after ${SERVE_TIMEOUT}s; stopping it" >&2
      stop_one "$name"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "serve.sh: $name ready"
}

case "$MODE" in
  base)
    start gen base "$GEN_PORT" "$GEN_MODEL" "$GEN_REVISION" "$GEN_GPU_UTIL"
    ;;
  lora)
    if [ ! -f "$LORA_PATH/adapter_config.json" ]; then
      echo "serve.sh: no LoRA adapter at $LORA_PATH (expected adapter_config.json)" >&2
      exit 1
    fi
    start gen lora "$GEN_PORT" "$GEN_MODEL" "$GEN_REVISION" "$GEN_GPU_UTIL" \
      --enable-lora --max-lora-rank "$LORA_RANK" --lora-modules "$LORA_NAME=$LORA_PATH"
    ;;
  judge)
    extra=()
    [ -z "${JUDGE_QUANT:-}" ] || extra+=(--quantization "$JUDGE_QUANT")
    start judge judge "$JUDGE_PORT" "$JUDGE_MODEL" "${JUDGE_REVISION:-}" "$JUDGE_GPU_UTIL" \
      ${extra[@]+"${extra[@]}"}
    ;;
esac
