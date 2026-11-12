#!/usr/bin/env bash
# One command to run the chat on the real B3 pipeline (Qwen3.5-4B + Conductor LoRA, bge-m3).
#
#   scripts/chat.sh         set up what is missing, start vLLM and the chat, print the address
#   scripts/chat.sh stop    stop both
#
# Every step is skipped when it is already done, so running it again is safe.
# Settings come from configs/host.env; any variable already in the environment wins.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

while IFS='=' read -r key value; do
  case "$key" in '' | \#*) continue ;; esac
  [ -n "${!key+x}" ] || export "$key=$value"
done <configs/host.env

MAKE_BIN="${MAKE_BIN:-make}"
SERVE_SH="${SERVE_SH:-$ROOT/scripts/serve.sh}"
VH_BIN="${VH_BIN:-$ROOT/.venv/bin/vihermes}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
OUT="${OUT:-outputs}"
mkdir -p "$LOG_DIR"
PIDFILE="$LOG_DIR/chat.pid"
HEALTH="http://127.0.0.1:$CHAT_PORT/api/health"

alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
healthy() { curl -sf "$HEALTH" >/dev/null 2>&1; }
die() { echo "chat.sh: $*" >&2; exit 1; }

stop_chat() {
  if [ -f "$PIDFILE" ]; then
    local pid
    pid="$(cat "$PIDFILE")"
    if alive "$pid"; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 15); do alive "$pid" || break; sleep 1; done
      alive "$pid" && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
}

case "${1:-start}" in
  stop)
    stop_chat
    "$SERVE_SH" stop
    echo "chat.sh: stopped"
    exit 0
    ;;
  start) ;;
  *) die "usage: scripts/chat.sh [start|stop]" ;;
esac

# 1. The chat has no offline mode, so without a GPU there is nothing to run.
command -v "$NVIDIA_SMI" >/dev/null || die "no GPU found ($NVIDIA_SMI missing); the chat needs a GPU host"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || die "python3 >= 3.10 is required"

# 2. Packages, vLLM, corpus, split check, and the dense index of the base bge-m3.
"$MAKE_BIN" setup
"$MAKE_BIN" "$OUT/dense-b1.npz"

# 3. The adapter. The token is read once, hidden, and handed only to the download command.
if [ ! -f "$LORA_PATH/adapter_config.json" ]; then
  token="${HF_TOKEN:-}"
  if [ -z "$token" ]; then
    read -rsp "Hugging Face token (READ access to $HUB_LORA_REPO): " token
    echo
  fi
  [ -n "$token" ] || die "no token given"
  HF_TOKEN="$token" "$VH_BIN" fetch --repo "$HUB_LORA_REPO" \
    --revision "$CHAT_LORA_REVISION" --dest "$LORA_PATH" >/dev/null
  unset token
fi
unset HF_TOKEN

# 4. One vLLM server: the base model plus the adapter.
"$SERVE_SH" lora

# 5. The chat.
if [ -f "$PIDFILE" ] && alive "$(cat "$PIDFILE")" && healthy; then
  echo "chat.sh: chat already running"
else
  stop_chat
  args=(chat --db "$OUT/vihermes.db" --index "$OUT/dense-b1.npz"
    --embedding sentence-transformers --embedding-model "$EMBED_BASE"
    --max-seq-length "$EMBED_MAX_SEQ_LENGTH" --dense-weight 0.5 -k 5 --no-thinking
    --model "$GEN_MODEL" --base-url "http://127.0.0.1:$GEN_PORT/v1"
    --conductor-model "$LORA_NAME" --host 127.0.0.1 --port "$CHAT_PORT")
  [ -z "${EMBED_REVISION:-}" ] || args+=(--embedding-revision "$EMBED_REVISION")
  nohup "$VH_BIN" "${args[@]}" >"$LOG_DIR/chat.log" 2>&1 &
  pid=$!
  echo "$pid" >"$PIDFILE"
  waited=0
  until healthy; do
    if ! alive "$pid"; then
      echo "chat.sh: the chat exited before it was ready; last log lines:" >&2
      tail -n 20 "$LOG_DIR/chat.log" >&2 || true
      rm -f "$PIDFILE"
      exit 1
    fi
    [ "$waited" -lt "$CHAT_TIMEOUT" ] || { stop_chat; die "the chat did not become ready in ${CHAT_TIMEOUT}s"; }
    sleep 2
    waited=$((waited + 2))
  done
fi

echo
echo "Chat is ready (B3 pipeline). On your own machine run:"
echo "  ssh -N -L $CHAT_PORT:127.0.0.1:$CHAT_PORT -p ${VAST_TCP_PORT_22:-<PORT>} root@${PUBLIC_IPADDR:-<IP>}"
echo "then open http://127.0.0.1:$CHAT_PORT"
echo "Stop with: bash scripts/chat.sh stop    Log: $LOG_DIR/chat.log"
