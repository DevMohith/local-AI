set -euo pipefail
MODE="${1:-app}"
echo "[DocAI] mode: $MODE"

if [ "$MODE" = "llama" ]; then
    MODEL_PATH="${MODEL_PATH:?need MODEL_PATH}"
    MODEL_PORT="${MODEL_PORT:?need MODEL_PORT}"
    MODEL_URL="${MODEL_URL:?need MODEL_URL}"
    CONTEXT_SIZE="${CONTEXT_SIZE:-4096}"
    CPU_THREADS="${CPU_THREADS:-4}"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "=============================================="
        echo " Downloading model (first time only, ~5-15min)"
        echo " $MODEL_PATH"
        echo "=============================================="
        mkdir -p "$(dirname "$MODEL_PATH")"
        wget --progress=bar:force:noscroll \
             --retry-connrefused --waitretry=10 --tries=5 --continue \
             -O "$MODEL_PATH" "$MODEL_URL"
        echo "Done: $(du -sh "$MODEL_PATH" | cut -f1)"
    else
        echo "Model ready: $(du -sh "$MODEL_PATH" | cut -f1)"
    fi

    exec llama-server \
        --model     "$MODEL_PATH" \
        --port      "$MODEL_PORT" \
        --host      0.0.0.0 \
        --ctx-size  "$CONTEXT_SIZE" \
        --threads   "$CPU_THREADS" \
        --parallel  2 \
        --cont-batching \
        --log-disable


elif [ "$MODE" = "app" ]; then
    PORT="${PORT:-8080}"
    exec python -m uvicorn backend.main:app \
        --host 0.0.0.0 --port "$PORT" --workers 2 --no-access-log
 
else
    echo "Unknown MODE: $MODE"
    exit 1
fi