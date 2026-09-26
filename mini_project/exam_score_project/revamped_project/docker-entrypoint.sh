#!/bin/bash
set -uo pipefail

API_PID=""
UI_PID=""

stop_children() {
    if [[ -n "$API_PID" ]]; then kill "$API_PID" 2>/dev/null || true; fi
    if [[ -n "$UI_PID" ]]; then kill "$UI_PID" 2>/dev/null || true; fi
    wait "$API_PID" 2>/dev/null || true
    wait "$UI_PID" 2>/dev/null || true
}

trap stop_children SIGTERM SIGINT

echo "[entrypoint] starting FastAPI on port 8000"
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "[entrypoint] waiting for the model-backed API"
API_READY=0
for _ in $(seq 1 45); do
    if python -c "import json, urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)); raise SystemExit(0 if d.get('model_loaded') else 1)" 2>/dev/null; then
        API_READY=1
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "[entrypoint] API exited before becoming ready"
        wait "$API_PID" || EXIT_CODE=$?
        stop_children
        exit "${EXIT_CODE:-1}"
    fi
    sleep 1
done

if [[ "$API_READY" -ne 1 ]]; then
    echo "[entrypoint] API did not become ready within 45 seconds"
    stop_children
    exit 1
fi

echo "[entrypoint] starting Streamlit on port 8501"
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 &
UI_PID=$!

# Exit when either service exits, then shut down its sibling. `wait -n` can return a
# non-zero status when a child crashes, so capture that status without errexit semantics.
EXIT_CODE=0
wait -n "$API_PID" "$UI_PID" || EXIT_CODE=$?
stop_children
exit "$EXIT_CODE"
