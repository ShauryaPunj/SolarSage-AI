#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="${0:A:h}"
cd "$ROOT_DIR"

PYTHON_BIN="python3.13"
UVICORN_PORT=8000

if [[ ! -d ".venv" ]]; then
  echo "[setup] Creating venv (.venv)..."
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -V

if [[ -f "requirements.txt" ]]; then
  echo "[setup] Installing Python deps..."
  pip install --upgrade pip
  pip install -r requirements.txt
fi

echo "[api] Starting FastAPI (uvicorn) on 127.0.0.1:${UVICORN_PORT} ..."
pkill -f "uvicorn.*service.app:app.*${UVICORN_PORT}" >/dev/null 2>&1 || true
( uvicorn service.app:app --host 127.0.0.1 --port ${UVICORN_PORT} ) &
API_PID=$!

cleanup() {
  echo "[cleanup] Stopping API..."
  kill ${API_PID} >/dev/null 2>&1 || true
  pkill -f "uvicorn.*service.app:app.*${UVICORN_PORT}" >/dev/null 2 &>/dev/null || true
}
trap cleanup EXIT

echo -n "[api] Waiting for /health ..."
for i in {1..60}; do
  if curl -sS --max-time 1 http://127.0.0.1:${UVICORN_PORT}/health | grep -q '"status":"ok"'; then
    echo " OK"
    break
  fi
  echo -n "."
  sleep 1
done

curl -sSf http://127.0.0.1:${UVICORN_PORT}/health >/dev/null
echo "[api] Health OK."

if [[ -f "package-lock.json" || -f "node_modules/electron/package.json" ]]; then
  echo "[electron] Launching via npm run start ..."
  npm run start
else
  echo "[electron] Launching via npx electron . ..."
  npx electron .
fi

echo "[done] Electron exited."
