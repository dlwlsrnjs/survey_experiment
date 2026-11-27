#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:-configs/local32b_pilot_imagegen.yaml}
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PROJECT_ROOT/$CONFIG"
fi
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/modern/bin/python}
PORT=${PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 30000))}
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export LOCAL_MODEL_API_KEY=${LOCAL_MODEL_API_KEY:-local}
export PYTHONUNBUFFERED=1

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python environment: $PYTHON_BIN. Run ./bootstrap.sh first." >&2
  exit 2
fi

MODEL_ID=$($PYTHON_BIN -c 'import sys; from survey_exp.config import load_config; print(load_config(sys.argv[1]).model.name)' "$CONFIG")
MODEL_REVISION=$($PYTHON_BIN -c 'import sys; from survey_exp.config import load_config; print(load_config(sys.argv[1]).model.revision or "")' "$CONFIG")
RUN_DIR=$($PYTHON_BIN -c 'import sys; from survey_exp.config import load_config; c=load_config(sys.argv[1]); print(c.output_root / c.experiment_id)' "$CONFIG")
JOB_TAG=${SLURM_JOB_ID:-local-$$}
SERVER_DIR="$PROJECT_ROOT/runs/model-server-$JOB_TAG"
mkdir -p "$SERVER_DIR"
SERVER_LOG="$SERVER_DIR/server.log"

"$PYTHON_BIN" -m survey_exp.local_server \
  --model "$MODEL_ID" --revision "$MODEL_REVISION" \
  --host 127.0.0.1 --port "$PORT" --max-pixels 1048576 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 200 "$SERVER_LOG" >&2
    exit 2
  fi
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m survey_exp.runner --config "$CONFIG"
"$PYTHON_BIN" -m survey_exp.annotation --run-dir "$RUN_DIR" --out-dir "$PROJECT_ROOT/annotations/$(basename "$RUN_DIR")"
"$PYTHON_BIN" -m survey_exp.metrics --run-dir "$RUN_DIR"
