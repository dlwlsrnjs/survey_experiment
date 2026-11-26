#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR=${1:-$PROJECT_ROOT/runs/source_fidelity_$(date -u +%Y%m%d)}
REPEATS=${REPEATS:-3}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/modern/bin/python}
PORT=${PORT:-8000}
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export LOCAL_MODEL_API_KEY=${LOCAL_MODEL_API_KEY:-local}
export PYTHONUNBUFFERED=1

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python environment: $PYTHON_BIN. Run ./bootstrap.sh --profile full first." >&2
  exit 2
fi

JOB_TAG=${SLURM_JOB_ID:-local-source-$$}
SERVER_DIR="$PROJECT_ROOT/runs/model-server-$JOB_TAG"
mkdir -p "$SERVER_DIR"
"$PYTHON_BIN" -m survey_exp.local_server \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --revision 7cfb30d71a1f4f49a57592323337a4a4727301da \
  --host 127.0.0.1 --port "$PORT" --max-pixels 1048576 \
  >"$SERVER_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 200 "$SERVER_DIR/server.log" >&2; exit 2; fi
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m survey_exp.source_fidelity_runner \
  --manifest "$PROJECT_ROOT/data/manifests/fixed_inputs.jsonl" \
  --output-dir "$OUTPUT_DIR" --repeats "$REPEATS" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --image-generator-config "$PROJECT_ROOT/configs/image_generators.yaml"
