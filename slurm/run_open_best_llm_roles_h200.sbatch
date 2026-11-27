#!/usr/bin/env bash
#SBATCH --job-name=ob-llm-roles
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  echo "Cannot resolve project root." >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/modern/bin/python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LOCAL_MODEL_API_KEY=${LOCAL_MODEL_API_KEY:-local}

: "${MODEL_ID:?MODEL_ID is required}"
: "${REVISION:?REVISION is required}"
: "${RUN_TAG:?RUN_TAG is required}"
SERVER_KIND=${SERVER_KIND:-text}
PORT=${PORT:-8000}
JOB_TAG=${SLURM_JOB_ID:-local-$$}
SERVER_DIR="$PROJECT_ROOT/runs/model-server-$JOB_TAG"
mkdir -p "$SERVER_DIR"

if [[ "$SERVER_KIND" == vlm ]]; then
  SERVER_MODULE=survey_exp.local_server
  SERVER_EXTRA=(--max-pixels 1048576)
else
  SERVER_MODULE=survey_exp.local_text_server
  SERVER_EXTRA=()
fi
"$PYTHON_BIN" -m "$SERVER_MODULE" --model "$MODEL_ID" --revision "$REVISION" \
  --host 127.0.0.1 --port "$PORT" "${SERVER_EXTRA[@]}" \
  >"$SERVER_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 200 "$SERVER_DIR/server.log" >&2
    exit 2
  fi
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m survey_exp.source_fidelity_runner \
  --manifest data/manifests/himrd_safebench_full.jsonl \
  --output-dir "runs/open_best_${RUN_TAG}_llm_roles200" \
  --repeats 1 --limit 200 --methods HADES HIMRD --no-image-generation \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" --revision "$REVISION" \
  --model-substitution "open_best text-role comparison; not source-fidelity model identity"
