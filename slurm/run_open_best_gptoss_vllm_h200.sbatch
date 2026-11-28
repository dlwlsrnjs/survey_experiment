#!/usr/bin/env bash
#SBATCH --job-name=ob-gptoss-vllm
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
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
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/gptoss-vllm/bin/python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LOCAL_MODEL_API_KEY=${LOCAL_MODEL_API_KEY:-local}

MODEL_ID=openai/gpt-oss-120b
REVISION=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
PORT=${PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 30000))}
JOB_TAG=${SLURM_JOB_ID:-local-$$}
SERVER_DIR="$PROJECT_ROOT/runs/model-server-$JOB_TAG"
mkdir -p "$SERVER_DIR"

"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" --revision "$REVISION" \
  --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$PORT" \
  --dtype auto --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  >"$SERVER_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 360); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 300 "$SERVER_DIR/server.log" >&2
    exit 2
  fi
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venvs/modern/bin/python" -m survey_exp.source_fidelity_runner \
  --manifest data/manifests/himrd_safebench_full.jsonl \
  --output-dir runs/open_best_gpt_oss120b_retry2_20261128_llm_roles200 \
  --repeats 1 --limit 200 --methods HADES HIMRD --no-image-generation \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" --revision "$REVISION" \
  --model-substitution "open_best text-role comparison via official gpt-oss vLLM runtime; not source-fidelity model identity"
