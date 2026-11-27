#!/usr/bin/env bash
#SBATCH --job-name=ob-q3vl30-600
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
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

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct}
REVISION=${REVISION:-9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
RUN_TAG=${RUN_TAG:-qwen3vl30b_instruct_20261127}
PORT=${PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 30000))}
JOB_TAG=${SLURM_JOB_ID:-local-$$}
SERVER_DIR="$PROJECT_ROOT/runs/model-server-$JOB_TAG"
mkdir -p "$SERVER_DIR"

"$PYTHON_BIN" -m survey_exp.local_server \
  --model "$MODEL_ID" --revision "$REVISION" \
  --host 127.0.0.1 --port "$PORT" --max-pixels 1048576 \
  >"$SERVER_DIR/server.log" 2>&1 &
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
    tail -n 200 "$SERVER_DIR/server.log" >&2
    exit 2
  fi
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

COMMON=(--base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL_ID" --revision "$REVISION" --limit 200)
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m survey_exp.vlm_benchmark_eval \
  --method FigStep --manifest data/manifests/figstep_safebench_200.jsonl \
  --output-dir "runs/open_best_${RUN_TAG}_figstep200" "${COMMON[@]}"
"$PYTHON_BIN" -m survey_exp.vlm_benchmark_eval \
  --method HADES --manifest data/manifests/hades_released_200.jsonl --hades-step 5 \
  --output-dir "runs/open_best_${RUN_TAG}_hades_step5_200" "${COMMON[@]}"
"$PYTHON_BIN" -m survey_exp.vlm_benchmark_eval \
  --method HIMRD --manifest data/manifests/himrd_safebench_full.jsonl \
  --himrd-state-dir runs/himrd_e2e_llava16_200_retry1_20261126/states \
  --output-dir "runs/open_best_${RUN_TAG}_himrd200" "${COMMON[@]}"
