#!/usr/bin/env bash
#SBATCH --job-name=ob-cross-vlm
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=5-00:00:00
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

PORT=${PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 30000))}
JOB_TAG=${SLURM_JOB_ID:-local-$$}
SERVER_DIR="$PROJECT_ROOT/runs/model-server-$JOB_TAG"
mkdir -p "$SERVER_DIR"
"$PYTHON_BIN" -m survey_exp.local_server \
  --model "$MODEL_ID" --revision "$REVISION" --host 127.0.0.1 --port "$PORT" --max-pixels 1048576 \
  >"$SERVER_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 200 "$SERVER_DIR/server.log" >&2; exit 2; fi
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m survey_exp.open_best_cross_vlm_eval \
  --image-run qwen2512=runs/open_best_cross_images_qwen_image2512_20261127 \
  --image-run qwen21=runs/open_best_cross_images_qwen_image21_20261127 \
  --image-run flux=runs/open_best_cross_images_flux1_dev_20261127 \
  --image-run pixart=runs/open_best_cross_images_pixart_sigma_20261127 \
  --image-run sd3med=runs/open_best_cross_images_sd3_medium_20261127 \
  --per-image-model 2000 \
  --output-dir "runs/open_best_cross_vlm_${RUN_TAG}" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL_ID" --revision "$REVISION" \
  --max-tokens 1024 --timeout-seconds 900 --max-error-rate 0.01
