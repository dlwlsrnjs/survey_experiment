#!/bin/bash
#SBATCH --job-name=vlmaux32b-full
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/ljk98/survey_experiment
#SBATCH --output=/home/ljk98/survey_experiment/runs/slurm-%x-%j.out
#SBATCH --error=/home/ljk98/survey_experiment/runs/slurm-%x-%j.err
set -euo pipefail

PROJECT=/home/ljk98/survey_experiment
PYTHON_BIN=/home/ljk98/RESORT/.venv_q3vl/bin/python
MODEL_ID=Qwen/Qwen2.5-VL-32B-Instruct
MODEL_REVISION=7cfb30d71a1f4f49a57592323337a4a4727301da
PORT=${PORT:-8000}
export PYTHONPATH="$PROJECT"
export HF_HOME=/home/ljk98/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export LOCAL_MODEL_API_KEY=local
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT/runs/model-server-${SLURM_JOB_ID}"
SERVER_LOG="$PROJECT/runs/model-server-${SLURM_JOB_ID}/server.log"

"$PYTHON_BIN" -m survey_exp.local_server \
  --model "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-pixels 1048576 \
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

"$PYTHON_BIN" -m survey_exp.runner --config configs/local32b_eval_memjack.yaml
"$PYTHON_BIN" -m survey_exp.annotation \
  --run-dir runs/local32b_memjack_eval_20261125 \
  --out-dir annotations/local32b_memjack_eval_20261125
"$PYTHON_BIN" -m survey_exp.metrics --run-dir runs/local32b_memjack_eval_20261125
