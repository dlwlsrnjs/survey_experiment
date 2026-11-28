#!/usr/bin/env bash
#SBATCH --job-name=ob-cross-img
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
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
: "${MODEL_ID:?MODEL_ID is required}"
: "${REVISION:?REVISION is required}"
: "${RUN_TAG:?RUN_TAG is required}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m survey_exp.open_best_cross_image_eval \
  --llm-run gptoss=runs/open_best_gpt_oss120b_retry2_20261128_llm_roles200 \
  --llm-run qwen3a3b=runs/open_best_qwen3_30ba3b_retry1_20261127_llm_roles200 \
  --llm-run mistral31=runs/open_best_mistral_small31_24b_retry1_20261127_llm_roles200 \
  --llm-run gemma3=runs/open_best_gemma3_27b_retry1_20261127_llm_roles200 \
  --llm-run qwen3_32b=runs/open_best_qwen3_32b_20261127_llm_roles200 \
  --per-method 200 \
  --output-dir "runs/open_best_cross_images_${RUN_TAG}" \
  --model "$MODEL_ID" --revision "$REVISION" \
  --width 1024 --height 1024 --steps 30 --guidance 4.0 --seed 20261127
