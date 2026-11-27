#!/usr/bin/env bash
#SBATCH --job-name=ob-image200
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
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

: "${MODEL_ID:?MODEL_ID is required}"
: "${REVISION:?REVISION is required}"
: "${RUN_TAG:?RUN_TAG is required}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m survey_exp.open_best_image_eval \
  --manifest data/manifests/open_best_image_prompts_200.jsonl \
  --output-dir "runs/open_best_${RUN_TAG}_image200" \
  --model "$MODEL_ID" --revision "$REVISION" \
  --limit 200 --width 1024 --height 1024 --steps 30 --guidance 4.0 --seed 20261127
