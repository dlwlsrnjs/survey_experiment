#!/usr/bin/env bash
#SBATCH --job-name=align-judge
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
: "${VLM_RUN_TAG:?VLM_RUN_TAG is required}"
: "${JUDGE_KIND:?JUDGE_KIND is required}"
: "${JUDGE_MODEL:?JUDGE_MODEL is required}"
: "${JUDGE_REVISION:?JUDGE_REVISION is required}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m survey_exp.open_best_cross_judge_eval \
  --vlm-run "runs/alignment_ablation_cross_vlm_${VLM_RUN_TAG}" \
  --output-dir "runs/alignment_ablation_cross_judge_${VLM_RUN_TAG}_${JUDGE_KIND}" \
  --judge-kind "$JUDGE_KIND" --model "$JUDGE_MODEL" --revision "$JUDGE_REVISION" \
  --expected 4000 --batch-size 16 --max-input-tokens 4096 --max-new-tokens 128
