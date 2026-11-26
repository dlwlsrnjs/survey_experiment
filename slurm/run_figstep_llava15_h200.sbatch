#!/usr/bin/env bash
#SBATCH --job-name=figstep-l15-200
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
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
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/HADES/eval/evaluate/inference${PYTHONPATH:+:$PYTHONPATH}"

FIGSTEP_LIMIT=${FIGSTEP_LIMIT:-200}
FIGSTEP_OUTPUT_DIR=${FIGSTEP_OUTPUT_DIR:-$PROJECT_ROOT/runs/figstep_llava15_200_20261126}
LLAVA_PATH="$HF_HOME/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234"
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venvs/legacy-vlm/bin/python" -m survey_exp.figstep_eval \
  --manifest "$PROJECT_ROOT/data/manifests/figstep_safebench_200.jsonl" \
  --output-dir "$FIGSTEP_OUTPUT_DIR" --limit "$FIGSTEP_LIMIT" \
  --llava-path "$LLAVA_PATH"
