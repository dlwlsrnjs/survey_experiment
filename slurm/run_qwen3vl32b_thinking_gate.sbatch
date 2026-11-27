#!/usr/bin/env bash
#SBATCH --job-name=vlmaux-q3v32b-gate
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/pyproject.toml" || ! -x "$PROJECT_ROOT/scripts/run_local32b.sh" ]]; then
  echo "Cannot resolve project root. Submit from the repository root or export PROJECT_ROOT." >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
export PROJECT_ROOT
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}
CONFIG=${CONFIG:-configs/qwen3vl32b_thinking_gate.yaml}
bash "$PROJECT_ROOT/scripts/run_local32b.sh" "$CONFIG"
