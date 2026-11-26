#!/usr/bin/env bash
#SBATCH --job-name=vlmaux-envs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/pyproject.toml" || ! -x "$PROJECT_ROOT/scripts/setup_model_environments.sh" ]]; then
  echo "Cannot resolve project root. Submit from the repository root or export PROJECT_ROOT." >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
PROFILE=${PROFILE:-core}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PROJECT_ROOT/.cache/pip}
bash "$PROJECT_ROOT/scripts/setup_model_environments.sh" "$PROFILE"
