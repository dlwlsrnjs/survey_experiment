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

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROFILE=${PROFILE:-core}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PROJECT_ROOT/.cache/pip}
bash "$PROJECT_ROOT/scripts/setup_model_environments.sh" "$PROFILE"
