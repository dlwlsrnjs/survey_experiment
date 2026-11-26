#!/usr/bin/env bash
#SBATCH --job-name=vlmaux-source
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PROJECT_ROOT
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}
bash "$PROJECT_ROOT/scripts/run_source_fidelity.sh"
