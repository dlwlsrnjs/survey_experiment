#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

LLM_MODEL=huihui-ai/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated
LLM_REV=e2f73ec7e99ee316beb8069ca90e4c3cbef8aa0f
VLM_MODEL=huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated
VLM_REV=936d030f188a041b9dd5b2c3ffe144c2453fcab6

job_id() { awk '{print $NF}'; }

DL_LLM=$(sbatch --job-name=align-dl-llm \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",REPO_ID="$LLM_MODEL",REVISION="$LLM_REV" \
  slurm/download_open_best_candidate.sbatch | job_id)
DL_VLM=$(sbatch --job-name=align-dl-vlm \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",REPO_ID="$VLM_MODEL",REVISION="$VLM_REV" \
  slurm/download_open_best_candidate.sbatch | job_id)

LLM_JOB=$(sbatch --dependency="afterok:$DL_LLM" --job-name=align-llm \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MODEL_ID="$LLM_MODEL",REVISION="$LLM_REV",RUN_TAG=qwen3_30ba3b_abliterated_20261128,SERVER_KIND=text \
  slurm/run_open_best_llm_roles_h200.sbatch | job_id)

declare -a IMAGE_JOBS=()
submit_image() {
  local job_name=$1 model=$2 revision=$3 tag=$4 python_bin=$5
  local submitted
  submitted=$(sbatch --dependency="afterok:$LLM_JOB" --job-name="$job_name" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",PYTHON_BIN="$python_bin",MODEL_ID="$model",REVISION="$revision",RUN_TAG="$tag" \
    slurm/run_alignment_ablation_images_h200.sbatch | job_id)
  IMAGE_JOBS+=("$submitted")
}

submit_image align-img-q2512 Qwen/Qwen-Image-2512 25468b98e3276ca6700de15c6628e51b7de54a26 qwen_image2512_20261128 "$PROJECT_ROOT/.venvs/modern/bin/python"
submit_image align-img-q21 Qwen/Qwen-Image-2.1 d26bb61231c349cf6b7896fa83353113880e1ba3 qwen_image21_20261128 "$PROJECT_ROOT/.venvs/qwen21/bin/python"
submit_image align-img-flux black-forest-labs/FLUX.1-dev 45d95c497444bbaebe766e1528792fc9158c0c3a flux1_dev_20261128 "$PROJECT_ROOT/.venvs/modern/bin/python"
submit_image align-img-pix PixArt-alpha/PixArt-Sigma-XL-2-1024-MS e102b3591cc82e97071b8b4cb90d834d0c487207 pixart_sigma_20261128 "$PROJECT_ROOT/.venvs/modern/bin/python"
submit_image align-img-sd3 stabilityai/stable-diffusion-3-medium-diffusers ea42f8cef0f178587cf766dc8129abd379c90671 sd3_medium_20261128 "$PROJECT_ROOT/.venvs/modern/bin/python"

IMAGE_DEP=$(IFS=:; echo "${IMAGE_JOBS[*]}")
VLM_ALIGNED=$(sbatch --dependency="afterok:$IMAGE_DEP" --job-name=align-vlm-base \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MODEL_ID=Qwen/Qwen3-VL-30B-A3B-Instruct,REVISION=9c4b90e1e4ba969fd3b5378b57d966d725f1b86c,RUN_TAG=aligned_qwen3vl30ba3b_20261128 \
  slurm/run_alignment_ablation_vlm_h200.sbatch | job_id)
VLM_ABLITERATED=$(sbatch --dependency="afterok:$IMAGE_DEP:$DL_VLM" --job-name=align-vlm-abl \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MODEL_ID="$VLM_MODEL",REVISION="$VLM_REV",RUN_TAG=abliterated_qwen3vl30ba3b_20261128 \
  slurm/run_alignment_ablation_vlm_h200.sbatch | job_id)

submit_judges() {
  local dependency=$1 tag=$2 prefix=$3
  sbatch --dependency="afterok:$dependency" --job-name="${prefix}-hb" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",VLM_RUN_TAG="$tag",JUDGE_KIND=harmbench,JUDGE_MODEL=cais/HarmBench-Mistral-7b-val-cls,JUDGE_REVISION=51182c7cdaf9b2e5f05b745b81b047b67e7384a1 \
    slurm/run_alignment_ablation_judge_h200.sbatch | job_id
  sbatch --dependency="afterok:$dependency" --job-name="${prefix}-qg" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",VLM_RUN_TAG="$tag",JUDGE_KIND=qwen_guard,JUDGE_MODEL=Qwen/Qwen3Guard-Gen-8B,JUDGE_REVISION=4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb \
    slurm/run_alignment_ablation_judge_h200.sbatch | job_id
}

JUDGE_BASE=$(submit_judges "$VLM_ALIGNED" aligned_qwen3vl30ba3b_20261128 align-j-base | paste -sd, -)
JUDGE_ABL=$(submit_judges "$VLM_ABLITERATED" abliterated_qwen3vl30ba3b_20261128 align-j-abl | paste -sd, -)

printf 'download_llm=%s\ndownload_vlm=%s\nllm=%s\nimages=%s\nvlm_aligned=%s\nvlm_abliterated=%s\njudges_aligned=%s\njudges_abliterated=%s\n' \
  "$DL_LLM" "$DL_VLM" "$LLM_JOB" "$IMAGE_DEP" "$VLM_ALIGNED" "$VLM_ABLITERATED" "$JUDGE_BASE" "$JUDGE_ABL"
