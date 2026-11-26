# 세션 인수인계: 현재 상태와 다음 작업

마지막 점검: **2026-11-26 13:57 KST**

이 문서는 다른 Codex/터미널 세션이 추측 없이 작업을 이어받기 위한 운영 메모다. 연구의 논문 연결은 `docs/SURVEY_TO_EXPERIMENT_KO.md`, 모델급·역할별 평가 기준은 `docs/MODEL_ROLE_CAPABILITY_EVALUATION_KO.md`, 구체적인 실행 설계는 `docs/EXPERIMENT_AND_INTERMEDIATE_ANALYSIS_KO.md`를 먼저 읽는다.

새 머신에서는 `docs/SETUP_KO.md`와 `./bootstrap.sh --profile core`를 사용한다. 원형 충실도용 추가 모델까지 필요하면 `--profile full`을 사용한다. 외부 저장소·데이터·모델 revision과 checksum은 `configs/assets.lock.json`, 이미지 생성 설정은 `configs/image_generators.yaml`이 기준이다.

## 1. 현재 상태

### 완료된 항목

- 고정 입력 manifest 생성 및 SHA-256 기록
- MemJack 공개 train 경험 은행 45,016개 구축
- 역할별 JSON schema, 자동 검사, 전체 trace 로깅 구현
- 블라인드 2인 사람평가 CSV 생성기 구현
- 로컬 32B 파일럿 v1/v2 실행
- 전체 eval의 일반 역할 300건 완료
  - `generation`: 50/50
  - `observation_planning`: 100/100
  - `evaluation_revision`: 150/150
- 중단 시 기존 완료 키를 건너뛰는 resume 로직 구현

### 부분 완료된 전체 실행

대상 실행은 `runs/local32b_memjack_eval_20261125`이며, 원시 `outputs.jsonl` 기준 393/2,100건이 존재한다.

| 역할/조건 | 현재 | 계획 | 남음 |
|---|---:|---:|---:|
| generation | 50 | 50 | 0 |
| observation_planning | 100 | 100 | 0 |
| evaluation_revision | 150 | 150 | 0 |
| experience A | 24 | 450 | 426 |
| experience B | 23 | 450 | 427 |
| experience C | 23 | 450 | 427 |
| experience C_random | 23 | 450 | 427 |
| **합계** | **393** | **2,100** | **1,707** |

Slurm job `59571`은 2026-11-25 16:48 KST경 사용자 취소로 종료됐다. 실행 오류 때문에 중단된 것으로 기록하면 안 된다. 서버 로그상 취소 전 요청은 HTTP 200으로 처리됐고, 실행 후반 93개의 experience 결과도 `outputs.jsonl`에 남아 있다.

`summary.json`은 2026-11-26 인수인계 과정에서 원시 393건을 기준으로 다시 생성했다. 하지만 experience 조건은 repeat 0의 앞부분만 포함하므로 이 수치로 A/B/C/C_random을 비교하거나 최종 결론을 내리면 안 된다.

### Slurm 및 별도 환경 작업

2026-11-26 13:32 KST 기준 상태는 다음과 같다.

- `59780 vlmaux32b-resume`: `FAILED`, exit code `127:0`. Slurm spool 디렉터리를 프로젝트 루트로 잘못 해석한 경로 오류였으며 원시 출력은 증가하지 않았다.
- `59787 vlmaux32b-resume`: 경로 계산을 수정하고 명시적인 `PROJECT_ROOT`로 다시 제출한 작업이다. H200 노드에서 실행 중이며 32B 모델의 18개 checkpoint shard 로딩, `server_ready`, 첫 HTTP 200 응답을 확인했다. 원시 출력도 393건에서 397건으로 증가해 실제 재개가 검증됐다.
- `59810 himrd-4omini-200`: HIMRD 공개 SafeBench 200개 고유 목표의 GPT 역할 600레코드를 `gpt-4o-mini-2024-07-18`로 실행한다. 13:57 KST 기준 RTX6000 노드에서 실행 중이었다. 세부 원칙과 파일럿 결과는 `docs/SOURCE_FIDELITY_EXECUTION_KO.md`에 있다.
- `59726 vlmaux-envs`: `COMPLETED`. modern/legacy VLM 환경 구성이 끝났다.
- `59725 vlmaux-models`: `FAILED`, exit code `2:0`. 여러 모델 파일의 다운로드 진행 기록은 남아 있지만 완료 검증은 통과하지 못한 상태로 취급한다. 로그와 `doctor` 결과를 확인하기 전에는 전체 모델 설치가 끝났다고 간주하지 않는다.

source-fidelity 후속 실행 전에 상태를 반드시 다시 확인한다.

```bash
squeue -u "$USER" -o '%.18i %.20j %.2t %.10M %.6D %R'
tail -n 100 runs/slurm-vlmaux-models-59725.out
tail -n 100 runs/slurm-vlmaux-envs-59726.out
```

## 2. 첫 점검 명령

새 세션은 다음 순서로 현재 상태를 확인한다.

```bash
cd /home/ljk98/survey_experiment

git status --short --branch
git log -1 --oneline

wc -l runs/local32b_memjack_eval_20261125/outputs.jsonl
jq -s 'group_by(.role + ":" + (.condition // "-")) |
  map({key:(.[0].role + ":" + (.[0].condition // "-")), n:length})' \
  runs/local32b_memjack_eval_20261125/outputs.jsonl

squeue -u "$USER" -o '%.18i %.20j %.2t %.10M %.6D %R'
```

## 3. 전체 32B 실행 재개

동일 config와 출력 디렉터리를 사용해야 resume가 작동한다.

```bash
cd /home/ljk98/survey_experiment
sbatch slurm/resume_local32b_20261125.sbatch
```

runner는 `(role, condition, repeat, condition_id)`가 이미 `outputs.jsonl`에 있으면 `resume_skip` 이벤트를 기록하고 건너뛴다. 기존 `outputs.jsonl`, `events.jsonl`, `retrieval_plan.jsonl`을 삭제하거나 다른 디렉터리로 옮기지 않는다.

재개 전에 확인할 사항:

1. 동일한 `configs/local32b_eval_memjack.yaml`을 사용하는가?
2. manifest와 경험 은행 경로가 존재하는가?
3. 모델 revision과 seed가 바뀌지 않았는가?
4. 이미 같은 experiment ID를 실행 중인 Slurm job이 없는가?
5. 남은 호출을 실행할 GPU 시간과 2일 time limit가 충분한가?

## 4. 실행 완료 후 해야 할 일

`completed.json`이 생성되고 출력이 2,100건인지 확인한다.

```bash
wc -l runs/local32b_memjack_eval_20261125/outputs.jsonl
jq . runs/local32b_memjack_eval_20261125/completed.json
python3 -m survey_exp.metrics \
  --run-dir runs/local32b_memjack_eval_20261125
```

그 다음 블라인드 평정지를 생성한다. 기존 파일이 있다면 먼저 내용이 비어 있는 template인지 확인하고, 평가가 시작된 파일은 덮어쓰지 않는다.

```bash
python3 -m survey_exp.annotation \
  --run-dir runs/local32b_memjack_eval_20261125 \
  --out-dir annotations/local32b_memjack_eval_20261125
```

두 평가자의 독립 평정 후:

1. 원본 `annotations_rater_a.csv`, `annotations_rater_b.csv` 보존
2. 불일치 조정본을 별도 CSV로 작성
3. `metrics.py`에 Cohen's kappa/weighted kappa 추가
4. A/B/C/C_random의 짝지어진 대비 통계 추가
5. `survey_exp.qualitative`로 역할별 2건의 대조 사례 선택
6. `docs/PRELIMINARY_LOCAL32B_RESULTS_KO.md`를 최종 결과 문서와 분리 유지

## 5. 알려진 분석·구현 주의점

- `target_query_cap: 3`은 config에 있으나 현재 auxiliary replay runner에서 실제 표적 모델 질의를 수행하거나 강제하지 않는다. “표적 질의 상한을 지킨 end-to-end 공격 평가”라고 주장하면 안 된다.
- 전체 Slurm 명령은 `--target-responses`를 넘기지 않는다. 따라서 `evaluation_revision`의 표적 응답 기반 `understood/fulfilled` 분석은 현재 실행 범위가 아니다.
- 현재 `metrics.py`는 experience 조건의 짝지어진 효과 검정과 평가자 간 신뢰도를 아직 계산하지 않는다.
- `generation`의 100% 거부, experience 부분 실행의 100% 거부는 이 모델·프롬프트·부분 표본의 관찰값일 뿐 일반적인 공격 성공률이 아니다.
- `SD`와 `SD_TYPO`는 같은 목표를 공유하므로 목표 단위로 짝지어 분석한다.
- 거부를 품질 0으로 넣지 않는다. API 오류, 형식 오류, 거부, 후보 거부, 사람 품질을 분리한다.
- `runs/`, `annotations/`, 원시 데이터와 49MB 경험 은행은 `.gitignore` 대상이다. Git에는 코드·config·고정 manifest·문서만 올린다.

## 6. 핵심 파일 지도

| 목적 | 파일 |
|---|---|
| 프로젝트 시작점 | `README.md` |
| 서베이–실험 연결과 연구 의도 | `docs/SURVEY_TO_EXPERIMENT_KO.md` |
| 모델급·의도 노출·역할별 지표·비용 | `docs/MODEL_ROLE_CAPABILITY_EVALUATION_KO.md` |
| 연구 설계·중간작업 분석 | `docs/EXPERIMENT_AND_INTERMEDIATE_ANALYSIS_KO.md` |
| 원형 보존/변형 경계 | `docs/PROTOCOL_ADHERENCE_KO.md` |
| 공개 코드 대응 | `docs/CODE_ANALYSIS_KO.md` |
| 역할·모델 실행환경 감사 | `docs/MODEL_ROLE_AND_ENVIRONMENT_AUDIT_KO.md` |
| 데이터/출처 근거 | `docs/SOURCE_LEDGER_KO.md` |
| 현재 부분 결과 | `docs/PRELIMINARY_LOCAL32B_RESULTS_KO.md` |
| 전체 config | `configs/local32b_eval_memjack.yaml` |
| 실행·resume | `survey_exp/runner.py` |
| 역할 prompt/schema | `survey_exp/roles.py` |
| 자동 집계 | `survey_exp/metrics.py` |
| 블라인드 평정지 | `survey_exp/annotation.py` |
| 질적 사례 선택 | `survey_exp/qualitative.py` |
| 기존 2026-11-25 실행 재개 | `slurm/resume_local32b_20261125.sbatch` |
| 신규 이미지 생성 전체 실행 | `configs/local32b_eval_imagegen.yaml`, `slurm/run_local32b_full.sbatch` |
| 새 머신 부트스트랩 | `bootstrap.sh`, `docs/SETUP_KO.md` |
| 자산 revision/checksum | `configs/assets.lock.json` |
| 이미지 생성 프로필 | `configs/image_generators.yaml` |

## 7. 새 세션에 바로 붙여 넣을 요청문

> `/home/ljk98/survey_experiment`의 작업을 이어서 진행해줘. 먼저 `AGENTS.md`, `docs/SURVEY_TO_EXPERIMENT_KO.md`, `docs/MODEL_ROLE_CAPABILITY_EVALUATION_KO.md`, `docs/EXPERIMENT_AND_INTERMEDIATE_ANALYSIS_KO.md`, `docs/SESSION_HANDOFF_KO.md`를 읽고 Git/Slurm/원시 출력 건수를 확인해. 기존 실행 파일을 삭제하거나 덮어쓰지 말고, 문서의 완료 조건과 분석 경계를 지켜서 남은 작업을 진행해줘.

## 8. 2026-11-26 HIMRD H200 종단간 상태

- GPT-4o mini 역할 출력은 `runs/himrd_gpt4omini_safebench200_20261126`에 600/600건 완료됐다.
- `survey_exp/himrd_end_to_end.py`와 `slurm/run_himrd_end_to_end_h200.sbatch`에 SD3 → LLaVA-v1.6-Mistral-7B → HarmBench → 조건부 prefix/suffix 반복 경로가 구현돼 있다.
- 기존 장면 설명 200건은 직접 재사용한다. 기존 prefix/suffix는 실제 VLM 응답 전 결과이므로 첫 반복 후보로만 사용한다.
- LLaVA와 `cais/HarmBench-Mistral-7b-val-cls` 가중치는 `$HOME/.cache/huggingface`에 완전하다.
- SD3 Medium은 gated 접근이 승인되지 않아 HTTP 403이다. snapshot에 실제 가중치가 없으므로 아직 H200 smoke/full job을 제출하지 않았다.
- Hugging Face 모델 페이지에서 `stabilityai/stable-diffusion-3-medium-diffusers` 접근을 승인받은 뒤 SD3를 고정 revision `ea42f8cef0f178587cf766dc8129abd379c90671`로 내려받고, 1건 smoke가 성공하면 200건 작업을 제출한다.
