# 세션 인수인계: 현재 상태와 다음 작업

마지막 점검: **2026-11-27 KST**

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
- FigStep LLaVA-1.5-7B 200/200 응답 생성 완료, 사람평가 미완료
- HADES LLaVA-1.5-7B·Beaver-Dam-7B 200목표×6단계 1,200/1,200 완료
- HIMRD GPT-4o mini → SD3 Medium → LLaVA-v1.6-Mistral-7B → HarmBench 200/200 완료
- 단계별 VLM·LLM·이미지 생성기·judge 라우팅을 `docs/PIPELINE_MODEL_ROUTING_KO.md`에 동결
- 모든 기법에 적용할 공개 VLM·LLM·이미지 생성기 최적 조합 탐색을 `docs/OPEN_BEST_MODEL_SEARCH_KO.md`와 `configs/open_best_candidates.yaml`에 분리

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
| 단계별 모델·모달리티·폐쇄형 API 선택 | `docs/PIPELINE_MODEL_ROUTING_KO.md` |
| 공개 모델 최적 조합 후보와 선정 절차 | `docs/OPEN_BEST_MODEL_SEARCH_KO.md`, `configs/open_best_candidates.yaml` |
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
- SD3 Medium gated 접근이 승인됐고 고정 revision `ea42f8cef0f178587cf766dc8129abd379c90671`의 38개 파일, 약 15GB 다운로드와 오프라인 해석 검증이 완료됐다.
- 최초 smoke `59824`는 SentencePiece/Protobuf 불일치로 실패했다. SentencePiece 0.2.0 고정 후 재제출한 smoke는 `59834`, 성공 후 자동 시작할 200건 본 작업은 `59835`다.
- smoke 출력은 `runs/himrd_e2e_llava16_smoke1_20261126`, 본 출력은 `runs/himrd_e2e_llava16_200_20261126`이다.

## 9. 2026-11-26 다른 원형 충실도 작업

- HADES 공식 데이터 revision `8c7388c566c8844a0732a2ad2156b72ae00c92c8`의 4,500행을 내려받았다. 5개 scenario에서 40개 목표씩, 각 0–5단계 이미지를 포함한 1,200행 manifest는 `data/manifests/hades_released_200.jsonl`이다.
- HADES 최초 smoke `59826`은 오프라인 CLIP 경로 해석 실패로 종료됐다. LLaVA config에 고정 local CLIP snapshot을 주입하는 overlay를 적용한 새 smoke는 `59836`, 성공 후 200목표×6단계(1,200행) 본 작업은 `59837`이다.
- FigStep 공식 SafeBench에서 범주별 20개씩 뽑은 200행 manifest는 `data/manifests/figstep_safebench_200.jsonl`이다. 같은 local CLIP overlay를 적용한 smoke는 `59838`, 성공 후 본 작업은 `59839`다.
- FigStep 주 평가는 논문처럼 사람 판정이다. `completed.json`의 응답 완료와 ASR 확정을 혼동하지 않는다.
- MM-Plan VE-Safety는 현재 Hugging Face 계정에서 `test.parquet` 다운로드가 HTTP 403이다. DUA 접근 승인이 필요하다.
- MemJack 전체 원형 종단간 작업은 아직 제출하지 않았다. 논문 victim `Qwen3-VL-Plus`는 폐쇄형이고, 고정 공식 revision에는 `main.py`가 요구하는 `configs/safety_policy.txt`가 없다. 임의 정책을 원형으로 둔갑시키지 말고 저자 공개/확보 뒤 실행한다. 공개 local backend의 Qwen3-VL-8B 조건은 별도 대체 트랙이다. 32B 역할 재생은 2026-11-26 07:28 UTC에 2,100/2,100건으로 완료됐다.

## 10. 2026-11-27 공개 모델 조합 탐색

- 후보 registry는 `configs/open_best_candidates.yaml`이다. VLM 6개, LLM 5개, 이미지 생성기 5개를 기록했고 주 factorial에 들어갈 정확한 5/5/5 후보를 명시했다.
- `configs/open_best_combinations.jsonl`에는 revision을 포함한 125개 조합이 있으며 sidecar SHA-256으로 고정했다.
- 본선은 모든 125개 조합을 FigStep, HADES, HIMRD, MM-Plan, MemJack의 고유 목표 200개씩 실행한다. 1차 계획량은 125,000 조합–기법–목표 평가다.
- 데이터 freezer는 `scripts/freeze_open_best_manifests.py`다. 다섯 기법 중 하나라도 실제 benchmark 고유 group 200개를 제공하지 못하면 실패한다. 같은 200개를 조합 선택에 사용하므로 외부 독립 test로 표현하지 않는다.
- 현재 FigStep·HADES·HIMRD는 200개 이상 실제 benchmark 목표가 준비됐다. MM-Plan은 VE-Safety DUA 다운로드 403 때문에, MemJack은 공개 궤적의 원본 이미지 경로가 배포 환경의 절대경로이고 local victim 원형 파일이 불완전하기 때문에 본선 manifest를 아직 동결하지 않았다.
- Slurm `59949`는 `Qwen/Qwen3-VL-30B-A3B-Instruct` 고정 revision 다운로드, `59950`은 성공 의존 H200 10건 건강검사, `59951`은 자동 gate다. 이 10건은 모델 적재·형식·거부 상태만 확인하며 순위 데이터가 아니다.
- 본선 제출 전 `configs/open_best_protocol.yaml`의 데이터 pool 5개, 이미지 존재, matrix checksum, candidate cache, gated 라이선스 접근을 모두 확인한다. 일부 기법만으로 전역 순위를 내지 않는다.

### 먼저 실행 가능한 부분 비교

- `59949` 다운로드와 `59950` 10건 건강검사는 완료됐다. 오류 0, 형식 유효 10/10, 길이 종료 0이지만 generation 역할은 10/10 구조화된 거부여서 generation gate를 통과하지 못했다.
- 최초 gate `59951`의 실패 원인은 모델 결과가 아니라 `scripts/check_thinking_smoke.py`의 import 경로였다. 스크립트 경로를 보정한 뒤 동일 원시 출력으로 다시 계산해 위 결과를 확인했다.
- generation 거부와 VLM 표적 역할은 별개이므로, 준비된 실제 benchmark로 표적 VLM 성능을 먼저 측정한다. `survey_exp/vlm_benchmark_eval.py`는 FigStep 200목표, HADES 고정 step 5의 200목표, 기존 HIMRD SD3 이미지·고정 prompt 200목표를 새 디렉터리에 기록한다.
- Slurm `59955`는 위 600건을 한 번의 H200 모델 적재로 순차 실행한다. 출력은 각각 `runs/open_best_qwen3vl30b_figstep200_20261127`, `runs/open_best_qwen3vl30b_hades_step5_200_20261127`, `runs/open_best_qwen3vl30b_himrd200_20261127`이다.
- 이 600건은 가능한 구성요소의 부분 평가다. 자동 거부율과 응답 완료율은 보고할 수 있지만, 사람평가 전 ASR 또는 5개 기법 전역 최적 조합으로 표현하지 않는다.

### 전체 후보 실행 큐

- `59958`은 `open_best` profile의 revision 고정 자산 16개를 최대 동시 2개로 내려받는 Slurm array다. 모델별 결과는 `runs/open_best_download_status/task-XX.json`에 남는다.
- VLM 600건: Qwen Instruct `59957`, Qwen Thinking `59962`, InternVL `59963`, Mistral Small `59964`, Gemma 3 `59965`, GLM-4.5V 2×H200 상한 `59966`.
- LLM 텍스트 역할 200목표: gpt-oss `59967`, Qwen3 `59968`, Mistral Small `59969`, Gemma 3 `59970`, Llama 3.3 2×H200 `59971`. HADES caption/diffusion-prompt와 HIMRD scene/prefix/suffix의 5개 역할, 총 1,000레코드를 모델마다 계획한다. 이미지 생성은 이 단계에서 끈다.
- 이미지 생성기 200 prompt: Qwen-Image-2512 `59972`, Qwen-Image-2.1 `59973`, SD3.5 Large `59974`, FLUX.1-dev `59975`, PixArt Sigma `59976`. 공통 입력은 `data/manifests/open_best_image_prompts_200.jsonl`이다.
- 모든 평가 작업은 대응 다운로드 array element의 `afterok` 의존성을 갖는다. gated 접근 실패 또는 로더 비호환은 해당 작업만 실패하며, 성공한 다른 모델의 원시 결과를 삭제하지 않는다.
- 위 작업은 구성요소별 전수 평가다. 125개 조합의 end-to-end 본선은 MM-Plan·MemJack을 포함한 다섯 방법의 200개 고정 pool과 구성요소 gate가 모두 준비된 뒤 별도 제출한다.

### 최대 경우의 수 확장

- 포트 8000 고정 때문에 같은 H200 노드의 서로 다른 작업이 잘못된 서버에 연결되는 문제가 확인됐다. Slurm job ID 기반 고유 포트로 수정했다.
- Mistral·Gemma 최초 VLM 600건은 전부 HTTP 500이므로 유효 완료가 아니다. FP32 image tensor/BF16 weight 불일치도 수정했고 `60030`, `60031` retry에서 새 출력 디렉터리를 사용한다.
- Qwen Thinking은 FigStep·HADES 완료 뒤 포트 충돌로 HIMRD 12건에서 중단됐다. `60029`가 세 기법 600건을 새 ID로 재실행한다.
- GLM-4.5V 최초 2×H200 적재는 46 shard 중 28번째에서 비정상 종료됐다. `60032`는 4×H200·300GB RAM으로 재시도한다.
- Llama 3.3과 SD3.5 Large는 현재 HF 계정에서 각각 403이다. 원 후보를 삭제하지 않고 Qwen3-32B와 이미 승인된 SD3 Medium을 실행 대체 후보로 추가했다. 다운로드 array `60027`, 후속 LLM `60036`, 이미지 `60037`이다.
- 확장 행렬 `configs/open_best_combinations_expanded.jsonl`은 6×6×6=216개다. 현재 접근 가능한 실행 조합은 최대 6×5×5=150개이며, blocked 조합도 누락하지 않고 이유와 함께 집계한다.

### 150개 실행 가능 조합의 종단간 교차 큐

- HADES와 HIMRD의 실제 3단 경로에서 `5 LLM × 5 이미지 생성기 × 6 VLM × 2기법 × 200목표 = 60,000`개 표적 VLM 호출을 계획했다. 각 성공 출력에 HarmBench와 Qwen3Guard를 적용하므로 자동 판정 계획량은 120,000개다.
- 이미지 교차 작업은 Qwen-Image-2512 `60039`, Qwen-Image-2.1 retry `60048`, FLUX `60041`, PixArt `60042`, SD3 Medium `60043`이다. 각 작업은 다섯 LLM 결과가 모두 끝난 뒤 2,000개 이미지를 생성한다.
- Qwen-Image-2.1 최초 구성요소 작업 `59973`은 설치된 diffusers 0.35.2에 `QwenImage21Pipeline`이 없어 실패했다. 공식 요구사항을 다른 모델과 분리한 `.venvs/qwen21`을 만들었고, 재현 설치는 `scripts/install_qwen_image21_env.sh`, 구성요소 retry는 `60047`이다. 실패한 원시 provenance는 보존한다.
- 10,000건 VLM 교차 작업은 Qwen Instruct `60055`, InternVL `60056`, Qwen Thinking `60051`, Mistral `60052`, Gemma `60053`, GLM `60054`다. GLM만 4×H200·300GB를 요청한다.
- 각 VLM 작업 뒤 두 judge 작업을 제출했다. `60057`–`60068`이 HarmBench/Qwen3Guard 12개 작업이다. 이들은 자동 판정이며 사람평가를 대체하지 않는다.
- FigStep·MM-Plan처럼 이미지 생성 축이 구조적으로 없는 기법에 T2I 모델을 억지로 삽입하지 않는다. 해당 축은 `not_applicable`로 기록하고 실제 역할에 맞는 부분 행렬로 평가한다. MM-Plan·MemJack 200개 pool blocker가 해소되기 전에는 다섯 기법 전역 순위를 내지 않는다.
- 교차 runner는 `survey_exp/open_best_cross_image_eval.py`, `survey_exp/open_best_cross_vlm_eval.py`, `survey_exp/open_best_cross_judge_eval.py`이고 Slurm entrypoint는 대응하는 `slurm/run_open_best_cross_*_h200.sbatch`다.

### 2026-11-28 실패 수정과 실행 가능한 125조합 재제출

- 기존 gpt-oss Transformers 작업 `60038`은 MXFP4용 Triton 조건을 충족하지 못해 BF16·CPU offload로 실행됐다. 약 17시간에 18/1,000레코드뿐이어서 취소했고, 그 18개 원시 레코드는 `runs/open_best_gpt_oss120b_retry1_20261127_llm_roles200`에 보존한다.
- 모델 카드의 과거 전용 vLLM wheel은 더 이상 제공되지 않는 Torch nightly를 요구했다. 현재 설치 가능한 `vllm==0.34.0`과 Torch 2.14.0의 격리 환경 `.venvs/gptoss-vllm`을 만들었고, 설치 재현은 `scripts/install_gptoss_vllm_env.sh`에 고정했다. 새 gpt-oss 작업은 `60314`, 출력은 `runs/open_best_gpt_oss120b_retry2_20261128_llm_roles200`이다.
- Qwen-Image-2.1 retry1 `60047`은 `Qwen3VLVideoProcessor`가 요구하는 torchvision 누락으로 실패했다. `.venvs/qwen21`에 `torchvision==0.22.1`을 추가하고 pipeline import를 검증했다. retry2는 `60313`, 새 출력은 `runs/open_best_qwen_image21_retry2_20261128_image200`이다.
- gated 403인 Llama 3.3·SD3.5 Large를 Qwen3-32B·SD3 Medium으로 명시적으로 대체한 완주 가능 125행을 `configs/open_best_combinations_executable125.jsonl`에 고정했다. SHA-256은 `3cbb50754c32af5628b729962956a1dd86288b42175f1c07df4737fdc7f81ff6`이다. 원래 125행렬 결과로 표현하면 안 된다.
- 새 이미지 교차 작업은 `60315`–`60319`, 5개 표적 VLM 작업은 `60320`–`60324`, 두 judge의 10개 작업은 `60325`–`60334`다. 이 체인은 5×5×5=125개 조합, HADES·HIMRD 각 200목표의 50,000개 표적 호출과 100,000개 자동 판정을 계획한다.
- 이 작업들은 제출 완료 상태일 뿐 결과 완료 상태가 아니다. `60313`과 `60314`의 smoke·완료를 먼저 확인하고, downstream 완료는 각 `completed.json`과 원시 행 수로 판정한다.

### 안전정렬 제거 짝지어진 추가 트랙

- 기존 125조건은 보존하고 `configs/alignment_ablation_combinations.jsonl`에 별도 20조건을 고정했다.
- LLM은 Qwen3-30B-A3B-Instruct-2507 원본/abliterated, VLM은 Qwen3-VL-30B-A3B-Instruct 원본/abliterated를 짝지었다. 두 community 파생 모델 revision은 `configs/assets.lock.json`에 고정했다.
- HADES·HIMRD 각 200목표, 이미지 생성기 5개를 사용해 표적 호출 8,000건과 HarmBench·Qwen3Guard 판정 16,000건을 계획한다.
- `scripts/submit_alignment_ablation.sh`가 모델 다운로드 → abliterated LLM 역할 산출물 → 5개 이미지 생성 → aligned/abliterated VLM → 두 judge의 dependency chain을 제출한다.
- 2026-11-28 제출 ID는 모델 다운로드 `60336`–`60337`, abliterated LLM `60338`, 이미지 생성 `60339`–`60343`, aligned/abliterated VLM `60344`–`60345`, 두 judge `60346`–`60349`다. 제출 직후 다운로드 두 작업은 실행 중이고 나머지는 정상적으로 dependency 대기 중이었다.
- 점검 중 최초 다운로드 `60336`–`60337`만 프로젝트 내부 `.cache/huggingface`를 사용하고 후속 H200 작업은 `$HF_HOME` 또는 사용자 HF cache를 사용하는 기본값 불일치를 확인했다. 약 9.2GB의 부분 다운로드는 삭제하지 않고 보존했으며, 두 작업을 취소한 뒤 공유 cache를 명시한 `60350`–`60351`로 재제출했다. `60338`과 `60345`의 dependency도 새 ID로 갱신했다. `slurm/download_open_best_candidate.sbatch`와 제출 스크립트는 이후 모든 단계에 동일한 `HF_HOME`을 전달한다.
- 출력은 모두 `runs/alignment_ablation_*` 또는 `runs/open_best_qwen3_30ba3b_abliterated_20261128_llm_roles200` 아래 새 실험 ID를 사용한다. 기존 JSONL을 덮어쓰지 않는다.
- 이 트랙은 일반 공개 모델 주 순위와 분리하며, 거부 감소를 곧바로 품질 또는 공격 성공 증가로 해석하지 않는다.

### 같은 시점의 기존 큐 상태 보충

- Qwen-Image-2.1 retry2 `60313`은 실행 약 15분 시점에 `outputs.jsonl` 37행을 기록해 실제 생성이 진행 중이었다.
- gpt-oss vLLM retry2 `60314`는 H200에서 서버 적재를 시작했고 `GptOssForCausalLM` 및 고정 revision을 확인했다. 아직 역할 출력 완료 상태는 아니다.
- 확장 상한 후보 GLM-4.5V retry `60032`는 4×H200에서도 46개 shard 중 37번째 적재 이후 exit code 7로 실패했다. 이 후보는 실행 가능 대체 125행렬에 포함되지 않으므로 `60315`–`60334` 체인을 막지 않지만, 216개 확장 원장에서는 `load_failure`로 보존한다.
