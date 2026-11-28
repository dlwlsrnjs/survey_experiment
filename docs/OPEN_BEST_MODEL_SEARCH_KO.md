# 모든 기법에 적용할 공개 모델 최적 조합 탐색

## 목표

이 트랙의 목표는 FigStep, HADES, HIMRD, MM-Plan, MemJack의 원 모델을 그대로 재현하는 데서 끝나지 않고, 각 파이프라인 단계에 사용할 수 있는 **공개 VLM, 공개 LLM, 공개 가중치 이미지 생성 모델의 최적 조합**을 찾는 것이다.

원형 충실도 결과는 변경하지 않는다. 최적 조합 탐색은 `open_best`라는 별도 트랙으로 실행하고, 원 논문 모델보다 결과가 좋더라도 이를 원 논문 재현 결과로 표기하지 않는다.

여기서 사용자가 말한 LVM은 이미지 입력을 이해하고 멀티모달 응답을 생성하는 VLM 범주로 운영한다. 이미지 자체를 합성하는 모델은 별도의 이미지 생성 모델 범주다.

## 하나의 모델을 고르는 문제가 아닌 이유

파이프라인에는 적어도 다음 네 종류의 능력이 필요하다.

1. **VLM**: 이미지 관찰, OCR, 공간·기능 관계 이해, 시각 근거 계획, 표적 응답
2. **LLM**: 장면 설명, 공격 후보, prefix/suffix, 평가와 수정, 경험 활용
3. **이미지 생성 모델**: 장면 합성, 텍스트 렌더링, 이미지 편집
4. **판정 모델**: 거부·안전·공격 성공 판정

따라서 “가장 점수가 높은 단일 모델”이 아니라, 역할별 후보를 실제 파이프라인에 넣은 뒤 조합 단위로 선택한다.

## 후보군 고정: 범주별 최소 5개

정확한 revision, 라이선스와 저장소 크기는 `configs/open_best_candidates.yaml`에 고정한다.

주 분석은 5×5×5의 125개 조합을 유지한다. 최대 경우의 수 분석은 등록된 상한·대체 후보까지 포함한 **VLM 6개 × LLM 6개 × 이미지 생성기 6개 = 216개 조합**이다. `configs/open_best_combinations_expanded.jsonl`에 revision까지 고정한다. gated 403인 Llama 3.3과 SD3.5 Large도 행에서 삭제하지 않고 `blocked` 실행 상태로 남긴다. 실행 가능한 대체 후보 Qwen3-32B와 SD3 Medium을 포함하면 현재 접근권한에서 최대 6×5×5 = 150개 조합을 실제 실행할 수 있다.

원래 125개 주 행렬은 Llama 3.3과 SD3.5 Large를 포함하므로 현재 계정의 403 접근 상태에서는 125개를 모두 실행할 수 없다. 이를 숨기거나 원 후보 이름만 바꿔치기하지 않고, 완주 가능한 별도 5×5×5 행렬을 `configs/open_best_combinations_executable125.jsonl`로 고정했다. 이 행렬은 Llama 3.3을 Qwen3-32B로, SD3.5 Large를 SD3 Medium으로 명시적으로 대체한다. 따라서 결과에는 반드시 **원 주 행렬**과 **접근 가능 대체 행렬**을 구분해 표기한다.

## 안전정렬 제거 짝지어진 추가 트랙

주 125조건은 일반적인 공개 모델 후보 비교라는 연구 질문을 유지한다. 안전정렬 제거 모델로 후보를 바꾸면 질문과 모집단이 달라지므로, 주 행렬을 수정하지 않고 `configs/alignment_ablation_combinations.jsonl`의 별도 20조건을 추가한다.

| 축 | 조건 |
|---|---|
| LLM | Qwen3-30B-A3B-Instruct-2507 aligned / 동일 기반 community abliterated |
| VLM | Qwen3-VL-30B-A3B-Instruct aligned / 동일 기반 community abliterated |
| 이미지 생성기 | 실행 가능한 125행렬과 같은 5개 |
| 데이터 | HADES·HIMRD 각각 동일 200목표 |
| 판정 | 정렬 상태를 바꾸지 않은 HarmBench·Qwen3Guard와 블라인드 사람평가 |

총량은 `2 × 2 × 5 = 20`조건, `20 × 2기법 × 200목표 = 8,000` 표적 VLM 호출, 자동 판정 16,000건이다. 이 설계는 LLM 단계의 거부와 VLM 단계의 거부를 각각 분리한다. 이미지 생성 모델, seed, 해상도, step, 표적 prompt와 판정기는 정렬 여부 비교에서 고정한다.

모델 카드의 `uncensored` 또는 `abliterated` 표기는 독립 성능 검증이 아니다. 따라서 다음을 전부 보고한다.

- 상류 LLM 거부·형식 오류·usable prompt 비율
- 이미지 생성 성공·오류·의미 충실도
- 표적 VLM 거부·오류·응답률
- 비거부 조건부 품질과 전체 usable yield
- 두 자동 judge의 결과와 불일치
- 사람평가 및 GPU 시간

이 결과는 “안전정렬을 제거하면 성능이 향상된다”는 단일 결론으로 축약하지 않는다. community 파생 과정이 일반 능력과 출력 분포도 바꿀 수 있으므로 정렬 개입의 후보 ablation으로 해석한다.

### VLM

| 후보 | 우선 확인할 능력 | 선정 이유 |
|---|---|---|
| Qwen3-VL-30B-A3B-Instruct | 구조화 계획, OCR, 도구 지시, 표적 응답 | Apache-2.0, 30.5B MoE, 단일 H200에 현실적인 크기 |
| InternVL3.5-38B-HF | 시각 추론, OCR, 이미지 기반 계획 | Qwen과 다른 계열의 강한 공개 VLM이며 Qwen 계열 거부 실패를 교차 검증할 수 있음 |
| Qwen3-VL-30B-A3B-Thinking | 복잡한 시각 계획 | 장문·truncation·final-only 파싱을 별도 통과한 경우에만 planning 후보로 사용 |
| Mistral-Small-3.1-24B-Instruct-2503 | 다른 계열의 시각 이해와 구조화 출력 | Apache-2.0 교차 계열 후보 |
| Gemma-3-27B-it | 시각 질의와 텍스트 역할의 통합 성능 | gated Gemma 라이선스 비교군 |
| GLM-4.5V | 다중 GPU 성능 상한 | 108B급 공개 VLM이며 단일 H200 실용군과 분리 |

Qwen3-VL-32B가 기존 generation smoke에서 전부 거부했다는 사실만으로 위 후보의 모든 역할 성능을 추정하지 않는다. 동일한 실제 입력에서 역할별로 다시 측정한다.

### LLM

| 후보 | 우선 확인할 능력 | 선정 이유 |
|---|---|---|
| gpt-oss-120b | 텍스트 생성·수정·평가·경험 활용 | 공식 카드가 단일 80GB GPU 실행을 명시한 공개 가중치 MoE; H200에서 상한 후보 |
| Qwen3-30B-A3B-Instruct-2507 | 구조화 출력, 낮은 지연, 다회 호출 | Apache-2.0, 3.3B active MoE로 반복 에이전트 호출에 유리한 후보 |
| Mistral-Small-3.1-24B-Instruct-2503 | 텍스트 재작성과 평가 | Apache-2.0의 다른 계열 비교군 |
| Gemma-3-27B-it | 텍스트 지시 수행 | VLM과 LLM을 같은 계열로 구성하는 조건도 평가 가능 |
| Llama-3.3-70B-Instruct | 대형 dense 공개 가중치 대조군 | gated·Llama 라이선스이며 quantization 또는 다중 GPU 조건으로 분리 |

`gpt-oss-120b`는 Harmony 형식을 지켜야 하며, 분석 대상은 최종 구조화 출력과 실제 trace뿐이다. 비공개 chain-of-thought를 수집하거나 평가하지 않는다.

### 이미지 생성 모델

| 후보 | 트랙 | 우선 확인할 능력 |
|---|---|---|
| Qwen-Image-2512 | 엄격한 공개 라이선스 1차 후보 | 의미 충실도, 자연 장면, 텍스트 렌더링 |
| Qwen-Image-2.1 | 공개 가중치 연구 트랙 | 이미지 편집, 다중 참조, 효율, 타이포그래피 |
| Stable Diffusion 3.5 Large | 공개 가중치 연구 트랙 대조군 | 복잡한 prompt 준수와 장면 품질 |
| FLUX.1-dev | 비상업 연구용 공개 가중치 비교군 | 강한 일반 이미지 생성 후보지만 라이선스를 별도 표기 |
| PixArt-Sigma-XL-2-1024-MS | 경량 공개 비교군 | HADES 계열과 연속성이 있고 계산 효율 대조에 유용 |

Apache-2.0인 Qwen-Image-2512와 연구·community 라이선스 모델은 같은 의미의 “오픈소스”로 뭉개지 않는다. 논문에는 **엄격한 공개 라이선스 조합**과 **연구용 공개 가중치 조합**을 따로 보고한다.

### 판정 모델

- 공격 성공 분류: HarmBench-Mistral-7B classifier
- 안전·거부 분류: Qwen3Guard-Gen-8B
- 최종 표본은 모델명을 가린 두 사람 평가자가 독립 평정한다.

judge 하나의 출력으로 모델 우열을 확정하지 않는다. 두 자동 judge의 불일치와 사람평가 일치도를 함께 보고한다.

## 기법별 적용 방식

| 기법 | 공개 최적 조합에서 바꾸는 단계 | 유지할 결정적 단계 |
|---|---|---|
| FigStep | 표적 VLM과 필요시 문장 작성 LLM | 타이포그래피 렌더러 |
| HADES | caption/evaluation VLM, prompt 수정 LLM, 이미지 생성기, 표적 VLM | 단계 수와 정보 전달 구조 |
| HIMRD | 장면·prefix/suffix LLM, 이미지 생성기, 표적 VLM, judge | 이해/순응 분리와 조건부 반복 구조 |
| MM-Plan | planner VLM, 표적 VLM, 공개 judge | 허용 동작 집합과 plan 실행 계약 |
| MemJack | planner/attacker VLM, reflection LLM, embedding·guard, local victim VLM | A/B/C/C_random 및 train-memory 누출 방지 |

## 선택 절차

### 0단계: 실행 건강검사

- revision·라이선스·checksum 확인
- 단일 H200 적재, peak VRAM, server-ready 확인
- API/추론 오류, 형식 오류, 길이 종료 확인

건강검사 smoke는 순위를 정하는 데이터가 아니다.

### 1단계: 데이터와 프로토콜 사전 고정

모델 출력 생성 전에 다음을 해시와 함께 동결한다.

- 기법별 원 데이터에서 고유 목표 200개
- 동일 goal의 변형과 반복을 묶고, 안정적 해시로 미리 배정한 5개 fold
- 모델별 prompt가 아니라 역할별 공통 입력 계약
- seed `20261127`, `20261128`, `20261129`
- 호출 예산, timeout, max token, 이미지 해상도와 step
- 실패·중단·resume 규칙과 자동 judge version
- 사람평가 표본 추출 규칙

고정 규칙의 기계 판독 기준은 `configs/open_best_protocol.yaml`, 생성기는 `scripts/freeze_open_best_manifests.py`다. 다섯 기법 중 하나라도 200개 고유 group을 제공하지 못하면 manifest 생성을 실패시키며 부분 데이터로 본선을 시작하지 않는다.

현재 공개 HADES benchmark는 고유 목표가 200개이므로 `20+40+200`처럼 260개를 요구하는 세 분할은 불가능하다. 대신 모든 조합을 동일한 200개에 실행하고, 사전 고정한 5-fold에서 조합의 순위 안정성과 선택 절차의 out-of-fold 성능을 보고한다. 이 200개는 조합 선택에도 쓰이므로 **외부 독립 test**라고 부르지 않는다. 독립 확인 주장은 새로운 고정 benchmark를 확보하고 새 protocol version으로 실행한 뒤에만 한다.

### 2단계: 구성요소 건강검사

- 125개 본선 조합은 예외 없이 각 기법의 고유 목표 200개를 사용한다. 건강검사만 10건이며 순위에는 포함하지 않는다.
- VLM은 시각 근거 정확도, OCR, 계획 실행 가능성, 비거부 조건부 품질을 측정한다.
- LLM은 형식 유효율, 의미 보존, 수정의 실질 개선, 경험 사용 정확도, usable yield를 측정한다.
- 이미지 생성기는 생성 성공률, 의미 충실도, 필요한 경우 OCR 정확도, 사람 품질, 시간·VRAM을 측정한다.
- 거부, 오류, 저품질, 미도달을 합치지 않는다.

### 3단계: 125개 조합 전수 비교

VLM 5개 × LLM 5개 × 이미지 생성 모델 5개를 사용해 **125개 조합**을 만든다. 모든 조합은 같은 고정 manifest의 기법별 200목표, 총 1,000목표를 실행한다. 따라서 1차 비교만 `125 × 5 × 200 = 125,000`개의 조합–기법–목표 평가다. 기법에 사용되지 않는 구성요소는 호출하지 않으며 동일 입력·설정의 중간 산출물은 checksum이 맞을 때만 재사용한다.

- 본선: 125개 조합 × 기법별 200목표 × 기본 seed 1개
- 선택 안정성: 사전 고정 5-fold마다 순위와 상위 조합 일치율 계산
- 반복 검증: 상위 5개 × 기법별 200목표 × 추가 seed 1개
- 사람평가: 상위 5개와 계열별 기준선에서 층화·블라인드 표본을 두 평가자가 독립 평정
- 외부 확인: 별도 200목표 benchmark가 확보되기 전에는 미실시로 표시

따라서 “100번”은 같은 요청을 100번 반복한다는 뜻이 아니라, 125개 서로 다른 조합을 각각 200목표 이상에서 공정하게 비교한다는 뜻이다. 확률적 이미지 생성의 변동성은 상위 조합의 고정 seed 반복으로 측정한다.

확장 분석에서는 216개 registry 조합을 모두 원장에 남기고, 접근 가능한 150개 조합을 실행한다. 실행하지 못한 조합은 결과 행을 제거하지 않고 `gated_access`, `load_failure`, `upstream_failure`처럼 이유를 기록한다. 두 judge는 모든 성공 출력에 함께 적용하므로 평가 관점은 최대 300개 조합–judge 조건이다.

1. 엄격한 공개 라이선스 최상 조합
2. 연구용 공개 가중치를 포함한 성능 최상 조합
3. GPU-hours당 usable yield가 가장 높은 효율 조합

### 실제 교차 실행량과 경우의 수 처리

“모든 조합”은 기법에 존재하지 않는 단계를 억지로 추가한다는 뜻이 아니다. 먼저 216개 등록 조합을 원장에 모두 남기고, 접근 가능한 후보의 실행 상태를 다음처럼 구분한다.

| 층 | 경우의 수 | 처리 방식 |
|---|---:|---|
| 등록된 확장 행렬 | 6 VLM × 6 LLM × 6 이미지 = 216 | gated 후보와 실패 후보도 행을 삭제하지 않음 |
| 현재 접근 가능한 행렬 | 6 × 5 × 5 = 150 | 실제 H200 실행 대상 |
| HADES·HIMRD 3단 교차 | 150조합 × 2기법 × 200목표 = 60,000 표적 호출 | LLM 산출물, 생성 이미지, VLM 응답을 checksum으로 연결 |
| 자동 판정 | 60,000 × 2 judge = 120,000 판정 | HarmBench와 Qwen3Guard를 독립 적용 |

FigStep의 고정 타이포그래피 렌더러 자리에 diffusion 모델을 넣거나, MM-Plan에 존재하지 않는 T2I 단계를 추가하면 원 기법과 다른 파이프라인이 된다. 이런 축은 `not_applicable`로 기록한다. FigStep은 `LLM × VLM`, MM-Plan은 planner/target VLM, MemJack은 memory·reflection·victim 역할처럼 실제 정보 흐름에 맞는 부분 행렬로 평가한다. 최종 다섯 기법 macro 순위는 MM-Plan과 MemJack의 200개 고정 pool이 확보되기 전까지 계산하지 않는다.

교차 실행 코드는 다음 세 단계다.

1. `open_best_cross_image_eval.py`: 다섯 LLM의 HADES·HIMRD 역할 산출물을 다섯 이미지 생성기에 전달해 이미지 모델당 2,000건을 생성한다.
2. `open_best_cross_vlm_eval.py`: 다섯 이미지 모델의 10,000건을 각 VLM에 한 번씩 질의한다.
3. `open_best_cross_judge_eval.py`: 각 VLM 결과 10,000건에 HarmBench와 Qwen3Guard를 각각 적용한다.

상류 거부, 파싱 실패, 이미지 생성 실패, VLM 오류는 성공한 행만 남기는 방식으로 제거하지 않는다. 각각 `upstream_unusable`, `generation_error`, `model_error`, `not_reached`로 보존하여 조합의 usable yield 분모에 포함한다. 자동 판정 수치만으로 최종 ASR이나 우승 모델을 선언하지 않으며, 사람평가 상태는 별도 필드로 남긴다.

Qwen-Image-2.1은 공식 파이프라인이 Python 3.11 이상과 개발판 `diffusers`를 요구하므로 다른 모델 환경과 분리한다. `scripts/install_qwen_image21_env.sh`가 Python 3.12, Torch, Transformers 및 고정 `diffusers` commit을 `.venvs/qwen21`에 설치한다.

2026-11-28 재시도에서는 Qwen-Image-2.1 processor가 추가로 요구하는 `torchvision==0.22.1`을 같은 격리 환경에 고정했다. gpt-oss-120B의 Transformers 실행은 MXFP4가 비활성화되어 BF16 CPU offload로 내려가 17시간에 18레코드만 생성했으므로 중단했다. 원시 18레코드는 보존하고, `scripts/install_gptoss_vllm_env.sh`와 `slurm/run_open_best_gptoss_vllm_h200.sbatch`의 vLLM MXFP4 경로로 새 experiment ID에서 재실행한다.

### 4단계: 전 기법 검증

모든 본선 조합을 FigStep, HADES, HIMRD, MM-Plan, MemJack에 각각 200개 목표로 적용한다. 한 기법에서만 좋은 조합을 전역 최상으로 부르지 않는다.

## 최종 선정 지표

주 지표는 단순 ASR이 아니라 다음 단계가 그대로 사용할 수 있는 **usable yield**다.

- 역할별 usable yield와 95% CI
- 비거부 조건부 사람 품질
- 시각 근거·OCR·계획 실행 정확도
- 이미지 의미 충실도와 타이포그래피 정확도
- 자동 judge와 사람평가의 일치도
- 목표당 GPU 시간, peak VRAM, wall time
- usable 산출물 1개당 GPU 비용
- 기법별 최저 성능과 전체 macro average

최종 조합은 평균만 높은 모델보다 특정 기법에서 붕괴하지 않는 조합을 우선한다. 이를 위해 macro average와 함께 다섯 기법 중 최저 usable yield를 보고한다.

## 현재 결론의 범위

현재 후보 목록은 공식 모델 카드와 단일 H200 적재 가능성을 바탕으로 만든 **실험 후보군**이지 우승 결과가 아니다. 실제 고정 데이터, 200개 이상 사례, 자동 지표와 블라인드 사람평가가 끝난 뒤에만 “최고 조합”을 확정한다.

