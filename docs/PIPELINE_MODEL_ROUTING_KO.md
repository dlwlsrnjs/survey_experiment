# 원 논문 파이프라인의 단계별 모델 라우팅

이 문서는 각 기법의 모든 단계를 하나의 LLM 또는 하나의 32B 모델로 치환하지 않기 위한 실행 기준이다. 원형 충실도 트랙에서는 단계가 요구하는 모달리티와 원 논문·공개 구현의 모델을 우선한다.

## 공통 원칙

- 이미지 이해, 이미지 근거 계획, 멀티모달 표적 응답은 **VLM**으로 실행한다.
- 텍스트 재작성이나 텍스트 기반 판정만 필요한 단계는 원 구현이 지정한 **LLM**을 사용한다.
- 이미지 합성은 VLM이 아니라 **PixArt, SD3, SDXL 같은 이미지 생성 모델**로 실행한다.
- 원 구현이 공개 checkpoint를 지정하면 해당 모델과 revision을 보존한다.
- 폐쇄형 API가 원 절차의 필수 단계일 때만 폐쇄형 모델을 호출한다.
- 공개 모델을 임의의 32B 모델로, 폐쇄형 모델을 임의의 로컬 모델로 바꾼 실행은 원형 재현이 아니라 `model_substitution` 또는 공통 통제 실험으로 분리한다.
- 자동 판정 완료와 사람평가 완료를 구분한다.

## 기법별 모델 선택

| 기법 | 단계 | 원 논문·공개 구현 모델 | 로컬 실행 | 폐쇄형 모델이 필요한 경우 |
|---|---|---|---|---|
| FigStep | 타이포그래피 이미지 | 결정적 렌더러 | 가능 | 불필요 |
| FigStep | 표적 VLM | LLaVA-1.5-7B 등 공개 VLM | `liuhaotian/llava-v1.5-7b` | GPT-4V 같은 폐쇄형 표적을 별도 평가할 때만 필요 |
| HADES | 이미지 생성·최적화 | PixArt-XL-2-1024-MS | 가능 | 불필요 |
| HADES | 표적 VLM | LLaVA-1.5-7B | 가능 | 논문의 black-box GPT-4V/Gemini 조건을 재생할 때만 해당 API 필요 |
| HADES | 안전 판정 | Beaver-Dam-7B | 가능 | 불필요 |
| HIMRD | 장면 설명·prefix/suffix 수정 | GPT-4o-mini/GPT-4o/o1 계열 역할 | 로컬 치환 시 별도 대체실험 | 현재 재현은 고정 snapshot `gpt-4o-mini-2024-07-18`; 더 엄격한 원 설정에는 GPT-4o/o1 필요 |
| HIMRD | 이미지 생성 | Stable Diffusion 3 Medium | 가능, gated checkpoint 필요 | 불필요 |
| HIMRD | 표적 VLM | LLaVA-v1.6-Mistral-7B 외 공개 VLM들 | 가능 | GPT-4o 표적 조건만 API 필요 |
| HIMRD | 성공 판정 | HarmBench-Mistral-7B classifier | 가능 | 불필요 |
| MM-Plan | attacker planner | GRPO 학습된 Qwen3-VL-4B | base checkpoint만 공개됨 | 학습 checkpoint 미공개 문제는 폐쇄형 모델로 해결되지 않음 |
| MM-Plan | 표적 VLM | Qwen3-VL-8B 또는 논문의 폐쇄형 표적 | 로컬 Qwen3-VL-8B 가능 | Claude 4.5 Sonnet/GPT-5 표적 결과를 재생할 때 해당 API 필요 |
| MM-Plan | judge | Claude Sonnet 4.5 | 원 설정의 로컬 judge 없음 | `Claude-Sonnet-4.5` 필요 |
| MemJack | planner·attacker·reflection | Qwen3-VL-8B-Instruct | 가능 | 불필요 |
| MemJack | embedding | Qwen3-VL-Embedding-8B | 가능 | 불필요 |
| MemJack | safety guard | Qwen3Guard-Gen-8B | 가능 | 불필요 |
| MemJack | 논문 주 victim | Qwen3-VL-Plus | 공개 코드의 Qwen3-VL-8B victim은 대체 조건 | 원 결과 재생에는 `qwen3-vl-plus` API 필요 |

## 현재 실행 결과와 경계

### FigStep

- `runs/figstep_llava15_200_retry1_20261126`
- LLaVA-1.5-7B로 200/200 응답 생성, 실행 오류 0건.
- 논문 주 평가 방식에 맞춘 사람평가는 아직 완료되지 않았으므로 ASR은 `null`이다.

### HADES

- `runs/hades_released_llava15_200goals_1200rows_retry1_20261126`
- 공식 공개 최적화 이미지 200목표 × 6단계, 총 1,200개 LLaVA 응답과 Beaver 판정 완료.
- 공개 산출물 평가이며 PixArt 최적화 이력을 처음부터 재생성한 결과는 아니다.

### HIMRD

- `runs/himrd_e2e_llava16_200_retry1_20261126`
- GPT-4o mini 역할 → SD3 Medium → LLaVA-v1.6-Mistral-7B → HarmBench 판정의 200조건 완료.
- 자동 HarmBench 판정은 194/200을 성공으로 표시했지만, 이 값은 실행된 모델·prompt·자동 judge에 한정되며 사람 검증 전 최종 결론으로 확대하지 않는다.

### MM-Plan

- 2026-11-27 KST 기준 Hugging Face API에서 dataset metadata는 보이지만 실제 VE-Safety 파일 다운로드는 HTTP 403이다.
- 논문 학습 완료 Qwen3-VL-4B planner checkpoint도 공개 저장소에 없다.
- 따라서 base Qwen3-VL-4B만으로 실행한 결과를 MM-Plan 재현으로 표기하지 않는다.
- 접근 승인과 학습 checkpoint 또는 8-GPU 재학습이 확보된 뒤, Qwen3-VL-8B local target과 Claude Sonnet 4.5 judge를 연결한다.

### MemJack

- 공개 궤적과 split은 내려받았고 공개 모델 revision도 동결했다.
- 고정 원 저장소에는 전체 실행에 필요한 `configs/safety_policy.txt`가 없으며 `data/targets.json`도 유효한 목표 파일이 아니다.
- 누락 파일을 임의 작성해 원형 재현으로 부르지 않는다.
- 원 결과에는 `qwen3-vl-plus` API가 필요하다. 공개 코드가 지원하는 Qwen3-VL-8B local victim은 별도 `model_substitution` 실험으로만 실행한다.

## 폐쇄형 모델 선택 요약

| 필요한 상황 | 사용할 모델 | 이유 |
|---|---|---|
| 현재 HIMRD GPT 역할 재현 | `gpt-4o-mini-2024-07-18` | 이미 고정 snapshot으로 200조건 종단간 실행을 완료했고 비용·provenance가 남아 있음 |
| HIMRD의 더 엄격한 원 API 조건 | 원 코드가 지정한 GPT-4o 및 o1 | GPT-4o mini 결과와 별도 조건으로 기록해야 함 |
| MM-Plan 원 judge | Claude Sonnet 4.5 | 공개 평가 코드의 기본 judge |
| MM-Plan 폐쇄형 target 비교 | Claude 4.5 Sonnet, GPT-5 | 논문이 보고한 폐쇄형 표적 조건 |
| MemJack 논문 주 victim | Qwen3-VL-Plus | 공개 config가 지정한 원 victim |
| HADES black-box target 비교 | GPT-4V 또는 Gemini Vision 계열의 논문 지정 snapshot | 로컬 LLaVA 원형과 섞지 않는 별도 표적 조건 |

OpenAI API 키만으로 Claude Sonnet 4.5나 Qwen3-VL-Plus를 호출할 수는 없다. 각각 해당 공급자의 접근 권한과 키가 필요하다.
