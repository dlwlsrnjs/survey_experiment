# 공개 코드 기준 모델 역할 및 실행환경 감사

작성 기준: 각 저장소의 README와 실제 호출 코드를 우선했다. 논문에서 언급되지만 공개 코드가 호출하지 않는 모델, API 전용 모델, 로컬 체크포인트를 각각 구분한다.

## 1. FigStep

- 공격 입력 제작: `src/generate_prompts.py`의 PIL 기반 고정 타이포그래피 렌더러. 학습 모델이나 LLM 에이전트가 아니다.
- 대상: 생성된 이미지와 고정 질문을 VLM에 입력한다. 대상 VLM은 비교 실험에서 별도로 고정해야 한다.
- 의미: FigStep에 32B “작성 에이전트”를 추가하면 원 논문에 없던 구성 요소가 되므로, 결정론적 기준선으로만 실행한다.

## 2. HADES

공개 코드 흐름:

1. `GPT-3.5-turbo-0613` caption writer: harmful instruction을 배경 장면 caption으로 변환한다 (`black_box/generate_caption.py`).
2. `GPT-3.5-turbo-0613` diffusion-prompt writer: caption을 120단어 이하 PixArt prompt로 변환한다 (`black_box/generate_sd_caption.py`).
3. `PixArt-XL-2-1024-MS` image generator: 현재 diffusion prompt에서 이미지를 생성한다.
4. `LLaVA-1.5-7B` image captioner: 생성 이미지를 `Provide a detailed description of the image.`로 다시 설명한다.
5. `GPT-4-0613` caption judge: LLaVA caption에 1–10 유해도 점수와 설명을 부여한다.
6. `GPT-4-0613` red-team optimizer: 이전의 `<P, C, S, Exp>` 전체 이력을 받고 다음 diffusion prompt를 만든다.
7. 3–6을 초기 step 0 뒤 5회 최적화하므로 기본적으로 이미지 6장/샘플이 생긴다.
8. 대상 MLLM: README의 주 대상은 LLaVA-1.5와 Gemini Pro Vision이고 GPT-4V 평가 경로도 있다.
9. 최종 안전성 평가: README는 `PKU-Alignment/beaver-dam-7b`를 명시한다. 캡션 최적화용 GPT judge와 대상 응답의 최종 ASR 평가는 분리해야 한다.

로컬 32B 비교 시 교체 대상은 1, 2, 5, 6의 언어/시각 보조 호출이다. PixArt, 대상 VLM, 최종 평가기는 동일하게 고정한다. 특히 5의 caption judge 결과와 대상 응답 성공 여부를 합치지 않는다.

## 3. HIMRD

공개 코드 흐름:

1. `GPT-4o-mini` risk distributor: harmful phrase를 텍스트에 남길 부분과 이미지로 옮길 부분으로 한 번 분할한다 (`data_gen.py:split_phrase`).
2. `GPT-4o-mini` scene writer: 이미지로 옮긴 phrase를 20단어 이하 장면 설명으로 만든다 (`data_gen.py:get_image_caption`).
3. `Stable Diffusion 3 Medium`: 512×512, 50 inference steps, guidance 8.0으로 하단 이미지를 만든다.
4. 결정론적 PIL renderer: 분리된 phrase를 상단 타이포그래피 이미지로 만들고 SD3 이미지와 세로 결합한다.
5. 대상 VLM: 공개 코드는 LLaVA-1.5-7B, LLaVA-1.6-Mistral-7B, GLM-4V-9B, DeepSeek-VL-7B-Chat, Qwen-VL-Chat, Yi-VL-34B, MiniGPT-4, GPT-4o를 지원한다. 논문은 이외 상용 Qwen-VL-MAX/Gemini도 보고하지만 공개 실행 코드는 제공하지 않는다.
6. `HarmBench-Llama-2-13B classifier`: 대상 출력이 attack behavior를 실질적으로 수행했는지 Yes/No 판정한다.
7. 이해 여부 분기: 짧은 응답과 거부 prefix를 포함해 `Understand`/`NotUnderstand`를 구분한다.
8. understanding-enhancing prefix rewriter: `NotUnderstand` 표본에 대해 공개 템플릿으로 보편 prefix를 재작성한다. 코드의 모델은 GPT-4o이며 최대 5회 형식 재시도, 외부 탐색 루프는 5회다.
9. inducing suffix rewriter: 이해했지만 실패한 표본에 대해 보편 suffix를 재작성한다. 코드의 모델은 o1-mini-2024-09-12이며 최대 20회 형식 재시도, 외부 탐색 루프는 5회다.
10. 각 재작성 뒤 대상 VLM을 다시 호출하고 HarmBench/이해 판정을 갱신한다.

로컬 32B 비교에서는 1, 2, 8, 9를 32B로 교체할 수 있지만, 대상 VLM·SD3·HarmBench는 고정한다. 8과 9는 실제 대상 응답에 따른 분기 없이 독립 프롬프트 생성만 돌리면 HIMRD 재현이 아니다.

## 4. MM-Plan

공개 코드 흐름:

1. attacker planner: 기본 `Qwen3-VL-4B-Instruct`; harmful behavior와 원 이미지를 보고 persona/context/approach 및 crop/mask/blur/no-op가 포함된 전체 multi-turn plan을 한 번에 생성한다.
2. parser/executor: planner JSON을 파싱하고 각 turn의 이미지 연산을 실행한 뒤 순서대로 target MLLM에 보낸다.
3. target MLLM: README 예시는 `Qwen3-VL-8B`; 논문 대표 결과는 Claude 4.5 Sonnet/GPT-5다.
4. judge: 전체 target 응답을 behavior와 함께 받아 성공 여부를 판정한다. README 예시는 Claude-Sonnet-4.5다.
5. training: 하나의 입력에서 plan 여러 개를 sampling하고 target/judge reward로 GRPO 학습한다. 공개 스크립트 기본은 temperature 0.7, rollout `n=4`, 8 GPU, 10 epochs다.

로컬 32B 비교에는 두 가지를 분리한다. (a) 공개 4B base/trained planner 재현, (b) 같은 planner prompt/parser/executor를 사용하되 planner만 32B로 바꾼 추론 비교다. target와 judge는 두 조건에서 같아야 한다. 공개 저장소에는 논문 학습 완료 checkpoint가 포함되어 있지 않아 base 4B와 직접 학습한 checkpoint를 별도 표기한다.

## 5. MemJack

공개 코드/기본 config의 agent 역할:

1. Vulnerability Planner (`Qwen3-VL-8B-Instruct`): 이미지의 공격 anchor와 취약 지점을 분석한다.
2. Attacker Agent (기본 API attacker; 코드 fallback 이름 `Qwen/Qwen3-VL-8B-Instruct`): 이미지·goal·anchor·이전 feedback/reflection·retrieved memory를 받아 다음 공격 prompt를 만든다.
3. Victim VLM: 기본 API `qwen3-vl-plus`; 로컬 경로는 `Qwen3-VL-8B-Instruct`를 지원한다.
4. Safety Guard (`Qwen3Guard-Gen-8B`): victim response를 Safe/Controversial/Unsafe, category, refusal 및 연속 risk score로 판정한다.
5. Reflection VLM (`Qwen3-VL-8B-Instruct`): 실패 응답의 defense pattern, 실패 이유, 새 방향, 개선 계획을 만든다. 기본 config는 safe failure streak 조건과 interval을 적용한다.
6. Embedding model (`Qwen3-VL-Embedding-8B`): prompt/visual/goal representation을 만들고 성공·실패 경험을 검색한다.
7. Memory Manager: 모델이 아니라 FAISS/metadata/Q-value 기반 저장·검색·갱신 구성 요소다.
8. Victim last-layer feature와 null-space/refusal predictor: white-box/local 조건에서 후보 filtering 및 과거 실패와의 유사도 회피에 사용된다.
9. `models.llm_path=Qwen3-8B`는 기본 config에 있으나 현재 핵심 agent 호출 경로에서는 직접 참조되지 않는 잔존 설정이다.

로컬 32B 조건은 2(Attacker Agent)만 우선 교체한다. 1·4·5·6과 victim은 원 역할 모델로 고정해야 memory/reflection 효과를 분석할 수 있다. A(독립), B(feedback/reflection), C(relevant memory), C-random을 동일 입력·query cap·seed로 비교한다.

## 6. 체크포인트/외부 서비스 구분

- 이미 완전 보유: Qwen2.5-VL-32B, Qwen3-VL-8B, Qwen3Guard-Gen-8B, HarmBench Llama-2-13B, HarmBench Mistral-7B, Qwen3-8B.
- 다운로드 대상: PixArt, LLaVA-1.5-7B, Beaver-Dam-7B, SD3 Medium, HIMRD 공개 대상 VLM들, Qwen3-VL-4B, Qwen3-VL-Embedding-8B, 고정 비교 target Qwen2.5-VL-7B.
- API 전용/다운로드 불가: GPT-3.5/4/4o/o1, Gemini, Qwen-VL-MAX, Claude Sonnet, GPT-5, qwen3-vl-plus. 현 단계의 로컬 32B 실험에서는 해당 **보조 agent 자리만** 32B로 대체하되, 상용 target/judge 결과를 로컬 재현으로 가장하지 않는다.
- 별도 획득 필요: MiniGPT-4 HF 저장소는 코드/설정만 제공하며 Vicuna/EVA/Q-Former 및 project checkpoint를 별도 upstream 경로에서 받아야 한다.

## 7. 환경 분리 원칙

- `hades`: 구형 LLaVA 코드와 PixArt/diffusers 호환 환경.
- `himrd`: SD3 및 각 legacy VLM별 환경. Qwen-VL/DeepSeek/Yi/MiniGPT는 의존성이 충돌하므로 하나의 Python 환경에 억지로 합치지 않는다.
- `mmplan`: 저장소의 VERL/transformers 환경과 단순 inference 환경을 분리한다.
- `memjack`: 최신 Qwen3-VL, Qwen3Guard, FAISS 환경.
- `agent32b`: Qwen2.5-VL-32B OpenAI-compatible server 환경.

각 환경은 import smoke test → 한 모델 load test → 샘플 1건 종단간 test를 통과한 뒤 확대 실행 대상으로 승격한다.
