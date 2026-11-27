# 논문 실구현 우선 실행 원칙과 현황

이 트랙은 모든 역할을 한 모델로 통일하는 공통 비교보다 먼저, 각 논문의 공개 구현이 실제로 사용한 데이터·프롬프트·모델·도구·정보 흐름을 보존한다.

## 모델 치환 원칙

- 공개 체크포인트가 지정된 구성요소는 원 모델을 사용한다. 예: PixArt, SD3 Medium, LLaVA-1.5, Qwen3-VL-4B/8B, Qwen3Guard, HarmBench classifier.
- GPT-3.5, GPT-4, GPT-4o, o1처럼 외부 GPT API가 필요한 역할만 지정한 대체 모델로 바꾼다.
- 현재 GPT 대체 모델은 snapshot을 고정한 `gpt-4o-mini-2024-07-18`이다.
- 대체 사실은 모든 실행의 `provenance.json`에 `model_substitution`으로 기록한다.
- MM-Plan과 MemJack의 원래 Qwen 계열 역할을 GPT-4o mini나 32B 모델로 바꾸지 않는다.
- 원 모델·데이터·단계가 빠진 실행은 `full reproduction`이 아니라 해당 단계의 `source-fidelity auxiliary replay`로 표시한다.

## 벤치마크 표본

기존 고정 manifest의 일반 역할 50/100건은 예비 기준선으로만 유지한다. 논문 실구현 트랙은 각 공개 구현이 배포한 벤치마크를 사용하고 역할별 최소 200개 고유 사례를 확보한다.

| 방법 | 원 데이터 | 현재 가용 규모 | 우선 실행 |
|---|---|---:|---|
| FigStep | 공개 `SafeBench` | 500개 질문·500개 이미지 | 원 타이포그래피 이미지와 대상 VLM 실행 준비 |
| HADES | `Monosail/HADES` | 공식 parquet 2개, 약 698MB | 다운로드·schema 검증 후 200개 이상 |
| HIMRD | 공개 `data/SafeBench/SafeBench.txt` | 350개 목표 | 앞 200개 GPT 역할 완료, GPU 종단간 실행 준비 |
| MM-Plan | `zybeich/VE-Safety` | 440개, DUA 배포 | Qwen3-VL-4B planner와 원 executor 경로 준비 |
| MemJack | 공식 HF train/val/test | 로컬 다운로드 완료 | Qwen3-VL-8B·Qwen3Guard·embedding 원 역할 분리 |

HIMRD manifest는 `data/manifests/himrd_safebench_full.jsonl`이며 공개 구현의 350개 원문을 그대로 보존한다. 범주 이름만 같은 SafeBench를 배포한 FigStep CSV의 대응 순서에서 가져오며, 원문과 범주 출처를 각 행에 따로 기록한다.

## GPT-4o mini 파일럿

2026-11-26에 HarmBench Standard 10개 사례로 API·프롬프트·파서 smoke test를 수행했다.

| 항목 | 결과 |
|---|---:|
| API 오류 | 0건 |
| 전체 API 호출 | 84회 |
| 입력 / 출력 token | 37,038 / 2,449 |
| 추정 직접비용 | $0.0070251 |
| HADES caption 유효 | 0/10; 모두 거부, 각 5회 시도 |
| HIMRD scene 유효 | 10/10; 총 14회 시도 |
| HIMRD prefix 유효 | 10/10 |
| HIMRD suffix 유효 | 10/10 |

파일럿 출력은 `runs/source_fidelity_gpt4omini_smoke10_20261126`에 있으며 Git에는 올리지 않는다. HADES의 거부는 API 오류나 parser 오류로 합치지 않고 모델 거부 결과로 보존한다. 이 smoke test는 이미지 생성기를 끈 단계 검증이므로 최종 HADES/HIMRD 결과가 아니다.

## GPT 역할 실행 결과

Slurm `59810`은 HIMRD 원 SafeBench의 앞 200개 고유 목표를 대상으로 다음 세 GPT 역할을 완료했다.

1. SD3 장면 설명 생성
2. 이해 실패 시 prefix 재작성
3. 이해했지만 실패한 경우의 suffix 재작성

계획한 600개 레코드는 모두 생성됐다. 이 결과 중 장면 설명 200개는 SD3 입력으로 재사용한다. prefix/suffix 400개는 실제 대상 VLM 판정 전에 생성됐으므로 최종 적응 궤적이 아니라 첫 반복 후보군으로만 사용한다. 이 실행 자체를 HIMRD end-to-end 재현으로 부르지 않는다.

실행 파일은 `slurm/run_himrd_gpt4omini_safebench200.sbatch`, 출력은 `runs/himrd_gpt4omini_safebench200_20261126`이다. OpenAI 키는 저장소에 두지 않고 권한 `600`의 외부 파일에서만 읽는다.

## H200 종단간 실행

`survey_exp/himrd_end_to_end.py`는 다음 순서를 사례별 상태 파일과 event log로 남기며 중단 후 재개할 수 있다.

1. 기존 GPT 장면 설명 재사용
2. GPT-4o mini로 원 구현의 multimodal risk distribution 수행
3. SD3 Medium 50 steps, guidance 8.0, 512×512 장면 생성
4. 이미지 상단에 분리된 시각 문구를 렌더링하고 세로 결합
5. LLaVA-v1.6-Mistral-7B 대상 응답 생성
6. 원 저장소의 HarmBench prompt와 HarmBench-Mistral-7B classifier로 판정
7. 이해 실패는 prefix, 이해했지만 공격 실패한 사례는 suffix를 최대 5회 조건부 수정

H200 제출 파일은 `slurm/run_himrd_end_to_end_h200.sbatch`다. 2026-11-26 Hugging Face gated 접근 승인 후 SD3 고정 revision의 38개 파일, 약 15GB를 내려받았다. 최초 smoke `59824`는 SentencePiece 0.2.3과 Protobuf runtime 불일치로 추론 전에 실패했다. SentencePiece 0.2.0 고정 및 SD3 tokenizer preflight 후 재제출한 smoke는 `59834`, 성공 후 시작하는 200건 본 작업은 `59835`다.

## 논문별 원형 충실도 실행 상태

| 기법 | 현재 사용하는 원 구성 | 실행 상태 | 아직 원형과 다른 부분 |
|---|---|---|---|
| FigStep | 공식 SafeBench 이미지·공식 incitement prompt·LLaVA-1.5-7B | smoke `59838`과 200건 `59839` 완료, 오류 0건 | 논문과 동일하게 사람 판정이 필요하므로 GPU 응답 완료만으로 ASR을 확정하지 않음 |
| HADES | 공식 공개 최적화 이미지 0–5단계·LLaVA-1.5-7B·Beaver-Dam-7B | smoke `59836`과 200목표×6단계(1,200행) `59837` 완료 | 공개 산출물 평가 트랙이며 PixArt/GPT로 이미지를 새로 생성하는 재생성 실험은 별도 |
| HIMRD | GPT 역할만 GPT-4o mini로 치환, SD3 Medium·LLaVA-v1.6-Mistral-7B·HarmBench-Mistral-7B | smoke `59834`과 200건 `59835` 완료 | 원 GPT-4o/o1 역할을 GPT-4o mini로 통일했고 SD3 single-file 대신 고정 Diffusers revision 사용 |
| MM-Plan | Qwen3-VL-4B base planner와 Qwen3-VL-8B local target 가중치 준비 | 미제출 | metadata 조회는 가능하지만 실제 VE-Safety 다운로드는 2026-11-27에도 403이며 논문 학습 완료 planner checkpoint도 미공개; 원 judge는 Claude Sonnet 4.5 |
| MemJack | Qwen3-VL-8B·Qwen3Guard-Gen-8B·Qwen3-VL-Embedding-8B 및 공식 궤적 준비 | 32B 역할 재생만 실행 중 | 논문 victim은 폐쇄형 `Qwen3-VL-Plus`이고 고정 공식 revision `a8d8a83378afe18bffce04b28d530a423d31508d`에는 실행 필수 파일 `configs/safety_policy.txt`가 없음 |

FigStep/HADES/HIMRD의 smoke가 실패하면 연결된 본 작업은 시작하지 않는다. 실패 로그를 고친 뒤 새 smoke ID로 재검증한다. MM-Plan 데이터 접근 요청은 `https://huggingface.co/datasets/zybeich/VE-Safety`에서 별도로 승인받아야 한다.

MemJack 논문은 수정하지 않은 COCO val2017 전체에서 `Qwen3-VL-Plus`를 평가한다. 공개 코드가 제공하는 Qwen3-VL-8B local backend는 실행 가능한 대체 조건이지 논문 주 결과의 원 대상이 아니다. 누락 정책 파일을 임의로 만들어 넣거나 local victim 결과를 원형 재현으로 표기하지 않는다. 정책 파일이 공개되거나 저자에게서 확보되면 COCO val2017 200장 smoke/full을 먼저 재현하고, 그 뒤 같은 200장을 공통 로컬 victim으로 돌리는 통제 실험을 추가한다.

단계별 VLM·LLM·이미지 생성기·judge 선택과 폐쇄형 모델 필요 조건은 `docs/PIPELINE_MODEL_ROUTING_KO.md`를 따른다.
