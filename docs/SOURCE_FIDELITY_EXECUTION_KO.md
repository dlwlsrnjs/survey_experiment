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
| HIMRD | 공개 `data/SafeBench/SafeBench.txt` | 350개 목표 | 앞 200개 고유 목표의 GPT 역할 실행 중 |
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

## 현재 본 실행

Slurm `59810`은 HIMRD 원 SafeBench의 앞 200개 고유 목표를 대상으로 다음 세 GPT 역할을 실행한다.

1. SD3 장면 설명 생성
2. 이해 실패 시 prefix 재작성
3. 이해했지만 실패한 경우의 suffix 재작성

계획 레코드는 600개다. 현재 작업은 GPT 역할의 출력과 token·비용을 먼저 확보하며 `--no-image-generation`으로 실행된다. SD3 Medium 체크포인트가 완전하게 준비되고 표적 VLM·HarmBench 분기가 연결되기 전에는 이를 HIMRD end-to-end 재현으로 부르지 않는다.

실행 파일은 `slurm/run_himrd_gpt4omini_safebench200.sbatch`, 출력은 `runs/himrd_gpt4omini_safebench200_20261126`이다. OpenAI 키는 저장소에 두지 않고 권한 `600`의 외부 파일에서만 읽는다.
