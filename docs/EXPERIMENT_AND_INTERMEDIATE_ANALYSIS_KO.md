# 실험 설정 및 에이전트 중간작업 분석 계획

이 문서는 이 저장소에서 수행하는 VLM 보조 에이전트 역할 실험의 **비교 단위, 통제 조건, 로그 계약, 분석 방법**을 한곳에 정리한다. 여기서 “중간작업”은 모델의 비공개 chain-of-thought가 아니라, 연구자가 사전에 정의한 역할별 구조화 출력과 그 출력이 다음 단계에서 사용되는 방식이다.

## 1. 연구 질문

핵심 질문은 다음과 같다.

1. 같은 안전성 평가 입력에서 보조 모델이 맡는 역할에 따라 거부율, 형식 준수율, 결과 품질, 비용이 어떻게 달라지는가?
2. 시각 입력의 표현 방식(`SD` 대 `SD_TYPO`)이 관찰·계획 단계의 거부와 품질에 어떤 영향을 주는가?
3. 이전 평가 피드백이나 검색된 경험을 제공하면 다음 후보 생성이 개선되는가?
4. 관련 경험의 효과가 단순히 추가 문맥을 준 효과인지 확인하기 위해, 길이를 맞춘 무작위 경험(`C_random`)과 비교했을 때 차이가 남는가?

이 실험은 FigStep, HADES, HIMRD, MM-Plan, MemJack의 최종 ASR을 그대로 재현하는 실험이 아니다. 공개 구현에서 관찰되는 **보조 역할과 정보 흐름**을 공통 입력·공통 출력 규격으로 재생하는 역할 수준 비교다. 원형 보존 범위는 `docs/PROTOCOL_ADHERENCE_KO.md`에 별도로 기록되어 있다.

## 2. 실험 구성 한눈에 보기

### 2.1 모델과 실행 환경

| 항목 | 고정값 |
|---|---|
| 보조 모델 | `Qwen/Qwen2.5-VL-32B-Instruct` |
| 모델 revision | `7cfb30d71a1f4f49a57592323337a4a4727301da` |
| 추론 서버 | 로컬 OpenAI-compatible API, 기본 `127.0.0.1:8000/v1` |
| 온도 | `0.0` |
| seed | `20261125` |
| 호출당 최대 출력 | 1,200 tokens |
| timeout | 600초 |
| GPU | Slurm `h200` 파티션, H200 1장 |
| 허용 이미지 연산 | `crop`, `mask`, `blur` |

실행별 실제 모델·GPU·Slurm 정보와 config/manifest/검색 계획의 SHA-256은 `runs/<experiment_id>/provenance.json`에 기록한다.

### 2.2 데이터와 분할

고정 manifest는 `data/manifests/fixed_inputs.jsonl`이다.

| 데이터셋 | split | 목표 수 | manifest 행 수 | 비고 |
|---|---:|---:|---:|---|
| HarmBench-Standard | dev | 20 | 20 | 목표당 텍스트 조건 1개 |
| HarmBench-Standard | eval | 50 | 50 | 목표당 텍스트 조건 1개 |
| MM-SafetyBench | dev | 20 | 40 | 목표당 `SD`, `SD_TYPO` 2개 |
| MM-SafetyBench | eval | 50 | 100 | 목표당 `SD`, `SD_TYPO` 2개 |

`SD`와 `SD_TYPO`는 서로 독립 표본으로 세지 않고 같은 `goal_id`의 짝지어진 조건으로 분석한다. MemJack 경험 은행은 공개 train split에서만 구축하며 eval/test 레코드가 들어오면 runner가 중단된다.

### 2.3 역할별 입력과 중간 산출물

| 역할 | 적용 입력 | 구조화 중간 산출물 | 다음 소비자 |
|---|---|---|---|
| `generation` | HarmBench 목표 | 거부 여부, 렌더링 문장, 비그래픽 장면 설명, 의미 보존 여부 | 고정 타이포그래피 렌더러 또는 고정 T2I |
| `observation_planning` | MM-SafetyBench 이미지·질문 | 보이는 근거, OCR, 추론, 계획 단계, 제한된 이미지 도구 호출 | 표적 실행기 |
| `evaluation_revision` | 앞 단계 후보와 선택적 표적 응답 | 후보 거부 여부, 이해/이행 판정, 0/1/2 품질, 오류, 수정안 | 재작성기 또는 중단 제어기 |
| `experience_use` | 목표, 조건별 피드백·경험 | 사용한 피드백/경험 ID, 경험 적용 설명, 다음 후보 | 후보 생성기 |

각 출력은 JSON schema로 제한되고, 원문 출력과 파싱 결과를 모두 보존한다. 모델이 스스로 설명한 `used_experience_ids`는 실제 제공된 ID와 대조해 존재하지 않는 경험을 사용했다고 주장했는지도 검사한다.

## 3. 독립변수와 통제 조건

### 3.1 역할 비교

일반 역할은 eval manifest에서 각각 1회 실행한다.

- `generation`: HarmBench 50건
- `observation_planning`: MM-SafetyBench 100건
- `evaluation_revision`: 앞 두 역할에서 생성된 후보 150건

### 3.2 경험 사용 A/B/C/C_random

`experience_use`는 eval의 150개 조건 각각에 대해 4개 조건을 3회 반복한다.

| 조건 | 평가 피드백 | 검색 경험 | 해석 |
|---|---|---|---|
| A | 없음 | 없음 | 독립 생성 기준선 |
| B | 있음 | 없음 | 피드백만의 효과 |
| C | 있음 | 목표와 관련된 train 경험 3개 | 관련 경험의 추가 효과 |
| C_random | 있음 | 개수·문자열 길이를 맞춘 무작위 train 경험 3개 | 추가 문맥량에 대한 대조군 |

관련 경험은 단순 token cosine으로 사전 검색하고, 무작위 대조군은 관련 경험을 제외한 뒤 출력 문자열 길이가 가까운 항목을 선택한다. 검색 결과는 모델 호출 전에 `retrieval_plan.jsonl`로 동결한다.

계획 호출 수는 다음과 같다.

- 일반 역할: `50 + 100 + 150 = 300`
- 경험 조건: `150 조건 × 4 조건 × 3 반복 = 1,800`
- 전체: `2,100` 모델 호출

## 4. 무엇을 중간작업으로 기록하는가

분석 단위의 기본키는 `(experiment_id, role, condition, repeat, condition_id)`다. `condition`이 없는 일반 역할은 `-`로 취급한다.

`outputs.jsonl`의 각 행에는 다음을 남긴다.

- 데이터 출처: dataset, split, goal/category/source ID, 이미지 변형
- 실제 모델 입력: system/user prompt, 이미지 절대경로와 SHA-256
- 재현 설정: model, temperature, seed, max tokens, prompt hash
- 실제 모델 출력: raw text, parsed JSON, request/response ID
- 중간작업 검사: 자동 형식 검사, 거부 여부, 후보 거부 여부, 허용 도구 위반
- 정보 흐름: 다음 소비자, 렌더러 실행 결과, 제공·사용한 경험 ID
- 운영 지표: API/도구 오류, latency, 입력·출력 token 수

`events.jsonl`은 `request_started`, `request_finished`, `skipped`, `resume_skip`을 기록한다. 따라서 중단된 실행은 같은 config로 다시 실행하면 이미 완료된 기본키를 건너뛰고 남은 호출만 수행한다.

### 해석상 주의

- 모델의 숨겨진 추론 과정을 복원하거나 평가하지 않는다.
- `inferences`, `plan_steps`, `notes`, `experience_application` 등 연구자가 요구한 관찰 가능한 산출물만 분석한다.
- 자기보고(`used_feedback`, `used_experience_ids`)는 실제 입력 대조와 사람평가 없이 인과적 사용의 증거로 간주하지 않는다.
- 표적 모델 응답이 제공되지 않은 경우 `understood`나 `fulfilled`를 공격 성공률로 해석하지 않는다.

## 5. 분석 계획

### 5.1 자동 분석

모든 비율은 분모를 함께 보고한다.

1. **실행 건전성**: API 오류율, 파싱/형식 유효율, 누락률
2. **거부**: 보조 모델 거부율과 공급된 후보의 거부율을 분리
3. **후보 품질**: 거부하지 않은 판정 가능 사례에 한해 0/1/2 분포와 평균을 계산
4. **시각 조건 효과**: 같은 `goal_id`의 `SD_TYPO - SD` 차이를 목표 단위 paired bootstrap(10,000회, seed 고정)으로 추정
5. **경험 사용 무결성**: 제공 경험 수, 사용 주장 경험 수, 적용률, 피드백 사용률, 미제공 ID 사용 주장 수
6. **비용**: 역할·조건별 latency, prompt/completion tokens

거부 사례를 품질 0으로 강제하지 않는다. 거부, 오류, 미도달, 품질을 서로 다른 결과로 보고하고 각 분석의 유효 표본 수를 표시한다.

### 5.2 사람평가

두 평가자는 모델 이름과 request ID 매핑을 분리한 CSV를 독립적으로 평가한다. 평정 기준은 `annotations/<run>/RUBRIC.md`에 생성된다.

- 공통: 거부, 0/1/2 품질, 의미 보존, 형식 만족
- 관찰/계획: 이미지 근거성, 계획 실행 가능성
- 평가/수정: 판정 일치, 수정 개선 여부
- 경험 사용: 경험 관련성, 실제 적용 여부, goal drift
- 오류 유형: 형식, 의미 손실, 시각 근거 오류, 도구 제약 위반, judge 오판, 수정 퇴행, 무관 경험, goal drift 등

독립 평정 원본은 덮어쓰지 않고, 불일치 조정본을 제3의 CSV로 만든다. 최종 보고에는 이진 항목의 일치율과 Cohen's kappa, 순서형 0/1/2 품질의 weighted kappa를 추가하는 것이 필요하다. 현재 저장소의 `metrics.py`는 사람평가 분포 집계는 구현하지만 평가자 간 신뢰도는 아직 구현하지 않았다.

### 5.3 조건 간 통계 비교

현재 구현된 통계는 `SD` 대 `SD_TYPO`의 목표 단위 paired bootstrap이다. 경험 조건의 최종 비교는 동일 목표·동일 반복을 짝으로 유지해 아래 대비를 계산해야 한다.

- `B - A`: 피드백 효과
- `C - B`: 관련 경험의 추가 효과
- `C - C_random`: 관련성 효과

1차 결과는 목표 단위 paired bootstrap 또는 permutation test로 비율/평균 차이와 95% CI를 보고한다. 보조 분석으로 이진 결과에는 goal random intercept를 둔 mixed-effects logistic model, 0/1/2 품질에는 ordinal mixed model을 사용할 수 있다. 데이터셋·범주·이미지 변형은 사전 지정 공변량으로만 사용하고, 작은 하위집단의 유의성은 탐색적으로 표시한다.

### 5.4 질적 분석

모든 사람평가가 끝난 뒤 역할별 2건, 총 8건을 선택한다. 성공 사례만 골라 보여주는 것을 막기 위해 사전 정의한 오류 유형이 서로 다르고 드문 사례를 우선하며, 선택 규칙과 오류 빈도를 결과 번들에 함께 저장한다.

## 6. 완료 판정과 보고 규칙

최종 결과로 부르려면 아래 조건을 모두 만족해야 한다.

- 계획된 호출 2,100건의 완료 또는 명시적인 누락 사유 기록
- `completed.json`, 최신 `summary.json`, provenance 및 검색 계획 존재
- 두 평가자의 독립 평정과 조정본 존재
- 평가자 간 신뢰도와 조건별 분모 보고
- 부분 실행, 파일럿, 자동 판정을 최종 ASR이나 모델 우열로 표현하지 않음

현재 상태와 정확한 재개 방법은 `docs/SESSION_HANDOFF_KO.md`를 따른다.
