# 프로젝트 작업 지침

이 저장소는 VLM 보조 에이전트의 역할별 중간 산출물을 비교하는 AI 안전성 연구 프로젝트다.

작업을 시작하기 전에 다음 문서를 순서대로 읽는다.

1. 새 머신이나 환경 문제면 `docs/SETUP_KO.md`
2. `docs/SESSION_HANDOFF_KO.md`
3. `docs/EXPERIMENT_AND_INTERMEDIATE_ANALYSIS_KO.md`
4. 필요한 경우 `docs/PROTOCOL_ADHERENCE_KO.md`와 `docs/CODE_ANALYSIS_KO.md`

운영 원칙:

- `runs/`의 JSONL과 `annotations/`의 사람평가 파일은 연구 원자료다. 명시적 요청 없이 삭제하거나 덮어쓰지 않는다.
- 부분 실행과 자동 판정을 최종 ASR 또는 모델 우열로 표현하지 않는다.
- 모델의 비공개 chain-of-thought를 분석 대상으로 삼지 않는다. 사전 정의된 구조화 중간 산출물과 실제 입력·출력 trace만 분석한다.
- 거부, 후보 거부, API 오류, 형식 오류, 사람 품질을 분리해 집계한다.
- `SD`와 `SD_TYPO`는 같은 `goal_id`의 짝지어진 조건이다.
- 새 실험은 기존 `experiment_id`와 출력 디렉터리를 재사용할지 먼저 판단하고, resume 목적이 아니면 새 ID를 사용한다.
- 완료 여부는 `outputs.jsonl` 행 수만으로 판단하지 말고 `completed.json`, provenance, 계획 호출 수, 사람평가 상태를 함께 확인한다.
- 외부 저장소·데이터·모델 revision은 `configs/assets.lock.json`, 이미지 생성기는 `configs/image_generators.yaml`을 기준으로 하며 임의로 최신 revision으로 바꾸지 않는다.
