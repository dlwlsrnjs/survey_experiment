# VLM 보조 모델 역할 실험

첨부 실험 계획을 실행 가능한 형태로 옮긴 저장소다. 1차 보조 모델은 로컬 `Qwen/Qwen2.5-VL-32B-Instruct`, 후속 재생은 같은 manifest와 프롬프트 해시를 사용하는 GPT-4o로 구성했다.

처음 보는 연구자나 새 작업 세션은 다음 두 문서부터 읽는다.

- [실험 설정 및 에이전트 중간작업 분석 계획](docs/EXPERIMENT_AND_INTERMEDIATE_ANALYSIS_KO.md)
- [현재 상태와 세션 인수인계](docs/SESSION_HANDOFF_KO.md)

핵심 산출물:

- `third_party/`: FigStep, HADES, HIMRD, MM-Plan, MemJack, HarmBench, MM-SafetyBench 공식 공개 코드의 고정 커밋.
- `data/manifests/fixed_inputs.jsonl`: HarmBench 개발 20/평가 50과 MM-SafetyBench 개발 20/평가 50. MM 데이터는 목표마다 SD/SD_TYPO 두 행이지만 통계 단위는 하나의 `goal_id`다.
- `data/manifests/memjack_train_experience.jsonl`: 공개 MemJack-Bench의 공식 train ID만 사용한 45,016개 라운드 경험 은행.
- `survey_exp/`: 모델 호출, 역할별 JSON 규격, 자동 검사, 검색, 로그, 블라인드 사람 평가, 집계 코드.
- `configs/protocols.yaml`: 논문/공개 구현에서 보존한 설정과 통제 실험용 변형의 명시적 구분.
- `slurm/run_local32b_pilot.sbatch`: H200 한 장에서 32B VLM 서버를 띄우고 자료별 목표 10개 파일럿을 실행하는 작업.
- `slurm/run_local32b_pilot_v2.sbatch`: v1에서 발견한 임의 version 키와 후보/평가자 거부 혼동을 수정한 동결 후보 규격의 재검증 작업.

## 실행 순서

```bash
python3 -m survey_exp.dataset --root "$PWD"
python3 -m survey_exp.memjack --dataset-root data/raw/memjack \
  --output data/manifests/memjack_train_experience.jsonl --split train
python3 -m unittest discover -s tests -v
sbatch slurm/run_local32b_pilot.sbatch
```

파일럿 완료 뒤:

```bash
python3 -m survey_exp.annotation \
  --run-dir runs/local32b_pilot_20261125 \
  --out-dir annotations/local32b_pilot_20261125
python3 -m survey_exp.metrics --run-dir runs/local32b_pilot_20261125
```

전체 평가에서는 생성·관찰/계획·평가/수정은 고정 입력당 1회, MemJack 경험 사용의 A/B/C/C_random 조건만 각각 3회 반복한다. 역할별 반복 수는 config의 `role_repeats`에 명시되어 있어 GPT-4o 재생에서도 그대로 고정된다.

파일럿의 형식 유효성·지연·오류를 확인한 뒤 전체 32B 평가는 `sbatch slurm/run_local32b_full.sbatch`로 실행한다. 전체 실행은 2일 제한으로 잡고, 이미 완료된 호출은 `outputs.jsonl`을 기준으로 건너뛰므로 같은 명령으로 안전하게 재개할 수 있다.

두 평가자는 `annotations_*.csv`를 독립적으로 작성한다. 모델 이름 매핑은 `blind_key.json`에 분리되어 있다. 조정본이 생기면 `survey_exp.qualitative`로 역할별 대조 사례 두 개, 총 여덟 개를 오류 유형 분포에 따라 선택한다.

## 로그 계약

모든 호출은 `events.jsonl`과 `outputs.jsonl`에 요청 시작/종료, 모델·역할·조건, 목표 ID, 실제 system/user 입력, temperature/seed/max tokens, 이미지 경로와 SHA-256, 프롬프트 해시, 토큰 수, 지연, 원문 출력, 파싱 결과, 자동 형식 검사, API/도구 오류를 남긴다. API 키와 인증 헤더는 기록하지 않는다. `provenance.json`에는 config/manifest/retrieval-plan SHA-256, 공개 코드 커밋, Slurm job 정보가 들어간다.

세부적인 원형 보존/변형 판단은 [프로토콜 대응표](docs/PROTOCOL_ADHERENCE_KO.md), 공개 코드별 확인 위치는 [공개 코드 분석 기록](docs/CODE_ANALYSIS_KO.md), 근거 원장은 [출처 원장](docs/SOURCE_LEDGER_KO.md)에 있다.

로컬 32B 일반 역할 300건의 자동 중간 결과는 [로컬 32B 중간 결과](docs/PRELIMINARY_LOCAL32B_RESULTS_KO.md)에 있으며, MemJack 및 사람 평정 완료 전에는 최종 결과로 사용하지 않는다.

