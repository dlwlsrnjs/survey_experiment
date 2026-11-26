# 근거 원장

| 작업 사실 | 값 | 출처 |
|---|---|---|
| 1차 보조 모델 | Qwen/Qwen2.5-VL-32B-Instruct | 사용자의 현재 요청(로컬 32B 우선) |
| 후속 비교 모델 | GPT-4o | 사용자의 현재 요청(나중에 별도 실행) |
| HarmBench 표본 | 개발 20, 평가 50, 의미 범주 층화 | 실험 계획서 PDF p. 1 |
| MM-SafetyBench 표본 | 5개 범주별 개발 4, 평가 10 | 실험 계획서 PDF p. 1 |
| 이미지 조건 | 동일 목표의 SD와 SD_TYPO를 같은 분할에 배치 | 실험 계획서 PDF p. 1 |
| 파일럿 | 자료별 목표 10개 | 실험 계획서 PDF pp. 1, 5 |
| 고정 추적 사례 | MM-SafetyBench `06-Fraud`, ID `6`을 파일럿 개발 분할에 포함 | 실험 계획서 PDF p. 5 |
| 평가자 | 모델명을 가린 2명과 별도 조정본 | 실험 계획서 PDF p. 3 |
| MemJack 조건 | A/B/C 및 C_random, 3회 반복, 질의 상한 3 | 실험 계획서 PDF p. 4 |
| MemJack 공개 궤적 | 5,959 trajectories, 55,867 rounds | 첨부된 공개 MemJack-Bench의 실제 파일 행 수 |
| MM-SafetyBench 이미지 배포본 | ZIP SHA-256 `7e1a9f08212038ceaa74801964d2b97c219509663bbd017ab77abb1b063cc23e` | 공식 GitHub README의 Google Drive 배포본을 내려받아 계산 |

결과 수치, 거부율, 품질 분포, 비용은 아직 실행 산출물이므로 문서에서 미리 채우지 않는다.

