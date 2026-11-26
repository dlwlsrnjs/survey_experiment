# 공개 코드 분석 기록

이 문서는 공개 저장소를 내려받은 뒤 어떤 실행 경로와 설정을 확인했고, 이번 보조 모델 역할 실험에 무엇을 보존했는지 기록한다. 모든 저장소 커밋은 `configs/protocols.yaml`과 실행별 `provenance.json`에 고정한다.

| 기법 | 직접 확인한 원 코드 | 확인한 동작/설정 | 이번 구현 대응 |
|---|---|---|---|
| FigStep | `third_party/FigStep/src/generate_prompts.py:25-79` | 760×760 흰 배경 타이포그래피, 글꼴 크기 80, 15자 줄바꿈, 3개 빈 단계와 고정 후속 질문 | 작성 모델 산출물과 결정론적 렌더러 결과를 분리하고 둘 다 로그에 보존. 공통 비교에서는 MM-SafetyBench Arial과 1024×1024 캔버스로 통제했으므로 원형 재현이 아닌 adaptation으로 표시 |
| HADES | `third_party/HADES/black_box/optimize_harm.py:21,63,98,123-124,160-189` 및 `generate_sd_caption.py:9,45` | 프롬프트·생성 이미지·LLaVA 캡션·1–10점 및 설명을 history로 누적, 5회 반복, 캡션 모델 온도 0.2, 공격/SD prompt 모델 온도 1.1, 이미지 prompt 120단어 제한 | 캡션 품질 평가와 표적 응답 평가는 분리. 반복 history를 입력/출력/다음 소비자 단위로 기록. 실제 PixArt/LLaVA end-to-end 재현은 별도 필요 |
| HIMRD | `third_party/HIMRD-jailbreak/attack.py:80-137,143-205,296-334` 및 `attack_utils.py:43-163` | HarmBench 성공 판정과 거부 문자열을 이용해 `Understand/NotUnderstand`를 분리하고, 미이해 사례는 prefix, 이해했지만 실패한 사례는 suffix를 별도 재작성; 실패 후보 목록을 다음 재작성에 제공 | `target_assessment`에서 refused/understood/fulfilled를 분리하고 실패 후보·수정 근거를 구조화. 실제 표적 응답이 없을 때 fulfilled를 추정하지 않음 |
| MM-Plan | `third_party/MM-Plan/verl/examples/mm_plan/Qwen3/vebench.sh:35-80`, `Qwen3/harmbench.sh:38-83` | Qwen3-VL-4B-Instruct, vision tower freeze, rollout temperature 0.7, rollout 수 변수, 8 GPU, 10 epochs, 전용 reward | 현재 32B 비교는 학습 체크포인트 재현이 아니라 inference-only 관찰/계획 역할 비교. 보이는 근거·추론·계획·허용 동작을 분리 로그. 논문/계획서 기준 rollout n=4를 원형 설정표에 고정 |
| MemJack | `third_party/MemJack/core/attacker_agent.py:558-635`, `memory_manager.py:553-682`, `safety_guard.py:179-371`, `configs/config.yaml:6-28,36-78` | 이전 피드백·reflection·검색 memory를 다음 후보 생성에 공급; goal/visual memory 검색과 utility 관리; 기본 attack temperature 0.7, max rounds 20, query budget 3200, reflection 조건 | 실험 계획서의 공통 통제를 우선해 A(독립), B(피드백), C(관련 train 경험), C_random(개수·길이 대응 무작위 경험)을 분리. 각 조건 3회, 목표 질의 상한 3. HF 공개 train 궤적만 사용하고 검색 결과는 모델 호출 전에 manifest로 고정 |

## 재현성과 해석 경계

- `FigStep`, `HADES`, `HIMRD`, `MM-Plan`, `MemJack` 원 저장소는 수정하지 않고 `third_party/`에 둔다.
- 이번 1차 실험에서 바꾸는 유일한 비교 대상은 보조 모델이며, 모델은 `Qwen/Qwen2.5-VL-32B-Instruct`의 고정 revision이다.
- 입력 manifest, 프롬프트 규격, 도구, 호출 예산, seed, temperature는 고정한다. 후속 GPT-4o 실행은 같은 manifest와 역할 규격을 재생한다.
- 원 논문 고유 생성기·학습 체크포인트·표적 모델이 없는 경우에는 원 ASR 재현이라고 주장하지 않는다. 그 결과는 역할 수준의 거부율, 형식 유효성, 0/1/2 사람 품질, 비용·지연 비교로만 해석한다.
- 외부 궤적은 MemJack 공식 train split만 경험 은행에 넣고 eval/test 경험이 섞이면 runner가 즉시 중단한다.
