# VLM 보조 모델 역할 실험

VLM 안전성 연구에서 보조 모델이 맡는 생성, 관찰·계획, 평가·수정, 경험 사용 역할을 동일한 입력·출력 계약으로 비교하는 재현 가능한 실험 저장소다. 기본 보조 모델은 `Qwen/Qwen2.5-VL-32B-Instruct`이며, 같은 manifest와 역할 규격으로 다른 모델을 재생할 수 있다.

처음 보는 연구자나 새 작업 세션은 다음 문서부터 읽는다.

- [새 머신 설치 및 실행](docs/SETUP_KO.md)
- [실험 설정 및 중간작업 분석 계획](docs/EXPERIMENT_AND_INTERMEDIATE_ANALYSIS_KO.md)
- [현재 상태와 세션 인수인계](docs/SESSION_HANDOFF_KO.md)

## 새 머신 빠른 시작

필수 조건은 Linux, Git, Python 3.10 이상, CUDA GPU다. 비공개 저장소 접근권한과 Hugging Face 인증이 필요하다. MemJack 데이터와 `full` 프로필의 SD3 모델은 해당 배포 페이지에서 접근 조건을 먼저 수락해야 한다.

```bash
git clone https://github.com/dlwlsrnjs/survey_experiment.git
cd survey_experiment

export HF_TOKEN='YOUR_HUGGINGFACE_TOKEN'
# 대용량 공용 캐시를 쓰려면 선택적으로 지정
export HF_HOME=/path/to/huggingface-cache

./bootstrap.sh --profile core
```

`core`는 현재 통제 실험에 필요한 32B 보조 모델, SDXL 이미지 생성기, 데이터셋, 외부 저장소, Python 환경을 모두 준비한다. 원형 충실도 모델과 legacy 환경까지 받으려면 다음을 사용한다.

```bash
./bootstrap.sh --profile full
```

예상 여유 공간은 `core` 최소 약 100GB, `full` 약 350GB 이상이다. Hugging Face cache 공유 여부와 파일시스템의 hardlink 지원에 따라 달라진다. 세부 다운로드 목록과 고정 revision은 `configs/assets.lock.json`이 유일한 기준이다.

설치 상태 확인:

```bash
.venvs/modern/bin/python -m survey_exp.doctor --profile core --strict
```

## 이미지 생성 프로필

실험에서 이미지 생성기를 암묵적으로 바꾸지 않는다. `configs/image_generators.yaml`에 두 계열을 분리해 고정했다.

| 용도 | 모델 | 실제 사용 위치 |
|---|---|---|
| 공통 통제 `controlled_default` | `stabilityai/stable-diffusion-xl-base-1.0` | 일반 역할 비교와 GPT-4o 재생의 `generation` 결과 |
| HADES 원형 충실도 | `PixArt-alpha/PixArt-XL-2-1024-MS` | `source_fidelity_runner`의 HADES diffusion prompt |
| HIMRD 원형 충실도 | `stabilityai/stable-diffusion-3-medium-diffusers` | `source_fidelity_runner`의 HIMRD scene prompt |

모델 revision, dtype, seed, step, guidance, 해상도는 config와 결과 로그에 기록된다. 일반 역할 비교는 항상 `controlled_default`를 사용하며, 원형 충실도 실행은 기법별 원 생성기를 사용한다.

## 실행

로컬 GPU 파일럿:

```bash
bash scripts/run_local32b.sh configs/local32b_pilot_imagegen.yaml
```

Slurm 파일럿 및 전체 평가:

```bash
sbatch slurm/run_local32b_pilot.sbatch
sbatch slurm/run_local32b_full.sbatch
```

2026-11-25에 시작한 기존 393/2,100건 실행은 설정 SHA를 보존하기 위해 이미지 생성 설정을 추가하지 않았다. 그 실행만 `sbatch slurm/resume_local32b_20261125.sbatch`로 재개한다. 신규 실행은 위의 `*_imagegen.yaml` 설정을 사용한다.

원형 충실도 역할·이미지 생성 실행은 `full` 설치 후 수행한다.

```bash
bash scripts/run_source_fidelity.sh
# 또는
sbatch slurm/run_source_fidelity.sbatch
```

로컬 실행기는 모델 서버 기동, runner, 블라인드 평정지 생성, 자동 집계를 순서대로 수행한다. 전체 실행은 이미 완료된 `(role, condition, repeat, condition_id)`를 건너뛰므로 같은 experiment ID로 안전하게 재개할 수 있다.

## 저장소 구조

- `bootstrap.sh`: 환경·외부 저장소·데이터·모델·검증의 단일 진입점
- `configs/assets.lock.json`: 저장소, 데이터, 모델 revision과 checksum
- `configs/image_generators.yaml`: 통제/원형 충실도 이미지 생성 설정
- `third_party/`: 7개 공개 구현의 고정 commit checkout
- `data/manifests/fixed_inputs.jsonl`: HarmBench 개발 20/평가 50, MM-SafetyBench 개발 20/평가 50의 동결 입력
- `data/manifests/memjack_train_experience.jsonl`: 공개 train split의 45,016개 경험 레코드
- `survey_exp/`: 실행, 역할 schema, 경로 해석, 이미지 생성, 자동 검사, 사람평가, 집계, doctor
- `scripts/`: 로컬 환경 준비 및 실행기
- `slurm/`: 경로 독립적인 클러스터 래퍼

원시 데이터, 모델 cache, 실행 trace, 사람평가 파일, 외부 저장소 checkout은 Git에 올리지 않는다. 부트스트랩이 공개 배포본에서 다시 만들며 checksum과 revision을 검사한다.

## 로그와 해석 경계

모든 호출은 실제 system/user 입력, 이미지 SHA-256, 모델·역할·조건, seed/temperature, token 수, 지연, 원문·파싱 출력, 자동 검사와 도구 결과를 JSONL에 남긴다. 이미지 생성 결과에는 생성 모델 revision과 artifact SHA-256도 기록한다. API 키와 인증 헤더는 기록하지 않는다.

이 프로젝트의 중간작업 분석 대상은 사전 정의된 구조화 산출물이며 모델의 비공개 chain-of-thought가 아니다. 부분 실행과 자동 판정을 최종 ASR이나 모델 우열로 해석하지 않는다.

세부적인 원형 보존/변형 판단은 [프로토콜 대응표](docs/PROTOCOL_ADHERENCE_KO.md), 공개 코드별 확인 위치는 [공개 코드 분석 기록](docs/CODE_ANALYSIS_KO.md), 근거 원장은 [출처 원장](docs/SOURCE_LEDGER_KO.md)에 있다.
