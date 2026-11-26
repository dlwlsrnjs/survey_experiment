# 새 머신 설치 및 실험 실행 안내

이 문서는 빈 Linux 머신에서 저장소 clone 후 환경, 고정 외부 코드, 데이터셋, 모델을 준비하고 실험을 실행하는 절차를 설명한다.

## 1. 준비 사항

### 시스템

- Linux
- Git, Bash, curl
- Python 3.10 이상과 `venv` 모듈
- NVIDIA driver와 CUDA 사용 가능 GPU
- `core`: GPU 메모리 80GB 이상 권장, 디스크 여유 약 100GB 이상
- `full`: H200급 GPU 권장, 디스크 여유 약 350GB 이상

32B VLM과 이미지 생성기가 같은 GPU에서 사용되므로 작은 GPU에서는 `device_map`, offload 또는 별도 GPU 프로세스 구성이 추가로 필요하다. 현재 Slurm 설정은 H200 한 장을 기준으로 한다.

### 접근권한

1. 비공개 GitHub 저장소 clone 권한
2. Hugging Face 계정과 token
3. `senwuou/MemJack` gated dataset 접근 조건 수락
4. `full` 프로필 사용 시 Stability AI SD3 모델 라이선스/접근 조건 수락

인증값은 환경변수로만 제공하고 저장소 파일에 기록하지 않는다.

```bash
export HF_TOKEN='YOUR_HUGGINGFACE_TOKEN'
```

## 2. 단일 부트스트랩

```bash
git clone https://github.com/dlwlsrnjs/survey_experiment.git
cd survey_experiment
export HF_TOKEN='YOUR_HUGGINGFACE_TOKEN'
./bootstrap.sh --profile core
```

기본 `core` 프로필은 다음을 수행한다.

1. `.venvs/modern` 생성과 고정 requirements 설치
2. 7개 `third_party` 저장소 clone 및 고정 commit checkout
3. 공식 Google Drive의 MM-SafetyBench 이미지 ZIP 다운로드·SHA-256 검증·압축 해제
4. 고정 Hugging Face revision의 MemJack 궤적 다운로드 및 파일별 SHA-256 검증
5. MemJack train 경험 은행 45,016건 재생성
6. Qwen2.5-VL-32B 보조 모델, Qwen2.5-VL-7B 공통 타겟, 공통 SDXL 이미지 생성기 다운로드
7. 단위 테스트와 `doctor --strict` 실행

원형 충실도용 모델과 legacy 환경까지 준비:

```bash
./bootstrap.sh --profile full
```

부트스트랩은 이미 올바른 파일을 다시 받지 않으며, checksum이나 Git commit이 다르면 중단한다. 필요한 단계만 생략할 수도 있다.

```bash
./bootstrap.sh --profile core --skip-models
./bootstrap.sh --profile core --skip-data --skip-repos
```

## 3. 경로와 cache

스크립트는 설치 위치를 `bootstrap.sh` 또는 실행 스크립트 자신의 경로에서 계산한다. `/home/ljk98` 같은 사용자 경로에 의존하지 않는다.

- `PROJECT_ROOT`: 저장소 루트, 실행 스크립트가 자동 결정
- `VENV_ROOT`: 기본값 `<PROJECT_ROOT>/.venvs`
- `HF_HOME`: 기본값 `<PROJECT_ROOT>/.cache/huggingface`
- `PYTHON_BIN`: venv 생성에 사용할 Python, 기본값 `python3`

대용량 공용 cache를 쓰려면 bootstrap 전에 지정한다.

```bash
export HF_HOME=/shared/cache/huggingface
export VENV_ROOT=/shared/venvs/vlm-aux
export PYTHON_BIN=/usr/bin/python3.10
./bootstrap.sh --profile core
```

2026-11-25 동결 manifest는 진행 중 실행의 SHA-256을 보존하기 위해 당시 절대경로 문자열을 유지한다. runtime은 그 경로가 없는 새 머신에서 `data/` 또는 `third_party/` 이하 suffix를 현재 프로젝트 루트로 재매핑한다. 새로 생성하는 manifest는 처음부터 프로젝트 상대경로를 기록한다.

## 4. 다운로드 고정과 검증

`configs/assets.lock.json`에는 다음이 들어 있다.

- 외부 Git URL과 정확한 40자리 commit
- MM-SafetyBench Google Drive file ID와 ZIP SHA-256
- MemJack dataset repository revision과 파일별 SHA-256
- 모델 repository ID, revision, 사용 프로필

검증 명령:

```bash
.venvs/modern/bin/python -m survey_exp.doctor --profile core --strict
.venvs/modern/bin/python -m survey_exp.doctor --profile full --strict
```

GPU가 없는 로그인 노드에서는 CUDA 항목만 경고로 표시한다. 저장소, 데이터, manifest, 경험 은행, 이미지 생성 profile, 모델 cache 누락은 strict 검사 실패다.

## 5. 이미지 생성 설정

`configs/image_generators.yaml`은 두 실험 목적을 섞지 않도록 분리한다.

### 공통 통제

`controlled_default`는 SDXL base 1.0을 고정한다. 신규 실험용 `configs/local32b_pilot_imagegen.yaml`, `configs/local32b_eval_imagegen.yaml`과 `configs/gpt4o_replay.template.yaml`이 이 profile을 명시한다. 보조 모델이 비거부 `scene_description`을 만들면 runner가 실제 이미지를 생성하고 모델 revision, 설정, 경로, SHA-256을 `tool_result.image_generation`에 남긴다.

SDXL은 실행에 쓰지 않는 다른 프레임워크·정밀도 weight까지 내려받지 않도록 `assets.lock.json`의 allowlist로 Diffusers FP16 파일만 받는다. loader도 `variant: fp16`으로 고정한다.

기존 `configs/local32b_eval_memjack.yaml`은 2026-11-25에 시작한 393/2,100건 실행의 config SHA-256을 보존하기 위해 변경하지 않는다. 이 실행을 신규 이미지 생성 실험과 합치지 않는다.

### 원형 충실도

- HADES: 공개 구현의 PixArt-XL-2-1024-MS와 pipeline 기본 생성 설정
- HIMRD: 공개 구현의 Stable Diffusion 3 Medium, 50 steps, guidance 8.0, 512×512

`source_fidelity_runner`는 HADES diffusion prompt와 HIMRD scene prompt에 각각 해당 생성기를 적용한다. HIMRD의 공개 코드는 single-file checkpoint를 사용하지만 배포 자동화에서는 revision이 고정된 Diffusers 형식 저장소를 사용한다. 이 loader 형식 차이는 결과 해석 시 adaptation으로 기록한다.

## 6. 로컬 실행과 Slurm 실행 분리

로컬 GPU:

```bash
bash scripts/run_local32b.sh configs/local32b_pilot_imagegen.yaml
bash scripts/run_local32b.sh configs/local32b_eval_imagegen.yaml
```

Slurm:

```bash
cd /path/to/survey_experiment
sbatch slurm/run_local32b_pilot.sbatch
sbatch slurm/run_local32b_full.sbatch
```

기존 2026-11-25 전체 실행만 다음 명령으로 재개한다.

```bash
sbatch slurm/resume_local32b_20261125.sbatch
```

Slurm 파일은 자원 요청만 담당하고 실제 모델 서버·runner 실행은 로컬 실행기와 공유한다. 클러스터별 partition 이름만 필요할 경우 `#SBATCH --partition`을 조정한다.

원형 충실도:

```bash
bash scripts/run_source_fidelity.sh
sbatch slurm/run_source_fidelity.sbatch
```

## 7. 실패 시 확인

- `401/403` 또는 gated 오류: Hugging Face 웹에서 접근 조건 수락 후 `HF_TOKEN` 확인
- checksum 불일치: 배포본이 변경됐을 가능성이 있으므로 임의로 lock을 갱신하지 말고 출처를 재검토
- Git revision 불일치: `third_party/<name>`의 로컬 변경 여부 확인
- CUDA OOM: 다른 GPU 프로세스 확인, H200급 노드 사용, 이미지 생성 분리 실행 검토
- Google Drive quota 오류: 잠시 뒤 재시도하거나 공식 README의 대체 Baidu 배포본을 수동으로 받은 뒤 동일 ZIP checksum 확인

원시 데이터, 모델 cache와 실행 결과는 `.gitignore` 대상이다. 다른 머신으로 옮길 때 GitHub에는 코드·lock·문서만 두고 각 머신에서 부트스트랩으로 복원한다.
