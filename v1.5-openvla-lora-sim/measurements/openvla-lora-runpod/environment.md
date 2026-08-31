# 실행 환경 — openvla-lora-runpod

> 측정일: 2026-08-13
> 측정 대상: OpenVLA 7B LoRA 파인튜닝 1사이클 (클라우드 GPU)
> 재현 자산: [`scripts/`](scripts/) 의 Dockerfile 2개 + 시작 스크립트 + 등록 패치

## 0. 이 문서가 답하는 질문

> **"이 수치를 다시 얻으려면 무엇이 같아야 하는가."**

이 측정은 세 겹의 환경 위에서 나왔다 — 컨테이너 이미지(패키지 버전), 클라우드 pod(GPU·스토리지), 그리고 데이터셋 등록. 셋 중 하나가 달라지면 수치가 흔들리는 데서 끝나지 않고 학습이 시작되지 않거나 조용히 다른 값으로 학습된다.

이 문서에 해석이 없는 것이 정상이다. 수치의 의미는 [`methodology.md`](methodology.md), 판단은 [`findings.md`](findings.md) 에 있다.

읽기 전에 알아 둘 용어:

| 용어 | 뜻 |
|---|---|
| **network volume** | pod 과 분리된 저장소. pod 을 지워도 남는다 |
| **container disk** | pod 에 붙은 임시 저장소. pod 을 정지하면 초기화된다 |
| **핀**(pin) | 패키지 버전을 `==` 로 못 박는 것 |
| **전이 의존성** | 설치한 패키지가 다시 끌고 오는 패키지. 핀이 없으면 설치 시점의 최신이 들어온다 |

---

## 1. 컨테이너 이미지

두 겹이다. 로컬 검증용 v1 위에 pod 배포용 한 겹을 얹어 v2 를 만든다.

| 항목 | 값 |
|---|---|
| 베이스 | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` |
| 베이스 digest | `sha256:a55ff10111eb11f998884327d37361592e632899edd24fce99886b69289e33e6` |
| CUDA / python | 12.1.1 / 3.11 |
| openvla 기준 커밋 | `c8f03f48af692657d3060c19588038c7220e9af9` |
| 적용 패치 | [`scripts/openvla_registration.patch`](scripts/openvla_registration.patch) (md5 `9a0d1651a230c2f94b7c1e466e297123`) |
| 이미지 크기 | 23.3GB |

베이스 선택의 제약은 **python 버전**이다. openvla 가 `tensorflow==2.15.0` 을 핀으로 박는데 그 버전은 python 3.12 용 휠이 없다. 3.11 이면 핀이 그대로 선다.

### 1.1 설치된 버전

| 패키지 | 버전 | 출처 |
|---|---|---|
| torch | 2.2.0+cu121 | openvla 핀 |
| torchvision / torchaudio | 0.17.0 / 2.2.0 | openvla 핀 |
| timm | 0.9.10 | openvla 핀 |
| transformers | 4.40.1 | openvla 핀 |
| peft | 0.11.1 | openvla 핀 |
| tensorflow | 2.21.0 | 핀을 깼다 (§1.2) |
| tensorflow-datasets | 4.9.10 | 핀을 깼다 |
| tensorflow-metadata | 1.21.0 | 명시 고정 |
| protobuf | 6.33.6 | 명시 고정 |
| tensorflow-graphics | 2021.12.3 | 양쪽 동일 |
| flash-attn | 2.5.5 | 별도 설치 |

없는 것: `tensorflow-addons`, `tensorflow-estimator`, `bitsandbytes`.

### 1.2 리포 핀과 다른 항목

| 패키지 | `pyproject.toml` | 이미지 | 사유 |
|---|---|---|---|
| tensorflow | `==2.15.0` | 2.21.0 | 2.15 가 protobuf 를 5 미만으로 묶어 import 가 깨진다 |
| tensorflow-datasets | `==4.9.3` | 4.9.10 | 위와 한 묶음 |
| protobuf | 핀 없음 | 6.33.6 | 전이 의존성이라 명시 고정 |
| tensorflow-metadata | 명시 없음 (전이) | 1.21.0 | 같은 이유 |
| tensorflow-addons | 전이 | 제거 | TF 2.16 이상과 맞지 않는다 |
| tensorflow-estimator | 전이 | 제거 | TF 2.21 이 쓰지 않는 잔재 |
| flash-attn | 45행에 주석 처리 | 2.5.5 설치 | upstream 이 "editable install 뒤 별도 설치" 를 지시 |
| bitsandbytes | 명시 없음 | 미설치 | 4-bit 학습 경로를 쓰지 않는다 |

**torch / timm 계열은 핀을 지켰다.** 데이터 경로(TF 계열)만 검증된 조합으로 덮어썼다.

`tensorflow-metadata` 의 `_pb2.py` 는 protobuf 5.27+ 의 `runtime_version` 모듈을 요구하는데 `tensorflow==2.15.0` 이 protobuf 를 5 미만으로 캡한다. 선언된 하한이 실제 요구보다 느슨해 **설치는 조용히 성공하고 import 에서 터진다.**

### 1.3 pod 배포용 한 겹 (v2)

[`scripts/Dockerfile.pod`](scripts/Dockerfile.pod) — v1 위에 세 가지를 더한다.

| 변경 | 사유 |
|---|---|
| 코드를 `/workspace/openvla` -> `/opt/openvla` 로 복사 후 editable install 재지정 (`--no-deps`) | network volume 이 pod 의 `/workspace` 에 마운트되어 이미지 안의 같은 경로를 가린다. `--no-deps` 는 `tensorflow==2.15.0` 핀의 재해석을 막는다 |
| `openssh-server` 설치 + 시작 스크립트 | RunPod 은 `PUBLIC_KEY` 를 주입할 뿐이고 sshd 는 이미지의 몫이다 |
| 빌드 중 생성된 `/etc/ssh/ssh_host_*` 삭제 | 패키지 설치가 빌드 시점에 호스트 키를 만들어 이미지에 굽는다. public repository 에 올리면 개인키가 공개된다 |

[`scripts/runpod_start.sh`](scripts/runpod_start.sh) 가 컨테이너 시작 시 수행하는 것:

1. `PATH`, `LD_LIBRARY_PATH`, `CUDA_HOME`, `HF_*` 를 `/etc/environment` 로 내보낸다 — **SSH 로 들어온 셸은 Docker 환경변수를 물려받지 않는다**
2. `PUBLIC_KEY` 를 `authorized_keys` 에 등록
3. `ssh-keygen -A` 로 호스트 키 생성 후 sshd 기동
4. `sleep infinity` 로 컨테이너 유지

---

## 2. pod 사양

| 항목 | 값 |
|---|---|
| GPU | RTX 4090 1x, 24,564MiB |
| 호스트 | 60GB RAM, 12 vCPU |
| 드라이버 / CUDA | 570.195.03 / 12.8 |
| 데이터센터 / 티어 | EU-RO-1 / Secure Cloud |
| 시간당 요금 | GPU $0.74 + container disk $0.007 = $0.75 (2026-08 확인) |
| container disk | 50GB |
| network volume | 50GB, `/workspace` 에 마운트 (MooseFS) |
| 이미지 배포 | Docker Hub, RunPod Container Registry Auth 로 인증 pull |

이미지의 CUDA 는 12.1 이고 pod 드라이버는 12.8 이다. 드라이버가 상위이므로 구 버전 런타임이 그대로 동작한다.

`df -h /workspace` 가 총량을 수 PB 로 보여주는 것은 MooseFS 공유 파일시스템이기 때문이다. **50GB 는 쿼터로 걸리며 df 에 나타나지 않는다** — 사용량은 `du` 로 재야 한다.

### 2.1 이 환경이 강제한 경로 설계

| 대상 | 위치 | 이유 |
|---|---|---|
| 코드 | `/opt/openvla` (이미지) | volume 이 `/workspace` 를 덮는다 |
| 데이터셋 | `/workspace/data` | volume |
| base 가중치 캐시 | `/workspace/hf` (`HF_HOME`) | 15GB. pod 재생성 시 재다운로드 회피 |
| 체크포인트 | `/workspace/runs` | pod 회수 대비 |
| LoRA 어댑터 | `/workspace/adapter-tmp` | **기본값이 상대 경로라 container disk 에 떨어진다** |
| wandb 기록 | `/workspace/wandb` (`WANDB_DIR`) | 기본 위치는 container disk. loss 가 여기에만 있다 |

마지막 두 줄이 이 환경에서 가장 쉽게 잃는 것이다. 둘 다 기본값이 컨테이너 안이고, pod 을 정지하는 순간 사라진다.

---

## 3. 데이터셋과 base 모델

| 항목 | 값 |
|---|---|
| 데이터셋 | `maniskill_pickcube` (RLDS), 3,760 프레임 / 파일 5개 / 261,606,336 바이트 |
| 데이터 mixture | `maniskill_pickcube_only` (가중치 1.0) |
| 통계 파일 | [`raw/dataset_statistics.json`](raw/dataset_statistics.json) |
| base 모델 | `openvla/openvla-7b` |
| base 리비전 | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` |

데이터셋 등록은 upstream 리포의 3개 파일(`mixtures.py` / `configs.py` / `transforms.py`)을 수정해 이루어지며, 그 변경이 [`scripts/openvla_registration.patch`](scripts/openvla_registration.patch) 다. 기준 커밋에 `git apply` 로 적용한다.

**base 리비전이 재현의 핵심이다.** 어댑터는 이 리비전 위에서 학습됐고, [`raw/adapter_config.json`](raw/adapter_config.json) 의 `auto_mapping.parent_library` 에 그 해시가 박혀 있다. 다른 리비전의 base 에 합치면 모듈 구조가 어긋날 수 있다.

---

## 4. 재현 절차 요약

```bash
# 1. 이미지 (호스트)
docker build -t openvla-train:v1 .                      # scripts/Dockerfile
docker build -f Dockerfile.pod -t openvla-train:v2 .    # scripts/Dockerfile.pod
docker push <registry>/openvla-train:v2

# 2. pod (RunPod 콘솔)
#    - S3 API 를 지원하는 데이터센터에 network volume 생성 (회수 경로 확보)
#    - 커스텀 템플릿: 이미지 지정, TCP 22 노출, 시작 명령은 비움
#    - 환경변수: HF_HOME=/workspace/hf, HF_TOKEN, WANDB_DIR=/workspace/wandb

# 3. 데이터 전송 후 로드 검증 (pod)
python practice_load_check.py

# 4. 학습
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py ...
```

전체 절차와 각 단계의 함정은 `Studies/Phase 4.5/week3/PRACTICE.md` 에 있다.

---

## 5. 이 환경에서 확인하지 못한 것

- **volume 재기동 후 잔존 확인**: pod 을 정지·재시작해 volume 이 남는지 눈으로 보는 검증은 수행하지 않았다. 다만 pod 삭제 후에도 S3 API 로 volume 내용에 접근된 것이 간접 증거다
- **다른 GPU 에서의 동작**: flash-attn 은 로컬 RTX 4070(sm_89) 에서 컴파일된 바이너리이고 RTX 4090 도 같은 아키텍처다. A40/A100 등 다른 세대에서는 확인하지 않았다
- **batch 2 조합**: VRAM 여유가 6GB 남았으나 `batch_size 2 / grad_accumulation 8` 은 시도하지 않았다
