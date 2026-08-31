# 실행 환경 — openvla-maniskill-zeroshot

> 측정일: 2026-08-03
> 측정 코드 commit: `bfb58cf` (fix(phase4.5): wire action contract into week0 zeroshot baseline)
> 측정 실행: 2026-08-03 14:44 (`zeroshot_baseline.json`), 14:36 (`harness_check.json`) — 둘 다 위 commit 이후 실행

## 0. 이 문서가 답하는 질문

> **"이 수치를 다시 얻으려면 무엇이 같아야 하는가."**

이 측정은 두 스택을 한 프로세스에 올려 얻은 값이다 — 시뮬레이터(ManiSkill)와 7B 규모의 VLA 모델(OpenVLA). 둘 중 한쪽의 버전이 달라지면 수치가 달라지는 데서 끝나지 않고, 로드 자체가 실패하거나 렌더가 검은 화면으로 나온다. 그래서 조건([`methodology.md`](methodology.md))과 해석([`findings.md`](findings.md)) 앞에 환경을 먼저 고정해 적는다.

이 문서에 해석이 없는 것이 정상이다.

읽기 전에 알아 둘 용어:

| 용어 | 뜻 |
|---|---|
| **venv** | 파이썬 가상 환경. 프로젝트별로 패키지 버전을 격리하는 폴더 |
| **ICD**(Installable Client Driver) | Vulkan 이 GPU 드라이버를 찾아 쓰기 위한 등록 파일 |
| **Vulkan** | GPU 렌더링 API. SAPIEN(ManiSkill 의 렌더러)이 이것으로 화면을 그린다 |
| **4-bit 양자화** | 모델 가중치를 4비트로 줄여 VRAM 사용량을 낮추는 기법. 여기서는 `nf4` 방식 |
| **에셋**(asset) | 3D 모델·텍스처 등 시뮬레이터가 씬을 구성할 때 내려받는 외부 파일 |

## 1. 하드웨어

| 항목 | 값 | 확인 방법 |
|---|---|---|
| GPU | NVIDIA RTX 4070 12GB | `torch.cuda.get_device_name(0)` |
| CUDA (torch 빌드) | 13.0 | `torch.version.cuda` |
| GPU 가용성 | True | `torch.cuda.is_available()` |
| 실행 형태 | Docker 컨테이너 (ubuntu-dev), 헤드리스 | - |

## 2. venv 두 개와 각자의 역할

측정은 `.venv-vla` 한 곳에서 이루어졌다. `.venv-sim` 은 sim 단독 실습과 대조용으로 유지한다.

| venv | 경로 | 이 측정에서의 역할 |
|---|---|---|
| `.venv-vla` | `Studies/Phase 4/.venv-vla` | **측정 실행 환경.** OpenVLA 추론과 ManiSkill sim 을 한 프로세스에서 돌린다 |
| `.venv-sim` | `Studies/Phase 4.5/.venv-sim` | sim 단독 대조용. `.venv-vla` 에서 렌더 문제가 났을 때 sim 자체의 정상성을 가리는 기준 |

두 스택을 합친 판단의 근거는 [`methodology.md`](methodology.md) §1 에 있다. 요지는 ManiSkill 이 `.venv-vla` 에 이미 설치되어 있었고, Phase 4 측정 기록이 고정을 요구하는 버전 4개(transformers / tokenizers / timm / accelerate)가 그대로 보존되어 있었다는 것이다.

## 3. 패키지 버전 (`.venv-vla`, 측정 실행 환경)

| 항목 | 버전 | 고정이 필요한 이유 |
|---|---|---|
| Python | 3.12.3 | - |
| torch | 2.12.0+cu130 | - |
| transformers | 4.40.1 | OpenVLA remote code 가 이 버전대의 API 를 전제한다. Phase 4 측정 기록의 고정 항목 |
| tokenizers | 0.19.1 | 같음 |
| timm | 0.9.16 | 같음 (vision encoder) |
| accelerate | 1.0.1 | 같음 |
| bitsandbytes | 0.49.2 | 4-bit 양자화 로드 경로 |
| mani_skill | 3.0.1 | 환경 등록값(step cap, `goal_thresh`)이 이 버전의 소스에서 읽은 값이다 |
| sapien | 3.0.3 | 렌더러 |
| gymnasium | 1.3.0 | `TimeLimit` wrapper 가 step 예산을 집행한다 |
| numpy | 2.5.x | §6 참조 — 기록이 어긋나 있어 재확인 대상 |

`.venv-sim` 의 torch 는 2.13.0+cu130 으로 `.venv-vla` 와 다르다. 같은 sim 코드가 두 venv 에서 다르게 거동하면 이 차이를 첫 용의자로 본다.

## 4. 모델과 로드 설정

| 항목 | 값 |
|---|---|
| 모델 | `openvla/openvla-7b` (Hugging Face) |
| remote code | `trust_remote_code=True` — `modeling_prismatic.py` 를 모델 저장소에서 함께 받아 실행 |
| 양자화 | 4-bit, `nf4`, double quant, compute dtype `float16` |
| attention 구현 | `eager` |
| checkpoint shard | 3개 |
| 추론 디바이스 | `cuda:0` |

Phase 4 week6 의 latency 측정([`../openvla-rtx4070-int4/`](../openvla-rtx4070-int4/))과 같은 로드 설정이다. 즉 이 baseline 의 추론 1회 비용은 그 측정의 latency 수치를 그대로 원용할 수 있다.

## 5. 시뮬레이터 에셋과 렌더

| 항목 | 값 | 근거 |
|---|---|---|
| 다운로드 에셋 | 없음 | `~/.maniskill/data` 가 없는 상태에서 `PickCube-v1` 이 정상 생성됨. 큐브 + 테이블의 primitive shape 만 쓴다 |
| 렌더 동작 확인 | [`raw/env_check.png`](raw/env_check.png) (512x512 사람용 렌더) | 팔 + 테이블 + 큐브 + goal 마커가 보인다. 픽셀 평균 94.6 으로 검은 화면이 아니다 |

**Vulkan ICD 경고는 무해하다.** `sapien/_vulkan_tricks.py` 가 `Failed to find Vulkan ICD file` UserWarning 을 내지만, SAPIEN 이 자체 ICD 로 폴백해 렌더가 실제로 동작한다. 폴백이 실패하면 위 PNG 가 검은 화면으로 나오므로, 그 그림이 판정 수단이다.

## 6. 이 기록에 남은 구멍

- **numpy 버전이 한 값으로 확정되지 않았다.** 작업 중 기록에 `2.5.0` 과 `2.5.1` 이 함께 나타난다. 두 venv 를 각각 재확인해 이 표를 한 값으로 고쳐야 한다. numpy 는 `mplib` 의 ABI 실패([`findings.md`](findings.md) §1.2)와 직접 얽혀 있어 무해한 항목이 아니다.
- **Phase 4 측정 기록에 numpy 버전이 없다.** 그래서 ManiSkill 설치가 Phase 4 latency 수치의 재현성을 훼손했는지는 대조로 확인할 수 없고, 재측정으로만 확인된다.
- **드라이버 버전을 기록하지 않았다.** CUDA 13.0 은 torch 빌드 기준값이고, 호스트 NVIDIA 드라이버 버전은 남기지 않았다.
