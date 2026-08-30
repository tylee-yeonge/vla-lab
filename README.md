# vla-lab

Reproducible Vision-Language-Action (VLA) experiments — on-device inference benchmarks, model adaptation, and robot manipulation.

VLA 모델을 **실제로 돌게 만드는 과정**의 공개 기록입니다. 각 산출물은 실행 가능한 코드·측정 조건·수치·판단 근거를 함께 싣습니다 — 조건 없는 수치는 싣지 않습니다.

## 산출물

| # | 산출물 | 내용 | 상태 |
|---|---|---|---|
| v1.5 | [`v1.5-openvla-lora-sim/`](./v1.5-openvla-lora-sim/) | OpenVLA 7B 를 sim (ManiSkill) 데이터로 LoRA adaptation — zero-shot vs fine-tuned before/after 정량 분석 (N회 반복, 분산·부분 도달률 보고) | 작성 중 (2026.09) |
| v2.5 | `v2.5-smolvla-so101-real/` | SO-101 실기 teleop 데이터셋 (LeRobot 포맷, HF Hub) + SmolVLA 실기 zero-shot vs fine-tuned before/after | 예정 (2026.11-12) |

## 실행 환경 요약

- 추론·eval: RTX 4070 12GB (4-bit 양자화) / 학습: RunPod RTX 4090 24GB (LoRA) 또는 로컬 (SmolVLA)
- 로봇: SO-101 (SO-ARM101) 리더-팔로워, Feetech STS3215
- 스택: LeRobot (데이터·학습) + ROS 2 (배포·통합)

## 원칙

1. **결과 비의존 보고** — 성공률이 오르지 않아도 "왜"의 분석을 산출물로 삼는다.
2. **측정 장치 먼저 검증** — 성공률 0% 가 도메인 갭인지 통합 버그인지 구분할 수 있어야 수치가 해석된다.
3. **N 과 분산 명시** — 단발 수치는 증거가 아니다.
