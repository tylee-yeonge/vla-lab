# vla-lab

Reproducible Vision-Language-Action (VLA) experiments — on-device inference benchmarks, model adaptation, and robot manipulation.

VLA 모델을 **실제로 돌게 만드는 과정**의 공개 기록입니다. 각 산출물은 실행 가능한 코드·측정 조건·수치·판단 근거를 함께 싣습니다 — 조건 없는 수치는 싣지 않습니다.

## 산출물

| # | 산출물 | 내용 | 상태 |
|---|---|---|---|
| v1.5 | [`v1.5-openvla-lora-sim/`](./v1.5-openvla-lora-sim/) | OpenVLA 7B 를 sim (ManiSkill) 데이터로 LoRA adaptation — zero-shot vs fine-tuned, 동일 조건 98쌍, 부분 도달률·배제 분석 | **발행 (2026-08-31)** |
| v2.5 | [`v2.5-smolvla-so101-real/`](./v2.5-smolvla-so101-real/) | SO-101 실기에서 SmolVLA 를 teleop 시범 50개로 파인튜닝 — zero-shot vs fine-tuned, 동일 조건 49쌍, placed 0/49 vs 49/49, 절차 이탈 공개. 정책과 분리한 감독 층 (ROS 2 상태기계) 의 실기 증거 포함. [1분 영상](https://youtu.be/zaH7cQzPkc8) | **발행 (2026-10-27)** |

## 실행 환경 요약

- 추론·eval: RTX 4070 12GB (4-bit 양자화) / 학습: RunPod RTX 4090 24GB (LoRA) 또는 로컬 (SmolVLA)
- 로봇: SO-101 (SO-ARM101) 리더-팔로워, Feetech STS3215
- 스택: LeRobot (데이터·학습) + ROS 2 (배포·통합)

## 원칙

1. **결과 비의존 보고** — 성공률이 오르지 않아도 "왜"의 분석을 산출물로 삼는다.
2. **측정 장치 먼저 검증** — 성공률 0% 가 도메인 갭인지 통합 버그인지 구분할 수 있어야 수치가 해석된다.
3. **N 과 분산 명시** — 단발 수치는 증거가 아니다.
