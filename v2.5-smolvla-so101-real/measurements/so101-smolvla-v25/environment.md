# 측정 환경 — so101-smolvla-v25

> 기록일: 2026-10-22 (학습 결과 2026-10-23).

## 1. 실기 · 실행 환경

| 항목 | 값 |
|---|---|
| 로봇 | SO-101 (SO-ARM101) 리더-팔로워, Feetech STS3215 x 12. 측정은 팔로워만 (`so101_follower_01`, `/dev/so101_follower`) |
| 카메라 | 전체 뷰 `/dev/so101_cam_overview` (책상 위, 큐브 · 노란 칸 · 팔 전체), 손목 `/dev/so101_cam_wrist`. 둘 다 1280x720 MJPG 30 fps |
| 작업면 | 책상. 큐브 시작 지점 = 파란 X, 목표 = 노란 테이프 사각형. 큐브 42 x 42 x 25 mm (분홍). 전체 뷰 640x360 기준 큐브 윗면 한 변 42 mm ≈ 21 px (약 2 mm/px), 노란 칸 bbox ≈ 50x42 px (약 100 x 84 mm) |
| PC | RTX 4070 12 GB (드라이버 580.178.04), 컨테이너 (`vscode-tunnel` compose) |
| 소프트웨어 | Python 3.12.3, torch 2.11.0+cu130, lerobot 0.6.2 (`/workspace/venvs/lerobot`), scipy 1.18.1, matplotlib 3.11.2 |
| 정책 실행 | `lerobot-rollout` episodic, cuda, 30 fps. 추론 1회 (50-step 청크) 약 325 ms, 큐 pop 평균 4.8 ms, 루프 실효 28.5 Hz (2026-10-22 zero-shot 실행 로그) |
| 로컬 패치 | `lerobot_record.py` 에 경과 초 표시 (`SO101_TIMER`) — 수집에만 쓰였고 rollout 과 무관 |

## 2. 대상 모델

### 2.1 베이스

`lerobot/smolvla_base` (HF snapshot `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`). SmolVLM2-500M-Video-Instruct 기반, VLM 16층으로 축소, action expert 만 학습 (`train_expert_only=true`, `freeze_vision_encoder=true`). 입력 키 `observation.images.camera1/2/3` + `observation.state` (6), 출력 `action` (6), chunk 50.

### 2.2 모델 A (zero-shot)

베이스를 `Studies/Hardware-Arm/v25/outputs/smolvla_base_so101stats` 로 복사하고 normalizer · unnormalizer safetensors 에 데이터셋 `meta/stats.json` 의 `observation.state.mean/std` · `action.mean/std` 를 넣은 것. 가중치는 베이스 그대로. 한 프레임 검증 (에피소드 0 의 300 프레임, state `[30.6, 51.6, -50.2, 97.8, 33.5, 21.0]`): 원본 출력 `[-0.8, 0.6, -0.3, 0.6, 2.1, -0.5]` → 사본 `[34.5, 49.5, -7.2, 102.5, 26.7, 18.7]`.

### 2.3 모델 B (fine-tuned)

| 항목 | 값 |
|---|---|
| 데이터셋 | `tylee-yeonge/so101-pick-cube-v25` — 50 에피소드, 44900 프레임 (898/에피소드, 30 s), 30 fps, 카메라 `front` · `wrist` 1280x720 → 학습 시 256x256 리사이즈 |
| 명령 | `lerobot-train --policy.path=lerobot/smolvla_base --rename_map='{front→camera1, wrist→camera2}' --batch_size=32 --steps=20000 --save_freq=5000` (v25 PRACTICE §3 그대로) |
| 최적화 | AdamW lr 1e-4 (warmup 1000, cosine decay → 2.5e-6), weight decay 1e-10, grad clip 10, seed 1000, 정밀도 fp32, batch 32 (유효 32) |
| 정규화 통계 | 학습 데이터셋 통계로 덮어씀 (`lerobot-train` 이 pretrained 체크포인트의 normalizer 를 데이터셋 통계로 교체) |
| 실행 | 2026-10-22 23:37 - 2026-10-23 03:13, 벽시계 3 h 35 min (1.55 step/s, 0.65 s/step), 14.25 epoch, 4070 단독 |
| VRAM | 9.9 GB (nvidia-smi, step 300 시점. 12 GB 중) |
| loss | 0.297 (step 200) → 0.131 (2000) → 0.052 (10000) → 0.035 (20000). 최소 0.033 (step 18400). 마지막 4000 step 은 0.035-0.036 으로 평탄. 곡선 `Studies/Hardware-Arm/v25/outputs/plots/train_loss.png`, 원본 `train_loss.csv` |
| grad norm | 3.09 (step 200) → 0.54 (20000) |
| 체크포인트 | `Studies/Hardware-Arm/v25/outputs/smolvla_v25/checkpoints/{005000,010000,015000,020000}`, `last` → `020000` (각 869 MB, 총 5.0 GB). 저장된 정규화 통계는 데이터셋 통계 (`observation.state.mean` = `[3.0, -48.6, 47.0, 57.9, 5.1, 12.5]`), rename 단계 `front→camera1 · wrist→camera2`, 입력 키 `camera1/2/3` |

### 2.4 오프라인 확인 (팔 없음, 측정 전)

학습 시범 3개 (에피소드 0 · 25 · 49) 의 18 프레임에서 두 모델의 첫 action 을 데이터셋 action 과 비교했다. 학습에 쓴 프레임이라 일반화 증거가 아니고, B 가 입력 키 · 정규화 · 출력 단위까지 제대로 이어졌는지만 보는 확인이다.

| 모델 | 관절별 평균 절대 오차 (도) pan / lift / elbow / wrist_flex / wrist_roll / gripper | 전체 |
|---|---|---|
| A zero-shot | 2.9 / 14.3 / 13.4 / 4.8 / 1.5 / 12.0 | 8.1 |
| B fine-tuned | 0.7 / 2.8 / 2.5 / 1.4 / 0.2 / 0.7 | 1.4 |

A 의 오차는 대부분 "현재 state 를 거의 그대로 내는" 데서 온다 — 정지 구간 (휴식 자세) 은 맞고, 움직이는 구간 (그리퍼를 여닫는 순간, 들어 올리는 순간) 에서 크게 틀린다. B 의 가장 큰 오차는 에피소드 0 의 첫 프레임 (들린 자세에서 출발한 유일한 시범) 의 33 도다.

## 3. 집계 · 분석 환경

GPU 불필요. `judge_episodes.py` (PyAV 디코딩 + 색 기반 검출) 와 `analyze_results.py` (scipy) 를 같은 venv 에서 돌린다.
