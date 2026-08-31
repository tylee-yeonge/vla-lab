# 측정 조건·절차 — openvla-maniskill-zeroshot

> 측정 대상: OpenVLA 7B (int4/nf4) 를 fine-tuning 없이 ManiSkill `PickCube-v1` 에 연결했을 때의 성공률과 부분 도달률
> 측정 코드: [`scripts/practice_zeroshot_baseline.py`](scripts/practice_zeroshot_baseline.py)
> 원본 데이터: [`raw/zeroshot_baseline.json`](raw/zeroshot_baseline.json) (episode 20건 + gripper 원값 951개)
> 대조군 코드·데이터: [`scripts/harness_check.py`](scripts/harness_check.py) / [`raw/harness_check.json`](raw/harness_check.json)

## 0. 이 문서가 답하는 질문

> **"이 숫자는 정확히 무엇을 잰 값인가."**

"zero-shot 성공률 0%" 는 그 자체로 의미가 확정되지 않는다. 어떤 task 인지, 성공을 무엇으로 판정했는지, 몇 step 을 줬는지, 모델 출력을 어떤 규칙으로 로봇 명령으로 바꿨는지에 따라 같은 모델에서 전혀 다른 값이 나온다. 이 문서는 그 조건을 전부 적어 수치의 의미를 확정한다.

문서 구성:

| 절 | 내용 |
|---|---|
| §1 | 측정 대상과 단일 프로세스 구성 |
| §2 | task·환경 조건 |
| §3 | 판정 정의 (성공 + 부분 도달률 4단계) |
| §4 | action 변환 계약 — 모델 출력을 로봇 명령으로 바꾸는 규칙 |
| §5 | 측정 절차 |
| §6 | 결과 수치 (본 측정) |
| §7 | 대조군 조건과 수치 (상한 / 하한) |
| §8 | 이 조건이 확정하지 않은 것 |

해석은 여기 쓰지 않는다 — "그래서 무엇을 뜻하는가" 는 [`findings.md`](findings.md) 가 담당한다. **모든 수치의 원본은 이 문서 한 곳**이다.

## 1. 측정 대상과 단일 프로세스 구성

재려는 것은 **adaptation 이전의 출발점**이다. 이후 fine-tuning 의 before 값이 되므로, 같은 조건을 나중에 그대로 재현할 수 있어야 한다.

OpenVLA 추론과 sim 을 **한 프로세스**에서 돌린다. 두 프로세스로 나눠 파일·소켓으로 주고받는 구성은 쓰지 않는다. 근거:

| 확인 항목 | 결과 |
|---|---|
| `.venv-vla` 에 ManiSkill 이 이미 설치되어 있는가 | 예. `mani_skill 3.0.1`, `sapien 3.0.3`, `pytorch-kinematics 0.7.6` 모두 `Requirement already satisfied` |
| Phase 4 측정 기록의 고정 버전이 보존됐는가 | 예. transformers 4.40.1 / tokenizers 0.19.1 / timm 0.9.16 / accelerate 1.0.1 그대로 |
| 한 인터프리터에서 두 스택이 동시에 import 되는가 | 예. `import mani_skill.envs` 후 `gymnasium.registry` 의 PickCube 항목 4개, `from transformers import AutoModelForVision2Seq` 성공, import 이후 `transformers.__version__` 변동 없음 |
| `pip check` | `No broken requirements found.` |

재현 명령:

```bash
source "/workspace/study/physical-ai-study/Studies/Phase 4/.venv-vla/bin/activate"
python -c "import gymnasium, mani_skill.envs, transformers; \
print(sum(1 for k in gymnasium.registry if 'PickCube' in k), transformers.__version__)"
pip check
```

`pip install --dry-run mani_skill` 의 `Would install` 빈 출력은 근거로 쓰지 않았다. pip 은 이미 설치된 패키지에 대해 그 행을 내지 않으므로, 빈 출력이 "충돌 없음"과 "이미 설치됨"을 구분하지 못한다.

## 2. task·환경 조건

| 항목 | 값 | 근거 |
|---|---|---|
| 환경 id | `PickCube-v1` | 등록된 PickCube 변종 중 Panda 기본형. 그 외 `PickCubeSO100-v1`, `PickCubeWidowXAI-v1`, `TwoRobotPickCube-v1` 이 등록돼 있다 |
| 로봇 | Panda 7축 + 2지 gripper (`agent/qpos` = (1, 9)) | 환경 기본값 |
| `obs_mode` | `rgb` | 정책이 카메라 이미지만 보게 한다 |
| `control_mode` | `pd_ee_delta_pose` | OpenVLA 출력(EEF 위치·회전 델타 + gripper)과 차원·의미가 대응하는 유일한 모드. 지원 11종 중 선택 |
| action 공간 | `Box(-1.0, 1.0, (7,), float32)` | 전 차원 `[-1, 1]` 정규화 입력 |
| 정책 입력 카메라 | `obs["sensor_data"]["base_camera"]["rgb"]`, (1, 224, 224, 3) | `sensor_configs=dict(width=224, height=224)` 로 지정 |
| 사람용 렌더 | `env.render()`, (1, 512, 512, 3) | 정책 입력과 무관. 통과 확인용 |
| 명령 문장 | `"pick up the cube"` | 영어 단문 고정 |
| prompt 틀 | `In: What action should the robot take to {명령}?\nOut:` | OpenVLA 가 학습된 문장 틀 |
| 디코딩 | `do_sample=False` | 결정적 출력. 제어에서 무작위성을 배제한다 |
| `unnorm_key` | `bridge_orig` | §4 참조 |
| seed 목록 | `list(range(20))` = 0-19 | 대조군과 동일 |
| env step 예산 | 200 (`gym.make(..., max_episode_steps=200)`) | 등록 기본값 50 을 명시적으로 올린 값 |
| action repeat | 4 | §4 의 제어 주기 불일치 보정 |
| 정책 결정 횟수 | 50 (= 200 / 4) | 위 두 값의 귀결. sim 시간 10초 |

**관측 해상도를 224x224 로 고정한 이유.** OpenVLA 의 vision encoder(SigLIP + DINOv2)가 224x224 로 학습됐다. ManiSkill 기본값 128x128 을 두면 추론 직전 128 -> 224 업샘플이 필요하고, 그 열화가 성공률에 섞여 들어가 "모델의 한계"와 "입력 화질의 한계"를 분리할 수 없게 된다. 128 을 유지할 이유는 렌더 비용뿐이고, 이 조합에는 128 기준 공개 수치가 없어 비교 가능성도 잃지 않는다. **이 값은 baseline 정의의 일부다** — 이후 측정과 비교할 때 해상도를 함께 명시한다.

## 3. 판정 정의

### 3.1 성공 (`placed`)

환경이 소유한 정의를 그대로 쓴다. `PickCube-v1.evaluate()` 의 소스를 직접 읽어 확인한 값이다.

```python
is_obj_placed   = ||goal_site.pose.p - cube.pose.p|| <= goal_thresh    # 3D 유클리드 거리
is_robot_static = max(|qvel[:-2]|) <= 0.2                              # 손가락 2관절 제외, arm 7관절
success         = is_obj_placed & is_robot_static
```

| 항목 | 값 |
|---|---|
| `goal_thresh` | 0.025 m |
| `cube_half_size` | 0.02 m |
| 큐브 초기 xy | `[-0.1, 0.1]^2` 균일 추출 |
| goal 위치 추출 | xy 는 큐브와 **독립적으로** `[-0.1, 0.1]^2`, z 는 `cube_half_size + U(0, 0.3)` |
| 정지 판정 | arm 7관절만, 임계 0.2 (max-abs, L2 아님) |

**`is_grasped` 는 success 에 들어가지 않는다.** `evaluate()` 가 계산해 `info` 로 내보내지만 AND 조건에서 빠져 있다. 이것이 §7.2 하한의 원인이다.

### 3.2 부분 도달률 3단계

최종 성공률만 재면 0 과 0 을 비교하게 되어 이후 개선을 볼 수 없다. 그래서 진행 단계를 쪼개 함께 기록한다. **임계값은 baseline 정의의 일부이므로 이후 측정에서 같은 값을 쓴다.**

| 단계 | 판정식 | 임계값의 근거 |
|---|---|---|
| `reached` | `||tcp - cube|| < 0.05` m | 대조군 scripted 정책이 쓴 접근 고도와 같은 값 -> 상한 대조와 같은 척도가 된다 |
| `grasped` | `info["is_grasped"]` | 환경이 이미 계산해 매 step 실어 보낸다. 판정식을 새로 만들면 상류 정의와 어긋날 위험만 생긴다 |
| `lifted` | `cube.pose.p[2] > 0.04` m | 큐브가 꼭짓점으로 기울어 서면 중심 높이가 최대 `0.02*sqrt(3)` = 0.0346 m 다. 그보다 위로 잡아야 "기울어짐"을 "들림"으로 오판하지 않는다 |

각 단계는 episode 안에서 한 번이라도 참이면 참으로 기록한다(OR 누적). 상태 접근 경로는 `env.unwrapped` 의 `cube`, `goal_site`, `agent.tcp`, `goal_thresh`, `cube_half_size` 다.

판정식의 음성·양성 대조를 직접 확인했다.

| 상태 | `reached` | `lifted` |
|---|---|---|
| 테이블 위 정지 (seed 0, cube z = 0.0200, tcp-cube = 0.1634) | False | False |
| 큐브를 TCP 위치로 강제 이동 (cube z = 0.1674, tcp-cube = 0.0147) | True | True |
| 꼭짓점 기립 최악 케이스 (중심 높이 0.0346) | - | False |

## 4. action 변환 계약

OpenVLA 출력과 ManiSkill 입력은 **단위·범위·부호가 서로 다르다.** 이 변환을 거치지 않고 모델 출력을 그대로 넣으면 예외 없이 조용히 틀린다 — 값이 우연히 `[-1, 1]` 안에 들어오기 때문이다.

| # | 항목 | OpenVLA 쪽 | ManiSkill 쪽 | 변환 규칙 |
|---|---|---|---|---|
| 1 | 위치 델타 | 미터. 출력 범위 = `[q01, q99]`, 최대 ±0.042 m/step (5 Hz) | 미터. ±0.1 m/step (20 Hz), `[-1,1]` 정규화 입력 | `a = d / 0.1` |
| 2 | 값의 범위 규약 | 출력은 **역정규화된 물리량** | 입력은 **`[-1, 1]` 정규화값** (`normalize_action=True`) | 물리량을 축 한계로 나눈다 |
| 3 | 기준 프레임 | bridge 규약의 EEF 델타, 병진은 베이스 기준 | `root_translation:root_aligned_body_rotation`, `use_delta=True` | 재사상 불필요 |
| 4 | 회전 표현 | axis-angle, 라디안, 최대 ±0.207 rad/step | XYZ 오일러, ±0.1 rad/step, **부호 반전**, 클리핑은 3벡터 노름 기준 | `a = r / (-0.1)`, 노름 > 1 이면 방향 유지한 채 축소 |
| 5 | gripper | mask=False -> raw 통과. `[0, 1]`, **0 = 닫힘 / 1 = 열림** | `[-1, 1]`, **-1 = 닫힘 / +1 = 열림** | `g = 2*raw - 1` |

```python
def to_maniskill_action(raw_action):
    pos = raw_action[:3] / 0.1                    # 미터 -> 정규화
    rot = raw_action[3:6] / -0.1                  # 라디안 -> 정규화, rot_lower 곱셈 때문에 부호 반전
    rot_norm = np.linalg.norm(rot)
    if rot_norm > 1.0:                            # 회전은 축별이 아니라 노름으로 제한된다
        rot = rot / rot_norm
    grip = 2.0 * raw_action[6] - 1.0              # [0,1](0=닫힘) -> [-1,1](-1=닫힘)
    action = np.concatenate([np.clip(pos, -1, 1), rot, [np.clip(grip, -1, 1)]])
    return action.astype(np.float32)
```

### 4.1 회전 부호 반전의 출처

`pd_ee_pose.py:232` 가 회전 벡터를 `rot_action * self.config.rot_lower` 로 스케일하고 Panda 의 `rot_lower` 가 `-0.1` 이다. 위치는 `clip_and_scale_action` 으로 `[low, high]` 에 정상 사상되는데 회전만 이 경로를 탄다. 코드를 읽지 않으면 잡히지 않고, 모르고 넣으면 회전이 전부 반대로 간다.

컨트롤러의 `_clip_and_scale_action` 을 직접 호출해 확인한 사상:

| 입력 action | 실제 물리량 |
|---|---|
| pos x = +1 / -1 | +0.1 / -0.1 m |
| rot z = +1 / -1 | **-0.1 / +0.1 rad** |
| gripper = -1 / +1 | 관절 목표 -0.01 / +0.04 m |

변환을 적용했을 때와 생략했을 때의 실측 차이 (정책 1회 = env 4 step, 위치 델타 = `q99` 근처):

| 경로 | 4 step 이동량 |
|---|---|
| 변환 적용 | 0.0910 m |
| 변환 생략 (raw 통과) | 0.0088 m (1/10.3) |

gripper 도 같은 방식으로 어긋난다. OpenVLA 의 "닫힘" 명령 `raw = 0.0` 을 그대로 넣으면 finger 관절이 0.015 m 에 머물고, 변환을 거친 `-1.0` 은 0.0 (완전 닫힘)까지 간다.

### 4.2 `unnorm_key` 선택

`bridge_orig` 를 쓴다. **역정규화 출력의 단위는 `unnorm_key` 마다 다르며 "미터"가 보편 규약이 아니다.**

| unnorm_key | 위치 q01 / q99 (x축) | 해석 | transitions |
|---|---|---|---|
| `bridge_orig` | -0.029 / 0.028 | 미터, 5 Hz | 2,135,463 |
| `fractal20220817_data` | -0.225 / 0.178 | 미터, 대역 8배 (RT-1, 3 Hz) | 3,786,400 |
| `taco_play` | -0.711 / 0.648 | 미터로 보기 어려운 대역 | 237,798 |
| `austin_buds_...` | -1.000 / 1.000 | 원본이 이미 정규화됨 -> 미터 해석 불가 | 34,112 |

채택 사유: (a) `[q01, q99]` 대역이 미터로 물리적으로 타당하고 테이블 위 근거리 조작 규모와 맞다, (b) 대역이 Panda 의 step 한계 ±0.1 m 안에 들어 클리핑 없이 사상된다 — `fractal20220817_data` 는 ±0.22 m 로 한계를 2배 넘어 위치 명령이 상시 포화한다, (c) transition 수가 두 번째로 많아 통계가 안정적이다.

`predict_action` 이 쓰는 통계량은 `q01` 과 `q99` 두 개뿐이다. `min`/`max` 는 참조되지 않는다 — `bridge_orig` 의 회전 z축 `max` 가 6.280 = 2π 인데 같은 축 `q99` 는 0.204 다. 6.28 은 각도 차분이 ±π 경계를 넘으며 생긴 wrap-around 아티팩트이고, `min`/`max` 로 스케일을 잡으면 이 이상치 하나가 사상 범위를 30배 늘린다.

### 4.3 제어 주기 불일치와 action repeat

`bridge_orig` 의 델타는 0.2초(5 Hz)에 적용될 값이고 ManiSkill step 은 0.05초(20 Hz)다. 1 step 에 그대로 넣으면 같은 변위를 1/4 시간에 실행해 팔이 학습 시보다 4배 빠르게 움직인다. 그래서 **모델을 한 번 호출하고 같은 action 을 4 step 연속 넣어** 실효 주기를 5 Hz 로 맞춘다. 정책 결정 횟수가 step 예산의 1/4 이 되므로 예산을 200 으로 올려 50회를 확보한다.

`bridge_orig` 의 5 Hz 는 이 환경에서 측정한 값이 아니라 데이터셋 규약에서 온 값이다 (§8 참조).

## 5. 측정 절차

정책 결정 1회가 하는 일:

1. 관측에서 정책 입력 카메라 이미지를 꺼낸다 (사람용 렌더가 아니다). GPU 텐서이므로 `.cpu()` 로 내리고 배치 차원을 벗긴다
2. 이미지와 prompt 를 OpenVLA 에 넣어 7개 숫자를 받는다 (`attention_mask` 는 넘기지 않는다 — Phase 4 week6 에서 확인한 크래시 회피)
3. `raw_action[6]` (gripper 원값)을 기록한다 — 계약 표 5번의 검증용
4. §4 의 변환 함수를 적용한다
5. 변환된 action 을 4 step 연속 환경에 넣는다
6. 매 env step 마다 §3 의 4단계를 OR 누적한다
7. 성공 또는 `terminated`/`truncated` 면 episode 를 끝낸다

seed 20개를 이 절차로 순회한다. 모델은 한 번만 로드한다.

## 6. 결과 수치 (본 측정)

원본: [`raw/zeroshot_baseline.json`](raw/zeroshot_baseline.json)

### 6.1 성공률과 부분 도달률

| 단계 | 도달 episode | 비율 |
|---|---|---|
| `reached` (tcp-cube < 0.05 m) | 0 | 0/20 |
| `grasped` (`info["is_grasped"]`) | 0 | 0/20 |
| `lifted` (cube z > 0.04 m) | 0 | 0/20 |
| `placed` (환경 success) | 1 | 1/20 |
| `placed` 중 하한 seed(§7.2) 제외 | 0 | **0/20** |

### 6.2 소비한 예산

| 항목 | 값 |
|---|---|
| 정책 결정 50회를 다 쓴 episode | 19개 (seed 0-7, 9-19) |
| 1회로 끝난 episode | 1개 (seed 8 — 첫 step 에서 success 가 잡혀 종료) |
| 총 정책 호출 수 | 951 (= 19x50 + 1) |

### 6.3 gripper 원값 분포 (`raw_action[6]`, n=951)

| 항목 | 값 |
|---|---|
| 최소 / 최대 / 평균 | 0.0 / 0.9961 / 0.3069 |
| 나타난 서로 다른 값 | **2개뿐** — `0.0` 과 `0.9961` |
| 0.1 미만 | 658회 |
| 0.9 초과 | 293회 |
| 그 사이 | 0회 |
| 음수 | 0회 |

계약 표 5번이 코드가 아니라 학습 파이프라인에서 추론한 `[0, 1]` 이산값 가정이었고, 이 분포가 그 가정을 실측으로 확인한다. `-1` 근처 값은 나오지 않았다.

## 7. 대조군 조건과 수치

본 측정과 **변인을 맞춘** 두 대조군이다. 조건이 하나라도 다르면 대조가 성립하지 않으므로 아래 표로 대조한다.

| 항목 | 본 측정 | 대조군 |
|---|---|---|
| 환경 id / `control_mode` / `obs_mode` | `PickCube-v1` / `pd_ee_delta_pose` / `rgb` | 동일 |
| `sensor_configs` | 224x224 | 동일 |
| env step 예산 | 200 | 동일 |
| seed 목록 | 0-19 | 동일 |
| 성공 판정 | `info["success"]` | 동일 (환경 소유) |

### 7.1 상한 — scripted 정책 (특권 정보 사용)

큐브·목표 좌표를 시뮬레이터에서 직접 읽어(카메라를 보지 않는다) 접근 -> 하강 -> 파지 -> 이동 -> 정지 5단계로 움직이는 정책이다. 임계값(0.008 / 0.006 m, 8 step, 3 cm/step)은 직접 정한 값이다.

| 항목 | 값 |
|---|---|
| 성공률 | **20/20** |
| 소비 step (20 episode 전체) | 최소 9 / 최대 49 / 평균 39.4 |
| 소비 step (실제 조작이 일어난 19 episode) | 최소 28 / 최대 49 / 평균 41.0 |
| step 예산 | 200 |
| 파지 없이 성공한 episode | seed 8 하나 |

ManiSkill 내장 **motion planning** 해법은 쓸 수 없었다. `mani_skill 3.0.1` 이 핀으로 요구하는 `mplib==0.1.1` 바이너리가 numpy 1.x 의 C API 로 빌드돼 있어 설치된 numpy 2.x 와 ABI 가 맞지 않고, `mplib.Planner` 생성 시점에 세그폴트로 죽는다 (`mplib/planner.py:65`). 증거는 `mplib/pymp*.so` 안의 `numpy.core` 문자열(numpy 2 에서는 `numpy._core` 로 개칭)과 `FAILURE obtaining numpy _ARRAY_API pointer` 다. numpy 를 1.26.4 로 내리면 mplib 은 살지만 `scipy` / `opencv-python` 이 numpy>=2 를 요구해 그쪽이 깨진다.

scripted 로 바꾼 것이 손해가 아니다 — motion planning 은 `pd_joint_pos` 를 쓰므로 본 측정과 `control_mode` 가 달라진다. scripted 는 본 측정과 같은 `pd_ee_delta_pose` 를 쓰므로 `[-1, 1]` 정규화 action 이 물리량으로 풀리는 경로까지 함께 지나간다.

### 7.2 하한 — 무행동 정책

전 차원 0 인 action 으로 같은 20 seed 를 돌린 결과다.

| 항목 | 값 |
|---|---|
| 성공률 | **1/20** |
| 성공한 seed | `[8]` |

원인은 goal 위치 추출 규칙이다 (§3.1). goal xy 가 큐브와 독립적으로 뽑히고 z 가 `U(0, 0.3)` 에서 낮게 뽑히면, 목표 구가 큐브 초기 위치와 겹쳐 생성된다. 그 episode 는 `is_obj_placed` 가 reset 직후부터 True 이고 초기 `qvel = 0` 이라 `is_robot_static` 도 True 이므로 성공으로 집계된다.

reset 직후(step 을 한 번도 부르기 전) 20개 seed 의 goal-cube 거리:

| seed | 거리 (m) | seed | 거리 (m) | seed | 거리 (m) | seed | 거리 (m) |
|---|---|---|---|---|---|---|---|
| 0 | 0.2760 | 5 | 0.2635 | 10 | 0.2940 | 15 | 0.1797 |
| 1 | 0.2277 | 6 | 0.2270 | 11 | 0.1171 | 16 | 0.1930 |
| 2 | 0.0313 | 7 | 0.1902 | 12 | 0.1281 | 17 | 0.1683 |
| 3 | 0.2601 | **8** | **0.0148** | 13 | 0.2828 | 18 | 0.1657 |
| 4 | 0.3056 | 9 | 0.2914 | 14 | 0.0600 | 19 | 0.1730 |

`goal_thresh` 는 0.025 m 다. seed 8 만 그 안에 들어오고, 다음으로 가까운 seed 2 가 0.0313 m 로 임계값을 간신히 넘긴다. `is_robot_static` 은 20개 seed 전부 reset 시점에 True 이므로 **유일한 관문은 `is_obj_placed`** 다. 이 표는 `.venv-sim`(실습 3 시점)과 `.venv-vla`(측정 시점) 양쪽에서 같은 값으로 재현됐다.

seed 목록에서 seed 8 을 빼지 않는다. 빼면 `PickCube-v1` 의 원래 초기 배치 분포보다 인위적으로 어려운 조건이 되어 공개 수치와의 비교 가능성이 떨어진다. 대신 **하한을 기록해 해석 단계에서 걷어낸다** (§6.1 마지막 행).

## 8. 이 조건이 확정하지 않은 것

- **`bridge_orig` 의 5 Hz 는 이 환경에서 측정한 값이 아니다.** §4.3 의 action repeat 4 와 §4 의 단위 판정이 이 전제 위에 있다. BridgeData V2 데이터셋 카드 또는 Open-X Embodiment 논문의 주기 표로 확인해야 하며, 틀리면 repeat 배수와 단위 판정 근거가 함께 바뀐다.
- **공개 수치 대조를 하지 않았다.** OpenVLA 의 sim zero-shot 성적을 같은 task 에서 보고한 외부 수치와 맞춰 보지 않았다. 이것이 없으면 남은 두 가설(도메인 갭 / 변환 레이어의 잔여 버그)을 더 가를 수 없다.
- **명령 문장 1개만 썼다.** `"pick up the cube"` 고정이며, 문장 표현을 바꿨을 때의 민감도는 재지 않았다.
- **카메라 시점 1개만 썼다.** `base_camera` 고정이며, OpenVLA 학습 분포(bridge 의 3인칭 시점)와의 정합은 확인하지 않았다.
- **task 의 성공 조건과 명령 문장이 대응하지 않는다.** success 는 큐브를 공중의 목표 지점에 두는 것을 요구하지만, 명령 문장은 목표 위치를 담고 있지 않다. 즉 `placed` 는 조건상 도달하기 어려운 지표이고, 신호를 볼 자리는 그 앞 3단계다.
