"""
실습 5: 기성 해법으로 성공률 상한을 확인 (루프·환경·판정 검증)

ManiSkill 내장 motion planning 해법은 이 환경에서 쓸 수 없다. mani_skill 3.0.1 이 핀으로 박은
mplib==0.1.1 바이너리가 numpy 1.x C API 로 빌드돼 있어 numpy 2.5.1 과 ABI 가 안 맞고,
mplib.Planner 생성 시점에 세그폴트로 죽는다 (numpy 를 1.26 으로 내리면 scipy/opencv 가 깨진다).
그래서 README §8 이 함께 허용하는 다른 기성 해법 — scripted 정책 — 으로 상한을 잡는다.

이 scripted 정책은 카메라를 보지 않고 큐브·목표 좌표를 시뮬레이터에서 직접 읽는다 (인식 문제를 우회).
motion planning 과 달리 실습 6 과 **같은 control_mode(pd_ee_delta_pose)** 를 쓰므로
정규화 action 경로까지 함께 검증된다. 단 OpenVLA 출력을 변환하는 레이어는 지나가지 않으므로
변환 레이어는 이 검증으로 배제되지 않는다.
"""
import json                                    # 결과를 원시 형태로 저장하기 위해
import numpy as np
import gymnasium as gym
import mani_skill.envs


ENV_ID = "PickCube-v1"                         # <- 실습 2 확정값
MAX_EPISODE_STEPS = 200                        # <- 실습 4-3 확정값 (실습 6 과 동일해야 한다)
SEEDS = list(range(20))                        # <- 실습 6 과 공유할 고정 목록
POS_LIMIT = 0.1                                # pd_ee_delta_pose 의 위치 한계 (m). action 1.0 = 0.1 m
MAX_STEP_M = 0.03                              # 한 step 에 요청할 최대 이동량 (m). 크게 잡으면 PD 추종이 흔들린다
APPROACH_HEIGHT = 0.05                         # 큐브 위 어느 높이에서 하강을 시작할지 (m)
CLOSE_STEPS = 8                                # 그리퍼가 실제로 닫히기까지 기다리는 step 수


def to_vec(pose_field):
    """배치 텐서로 오는 좌표를 (3,) numpy 벡터로 바꾼다.

    Args:
        pose_field: `pose.p` 같은 (1, 3) 형태의 GPU 텐서

    Returns:
        (3,) float numpy 배열
    """
    return np.asarray(pose_field.cpu())[0]     # GPU -> CPU -> numpy, 배치 차원 제거


def run_episode(env, seed):
    """scripted 정책으로 한 episode 를 수행한다.

    좌표를 직접 읽어 접근 -> 하강 -> 파지 -> 목표 이동 -> 정지 순서로 진행한다.
    회전 델타는 전부 0 이다 — Panda 의 초기 자세가 이미 그리퍼를 아래로 향하고 있어
    위치 제어만으로 큐브를 잡을 수 있다.

    Args:
        env: `pd_ee_delta_pose` 로 생성된 ManiSkill 환경
        seed: 초기 배치를 재현하는 정수

    Returns:
        (성공 여부, 소비한 env step 수, 마지막 info 의 부분 조건 딕셔너리)
    """
    base = env.unwrapped                       # 큐브·목표 좌표는 wrapper 를 벗겨야 보인다
    obs, info = env.reset(seed=seed)            # 고정 seed 로 초기 배치 재현
    phase = "above"                            # 현재 단계 (above -> descend -> close -> lift -> hold)
    close_count = 0                            # 그리퍼 닫기 명령을 몇 step 유지했는지

    for step in range(MAX_EPISODE_STEPS):
        tcp = to_vec(base.agent.tcp.pose.p)    # 그리퍼 끝(TCP) 현재 위치
        cube = to_vec(base.cube.pose.p)        # 큐브 현재 위치 (특권 정보 — 카메라를 안 본다)
        goal = to_vec(base.goal_site.pose.p)   # 목표 지점 위치
        grip = 1.0                             # 기본은 열림 (+1 = 열림, 계약 표 5번)

        if phase == "above":                   # 1) 큐브 위쪽으로 이동
            target = cube + np.array([0.0, 0.0, APPROACH_HEIGHT])
            if np.linalg.norm(target - tcp) < 0.008:      # 충분히 도달하면 다음 단계
                phase = "descend"
        elif phase == "descend":               # 2) 큐브 중심 높이까지 하강
            target = cube
            if np.linalg.norm(target - tcp) < 0.006:
                phase = "close"
        elif phase == "close":                 # 3) 제자리에서 그리퍼 닫기
            target = tcp                       # 이동 없음 (델타 0)
            grip = -1.0                        # -1 = 닫힘
            close_count += 1
            if close_count >= CLOSE_STEPS:
                phase = "lift"
        elif phase == "lift":                  # 4) 큐브를 든 채 목표 지점으로 이동
            target = goal
            grip = -1.0
            if np.linalg.norm(goal - cube) < 0.02:        # goal_thresh(0.025) 보다 보수적으로
                phase = "hold"
        else:                                  # 5) 정지 — success 는 is_robot_static 도 요구한다
            target = tcp
            grip = -1.0

        delta = np.clip(target - tcp, -MAX_STEP_M, MAX_STEP_M)     # step당 이동량 제한
        action = np.concatenate([
            delta / POS_LIMIT,                 # 미터 -> [-1, 1] 정규화 (계약 표 1번과 같은 규칙)
            np.zeros(3),                       # 회전 델타 없음
            [grip],                            # gripper 명령
        ]).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        if bool(info["success"].item()):       # 성공 즉시 종료
            return True, step + 1, info
        if terminated or truncated:            # 환경이 스스로 끝냈으면 종료
            break

    return False, step + 1, info               # step 예산을 다 쓰고 실패


print("=" * 60)
print("실습 5: 상한 대조 (scripted 정책)")
print("=" * 60)


# -- 5-1. 실습 6 과 같은 조건으로 환경 생성 --
env = gym.make(
    ENV_ID,
    obs_mode="rgb",                            # 이 해법은 이미지를 안 쓰지만 조건을 맞춰 둔다
    control_mode="pd_ee_delta_pose",           # 실습 6 과 동일 — 정규화 action 경로까지 검증된다
    render_mode="rgb_array",                   # 헤드리스
    sensor_configs=dict(width=224, height=224),   # 실습 6 과 동일한 관측 카메라 설정
    max_episode_steps=MAX_EPISODE_STEPS,       # 실습 6 과 같은 env step 예산
)


# -- 5-2. 20 episode 실행 --
print("\n[5-2] scripted 해법 20 episode")
success_count = 0                              # 성공 episode 수
records = []                                   # episode 별 결과
for seed in SEEDS:
    solved, steps, info = run_episode(env, seed)
    stages = {                                 # 실패한 경우 어디까지 갔는지 남긴다
        "is_grasped": bool(info["is_grasped"].item()),
        "is_obj_placed": bool(info["is_obj_placed"].item()),
        "is_robot_static": bool(info["is_robot_static"].item()),
    }
    success_count += int(solved)               # 성공이면 1 누적
    records.append({"seed": seed, "solved": solved, "steps": steps, **stages})
    print(f"   seed{seed:02d}: solved={solved} steps={steps} grasped={stages['is_grasped']}")


print(f"\n상한 성공률: {success_count}/{len(SEEDS)}")
steps_used = [record["steps"] for record in records]
print(f"소비 step: 최소 {min(steps_used)} / 최대 {max(steps_used)} / 평균 {sum(steps_used) / len(steps_used):.1f}"
      f" (실습 6 예산 {MAX_EPISODE_STEPS})")


# -- 5-3. 하한 대조 — 아무것도 하지 않는 정책의 성공률 --
# PickCube 는 목표 지점을 무작위로 뽑으므로, 목표가 큐브 초기 위치에서 goal_thresh 안에
# 떨어지는 seed 가 섞인다. 그 seed 는 조작 없이도 success 가 되어 성공률을 공짜로 올린다.
# 실습 6 의 성공률을 해석하려면 이 공짜분을 먼저 알아야 한다.
print("\n[5-3] 무행동 정책 20 episode (하한)")
zero_action = np.zeros(7, dtype=np.float32)    # 전 차원 0 = 이동·회전 없음
noop_hits = []                                 # 무행동으로 성공한 seed 목록
for seed in SEEDS:
    obs, info = env.reset(seed=seed)
    for step in range(MAX_EPISODE_STEPS):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        if bool(info["success"].item()):       # 조작 없이 성공 -> 애초에 목표 안에 있던 배치
            noop_hits.append(seed)
            break
        if terminated or truncated:
            break
print(f"   무행동 성공 seed: {noop_hits} -> {len(noop_hits)}/{len(SEEDS)}")


# -- 5-4. 원시 결과 저장 (outputs/harness_check.md 작성의 근거) --
with open("outputs/harness_check.json", "w") as f:
    json.dump({"env_id": ENV_ID, "control_mode": "pd_ee_delta_pose",
               "max_episode_steps": MAX_EPISODE_STEPS, "solver": "scripted (privileged state)",
               "seeds": SEEDS, "upper_bound": success_count, "records": records,
               "noop_lower_bound": len(noop_hits), "noop_success_seeds": noop_hits}, f, indent=2)
print("저장 완료: outputs/harness_check.json")


env.close()
