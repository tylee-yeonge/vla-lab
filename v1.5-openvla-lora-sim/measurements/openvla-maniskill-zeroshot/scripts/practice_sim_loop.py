"""
실습 3: random action 으로 episode 를 돌리고 success 정의·action 공간을 확정
"""
import inspect                                 # 성공 판정 코드를 직접 읽기 위해
import gymnasium as gym
import mani_skill.envs


ENV_ID = "PickCube-v1"                         # <- 실습 2 에서 확정한 id


print("=" * 60)
print("실습 3: sim 단독 루프")
print("=" * 60)


env = gym.make(ENV_ID, obs_mode="rgb", control_mode="pd_ee_delta_pose")


# -- 3-1. action 공간의 실제 정의 (범위 규약 확정 — 변환 계약 2번 항목) --
print("\n[3-1] action space")
print("  ", env.action_space)                  # 차원 수와 상한/하한이 그대로 출력된다
print("   low :", env.action_space.low)        # 하한 — -1 이면 정규화된 입력을 기대한다는 뜻
print("   high:", env.action_space.high)       # 상한


# -- 3-2. 선택 가능한 control_mode 목록 --
print("\n[3-2] control modes")
print("  ", env.unwrapped.agent.supported_control_modes)   # 실제 지원 목록 (문서와 다를 수 있다)


# -- 3-3. 성공 판정의 소스코드를 직접 읽는다 (임계값을 추측하지 않는다) --
print("\n[3-3] 성공 판정 코드")
print(inspect.getsource(type(env.unwrapped).evaluate))     # evaluate 가 success 를 계산한다


# -- 3-4. 상태 접근 경로 탐색 (부분 도달률 계산에 필요 — 실습 6) --
print("\n[3-4] env 내부 접근 가능한 이름")
print("  ", [name for name in dir(env.unwrapped) if not name.startswith("_")])


# -- 3-5. random action 으로 20 episode (step cap 확정) --
print("\n[3-5] random action 20 episode")
STEP_CAP = 100                                 # 잠정값. 아래 출력으로 타당성을 판정한다
for episode in range(20):                      # seed 를 바꿔 20회
    obs, info = env.reset(seed=episode)        # 매번 다른 초기 배치
    success = False                            # 이 episode 의 성공 여부
    for step in range(STEP_CAP):               # step cap 까지만 시도
        action = env.action_space.sample()     # 무작위 action (정책 없음)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("success", False):         # 성공 플래그 — 키 이름은 3-3 에서 확인한 것으로
            success = True
            break                              # 성공 즉시 종료
        if terminated or truncated:            # 환경이 스스로 끝냈으면 종료
            break
    print(f"   ep{episode:02d}: success={success} steps={step + 1}")


env.close()
print("\n실습 3 완료")