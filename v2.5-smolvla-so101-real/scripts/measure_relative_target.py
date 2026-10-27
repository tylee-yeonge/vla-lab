"""착수 전 ② — max_relative_target 의 하한 실측: 시범 에피소드 하나를 상한값별로 재생하고 추종 오차를 잰다.

lerobot-replay 와 같은 방식 (에피소드의 action 을 30 fps 로 팔에 보낸다) 에 두 가지를 더한다:
  - 매 틱 팔의 실제 위치를 읽어 |명령 - 실제| 를 기록한다 (상한이 작으면 팔이 명령을 못 따라간다).
  - 상한값을 바꿔 가며 같은 에피소드를 반복한다 (기본 3, 6, 10, 15). 그리퍼는 별도 상한 (기본 25).
큐브를 시범과 같은 시작 지점에 놓으면 재생이 실제로 집어 옮기므로 "그 상한으로 과제가 되는가" 도 눈으로 본다.

사람이 하는 일: 큐브 놓기, 재생마다 Enter, 위험하면 Ctrl+C (팔은 그 자리에 멈추고 토크가 풀린다).

사용:
  /workspace/venvs/lerobot/bin/python /workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/scripts/measure_relative_target.py --episode 1 --limits 3 6 10 15
결과: outputs/relative_target_<episode>.json + 표 출력. 판정 기준은 methodology.md §1.4.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

HF_USER = "tylee-yeonge"
DS = f"{HF_USER}/so101-pick-cube-v25"
OUTPUTS = Path("/workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/outputs")
FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/so101_follower")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = JOINTS[:5]


def replay(robot: SO101Follower, actions: np.ndarray, fps: int) -> np.ndarray:
    """에피소드의 action 을 순서대로 보내며 매 틱 실제 위치를 읽는다. 돌아오는 값: (T, 6) 실제 위치."""
    states = np.zeros_like(actions)
    period = 1.0 / fps
    for t, a in enumerate(actions):
        t0 = time.perf_counter()
        obs = robot.get_observation()
        states[t] = [obs[f"{j}.pos"] for j in JOINTS]
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, a)})
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    return states


def settle(robot: SO101Follower, target: np.ndarray, seconds: float, fps: int) -> None:
    """같은 목표를 seconds 동안 반복해 보내 팔을 그 자세에 세운다 (재생 뒤 휴식 자세 복귀용)."""
    for _ in range(int(seconds * fps)):
        t0 = time.perf_counter()
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, target)})
        time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))


def summarize(actions: np.ndarray, states: np.ndarray, fps: int) -> dict:
    """추종 오차 요약: 관절별 |a-s| p95 · max, 3초 넘게 오차가 3도 이상 유지된 구간 (정지 의심), 끝 자세 오차."""
    err = np.abs(actions - states)
    out = {"p95": np.percentile(err, 95, axis=0).round(1).tolist(),
           "max": err.max(axis=0).round(1).tolist(),
           "final_err": err[-1].round(1).tolist()}
    stalls = {}
    for j, name in enumerate(ARM):
        over = err[:, j] > 3.0
        run = best = 0
        for v in over:
            run = run + 1 if v else 0
            best = max(best, run)
        stalls[name] = round(best / fps, 1)  # 오차가 3도를 넘긴 가장 긴 연속 구간 (초)
    out["longest_over3_s"] = stalls
    return out


def main():
    """상한값별로 같은 에피소드를 재생하고 결과를 json 으로 남긴다."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, default=1, help="재생할 시범 에피소드 (0 은 시작 자세가 특이해 피한다)")
    ap.add_argument("--limits", type=float, nargs="+", default=[3, 6, 10, 15], help="팔 5관절 상한 후보 (도/틱)")
    ap.add_argument("--limit-gripper", type=float, default=25.0, help="그리퍼 상한 (도/틱)")
    args = ap.parse_args()

    ds = LeRobotDataset(DS, episodes=[args.episode])
    actions = np.stack([ds[i]["action"].numpy() for i in range(len(ds))]).astype(float)
    fps = ds.fps
    print(f"에피소드 {args.episode}: {len(actions)} 프레임, {fps} fps, 첫 action {actions[0].round(1)}, 끝 action {actions[-1].round(1)}")

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="so101_follower_01", cameras={}, max_relative_target=3.0))
    robot.connect()
    results = {}
    try:
        for L in args.limits:
            limit = {j: float(L) for j in ARM}
            limit["gripper"] = args.limit_gripper
            robot.config.max_relative_target = limit  # send_action 이 매번 config 를 읽는다
            input(f"\n상한 {L} (그리퍼 {args.limit_gripper}): 큐브를 시작 지점에 놓고 Enter ...")
            states = replay(robot, actions, fps)
            settle(robot, actions[-1], 3.0, fps)  # 에피소드 끝 자세 (휴식) 에 세운다
            s = summarize(actions, states, fps)
            results[str(L)] = s
            print(f"  |a-s| p95 {s['p95']}  max {s['max']}  끝 오차 {s['final_err']}")
            print(f"  3도 초과 최장 연속 (초) {s['longest_over3_s']}")
            np.save(OUTPUTS / f"relative_target_ep{args.episode}_L{L:g}_states.npy", states)
    except KeyboardInterrupt:
        print("\n[중단] 토크를 풀고 연결을 끊는다.")
    finally:
        robot.disconnect()
    out = OUTPUTS / f"relative_target_ep{args.episode}.json"
    out.write_text(json.dumps({"episode": args.episode, "gripper_limit": args.limit_gripper, "results": results}, indent=2))
    print("저장:", out)


if __name__ == "__main__":
    main()
