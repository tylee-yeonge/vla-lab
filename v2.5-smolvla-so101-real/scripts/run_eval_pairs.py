"""v2.5 실기 측정 러너 — zero-shot (A) 과 fine-tuned (B) 를 같은 시작 배치에서 짝지어 N쌍 돌린다.

설계 (Measurements/so101-smolvla-v25/methodology.md):
  - 쌍 i 마다 두 모델을 연달아 1회씩 돌린다. 순서는 ABBA 블록 교차: 홀수 쌍 A→B, 짝수 쌍 B→A.
  - 한 실행 = lerobot-rollout 1회 (episodic, 30 s) = 데이터셋 폴더 1개 (lerobot 이 시각을 붙인다).
  - 실행마다 outputs/eval_pairs.jsonl 에 1줄 (쌍 번호 · 모델 · 순서 · 결과 폴더 · 시각 · 종료 코드) 을 남긴다.
  - 이미 기록된 (쌍, 모델) 은 건너뛰므로 같은 명령으로 이어서 할 수 있다. Ctrl+C 로 멈춘 실행은 aborted 로 남기고 다시 할지 묻는다.

사람이 하는 일: 실행 전 큐브를 시작 지점 (시범과 같은 자리) 에 놓고 팔이 휴식 자세인지 확인한 뒤 Enter.
에피소드를 일찍 끝내려면 → (오른쪽 화살표): 에피소드가 영상과 함께 저장되고 팔이 시작 자세로 돌아간다 — 판정 대상으로 남는다.
Ctrl+C 는 기록이 남지 않는다 (두 번 누르면 강제 종료로 영상이 버려진다) — 팔이 위험할 때만, 그다음은 전원 차단.

사용:
  /workspace/venvs/lerobot/bin/python /workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/scripts/run_eval_pairs.py --n 50 --limit-arm 6 --limit-gripper 25
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

# ---- 고정값 (PRACTICE §2 와 같다) ----
HF_USER = "tylee-yeonge"
V25 = Path("/workspace/study/physical-ai-study/Studies/Hardware-Arm/v25")
OUTPUTS = V25 / "outputs"
LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", "/root/.cache/huggingface/lerobot")) / HF_USER
ROLLOUT = "/workspace/venvs/lerobot/bin/lerobot-rollout"
FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/so101_follower")
TASK = "Pick up the pink cube and place it in the yellow square."
CAMS = (
    "{ camera1: {type: opencv, index_or_path: /dev/so101_cam_overview, width: 1280, height: 720, fps: 30, fourcc: MJPG},"
    " camera2: {type: opencv, index_or_path: /dev/so101_cam_wrist, width: 1280, height: 720, fps: 30, fourcc: MJPG} }"
)
# 모델 키 -> (체크포인트 경로, 데이터셋 이름 접미어, 로그 파일)
MODELS = {
    "zeroshot": (str(OUTPUTS / "smolvla_base_so101stats"), "zeroshot", OUTPUTS / "eval_zeroshot.log"),
    "finetuned": (str(OUTPUTS / "smolvla_v25/checkpoints/last/pretrained_model"), "finetuned", OUTPUTS / "eval_finetuned.log"),
}
RECORD = OUTPUTS / "eval_pairs.jsonl"


def load_done(path: Path) -> set:
    """기록 파일에서 정상 종료한 (쌍, 모델) 집합을 읽는다 (aborted · 비정상 종료는 안 끝난 것으로 본다)."""
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "pair" in r and not r.get("aborted") and r.get("returncode") == 0 and r.get("dataset_dir"):
                done.add((r["pair"], r["model"]))
    return done


def newest_dirs(suffix: str) -> set:
    """이 모델의 rollout 데이터셋 폴더 집합 (실행 전후 차이로 새 폴더를 찾는다)."""
    return {p.name for p in LEROBOT_HOME.glob(f"rollout_so101-pick-cube-v25-{suffix}_*")}


def ask(prompt: str, choices: str) -> str:
    """choices 중 한 글자를 받을 때까지 묻는다 (Enter 는 choices[0])."""
    while True:
        ans = input(prompt).strip().lower()
        if ans == "":
            return choices[0]
        if ans in choices:
            return ans


def run_once(pair: int, order: int, model: str, limit: dict, n: int) -> str:
    """rollout 1회. 돌아오는 값: 'ok' | 'retry' | 'skip' | 'quit'."""
    path, suffix, log = MODELS[model]
    print(f"\n=== 쌍 {pair}/{n}  {order}번째  모델 {model}  상한 {limit} ===")
    a = ask("큐브를 시작 지점에 놓고 팔이 휴식 자세인지 확인한 뒤 Enter (s: 이 실행 건너뜀, q: 종료): ", "\nsq")
    if a == "q":
        return "quit"
    if a == "s":
        return "skip"
    limit_str = "{" + ", ".join(f"{k}: {v}" for k, v in limit.items()) + "}"
    cmd = [
        ROLLOUT,
        "--strategy.type=episodic",
        f"--policy.path={path}",
        "--policy.device=cuda",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        "--robot.id=so101_follower_01",
        f"--robot.cameras={CAMS}",
        f"--robot.max_relative_target={limit_str}",
        f"--task={TASK}",
        "--fps=30",
        f"--dataset.repo_id={HF_USER}/rollout_so101-pick-cube-v25-{suffix}",
        "--dataset.num_episodes=1",
        "--dataset.episode_time_s=30",
        "--dataset.push_to_hub=false",
    ]
    before = newest_dirs(suffix)
    started = datetime.datetime.now().isoformat(timespec="seconds")
    aborted = False
    with open(log, "a") as lf:
        lf.write(f"\n##### run_eval_pairs pair={pair} order={order} model={model} started={started}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            # Ctrl+C 는 자식 (lerobot-rollout) 에도 전달된다 — 정리 (팔 복귀 · 연결 해제) 를 기다린다
            aborted = True
            print("\n[중단] rollout 정리를 기다린다 (팔이 시작 자세로 돌아간다)...")
            rc = proc.wait()
    ended = datetime.datetime.now().isoformat(timespec="seconds")
    new = sorted(newest_dirs(suffix) - before)
    dataset_dir = str(LEROBOT_HOME / new[-1]) if new else None
    rec = {"pair": pair, "model": model, "order": order, "dataset_dir": dataset_dir, "started": started,
           "ended": ended, "returncode": rc, "aborted": aborted, "limit": limit, "policy_path": path}
    with open(RECORD, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ok = (rc == 0) and (not aborted) and (dataset_dir is not None)
    print(f"[{'정상' if ok else '실패'}] rc={rc} aborted={aborted} 폴더={dataset_dir}")
    if ok:
        return "ok"
    a = ask("다시 할까? (Enter: 다시, n: 다음으로, q: 종료): ", "\nnq")
    return {"\n": "retry", "n": "skip", "q": "quit"}[a]


def main():
    """ABBA 스케줄로 N쌍을 돌린다."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="쌍 수 (methodology §1.3)")
    ap.add_argument("--limit-arm", type=float, default=6.0, help="팔 5관절의 max_relative_target (도/틱) — 착수 전 ② 측정값")
    ap.add_argument("--limit-gripper", type=float, default=25.0, help="그리퍼의 max_relative_target (도/틱)")
    ap.add_argument("--only", choices=list(MODELS), default=None, help="한 모델만 돌릴 때 (디버그용 — 측정에는 쓰지 않는다)")
    args = ap.parse_args()

    for model, (path, _, _) in MODELS.items():
        if args.only in (None, model) and not Path(path).exists():
            sys.exit(f"체크포인트가 없다: {path}")
    limit = {j: args.limit_arm for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")}
    limit["gripper"] = args.limit_gripper
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if not RECORD.exists():
        meta = {"_meta": {"task": TASK, "n_pairs": args.n, "limit": limit, "order": "ABBA (odd pair A->B, even pair B->A)",
                          "models": {k: v[0] for k, v in MODELS.items()}, "created": datetime.datetime.now().isoformat(timespec="seconds")}}
        RECORD.write_text(json.dumps(meta, ensure_ascii=False) + "\n")
    done = load_done(RECORD)
    print(f"기록: {RECORD}  (끝난 실행 {len(done)}개)")

    for pair in range(1, args.n + 1):
        order = ["zeroshot", "finetuned"] if pair % 2 == 1 else ["finetuned", "zeroshot"]
        for k, model in enumerate(order, start=1):
            if args.only and model != args.only:
                continue
            if (pair, model) in done:
                continue
            while True:
                r = run_once(pair, k, model, limit, args.n)
                if r == "quit":
                    print("종료. 같은 명령으로 이어서 할 수 있다.")
                    return
                if r != "retry":
                    break
    print("\n전부 끝났다. 판정은 judge_episodes.py, 집계는 analyze_results.py.")


if __name__ == "__main__":
    main()
