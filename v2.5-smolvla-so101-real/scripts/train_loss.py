"""train.log 에서 step 별 loss · grad norm · lr 을 뽑아 csv 와 그림으로 남긴다 (Measurements environment.md 용).

사용:
  /workspace/venvs/lerobot/bin/python /workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/scripts/train_loss.py
출력: outputs/train_loss.csv, outputs/plots/train_loss.png, 요약 (첫/마지막 step 시각, 벽시계 시간, loss 최소 · 마지막)
"""

import csv
import datetime
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUTS = Path("/workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/outputs")
LOG = OUTPUTS / "train.log"
# 예: INFO 2026-10-22 23:40:18 ot_train.py:778 step:200 smpl:6K ep:7 epch:0.14 loss:0.297 grdn:3.085 lr:1.5e-05 data_s:0.027 prep_s:0.005
# step 은 1000 이상이면 "2K" 처럼 줄여 찍힌다 — 진행 막대의 "n/20000" 에서 정확한 step 을 읽는다
PAT = re.compile(r"\| (\d+)/\d+ \[.*INFO (\S+ \S+) \S+ step:\S+ .*?loss:([\d.]+) grdn:([\d.]+) lr:([\d.e+-]+)")


def main():
    """로그를 파싱해 csv · 그림 · 요약을 만든다."""
    rows = []
    # 진행 막대와 INFO 줄이 \r 로만 나뉘어 한 줄에 붙어 있으므로 \n 기준으로만 자른다
    for line in LOG.read_text(errors="replace").splitlines():
        for m in PAT.finditer(line.replace("\r", " ")):
            rows.append({"time": m.group(2), "step": int(m.group(1)), "loss": float(m.group(3)), "grad_norm": float(m.group(4)), "lr": float(m.group(5))})
    if not rows:
        raise SystemExit("step 줄이 없다")
    with open(OUTPUTS / "train_loss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    t0 = datetime.datetime.fromisoformat(rows[0]["time"])
    t1 = datetime.datetime.fromisoformat(rows[-1]["time"])
    losses = [r["loss"] for r in rows]
    print(f"step {rows[0]['step']} ({rows[0]['time']}) -> step {rows[-1]['step']} ({rows[-1]['time']}), 벽시계 {(t1 - t0).total_seconds() / 3600:.2f} h")
    print(f"loss 첫 {losses[0]:.3f} / 최소 {min(losses):.3f} (step {rows[losses.index(min(losses))]['step']}) / 마지막 {losses[-1]:.3f}")
    print(f"마지막 1000 step 평균 loss {sum(losses[-5:]) / len(losses[-5:]):.3f}")
    (OUTPUTS / "plots").mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot([r["step"] for r in rows], losses)
    plt.xlabel("step")
    plt.ylabel("loss (flow matching)")
    plt.title("SmolVLA fine-tune on so101-pick-cube-v25 (batch 32)")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "plots/train_loss.png")
    print("저장:", OUTPUTS / "train_loss.csv", OUTPUTS / "plots/train_loss.png")


if __name__ == "__main__":
    main()
