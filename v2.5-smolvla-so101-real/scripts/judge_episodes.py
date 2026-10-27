"""측정 실행 (eval_pairs.jsonl) 의 영상에서 4단계 판정의 자동 부분과 사람 판정용 필름스트립을 만든다.

영상 판정 규칙 (Measurements/so101-smolvla-v25/methodology.md §1.2) 중 자동으로 되는 것:
  - placed_auto : 마지막 프레임에서 큐브 중심이 노란 테이프의 바깥 경계 안에 있고, 마지막 2초 동안 큐브가 정지
  - reached_auto: 어느 프레임에서든 보라색 (팔 · 그리퍼) 픽셀이 큐브 bbox 를 큐브 한 변의 0.5 배 넓힌 영역에 들어옴
  - moved_auto  : 큐브 중심이 시작 위치에서 큐브 한 변 이상 벗어난 적이 있음 (밀림 · 집힘 모두 포함)
grasped · lifted 는 자동으로 못 가르므로 필름스트립 (전체 뷰 + 손목 뷰, 3초 간격 10장) 을 보고 사람이 적는다.

출력:
  outputs/eval_judge.csv            — 실행별 자동 수치 + 최종 판정 열 (reached/grasped/lifted/placed). 자동값으로 채워 두고 사람이 고친다
  outputs/eval_judge/pairNN_model.png — 필름스트립

사용:
  /workspace/venvs/lerobot/bin/python /workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/scripts/judge_episodes.py
"""

import csv
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
from PIL import ImageDraw

OUTPUTS = Path("/workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/outputs")
RECORD = OUTPUTS / "eval_pairs.jsonl"
JUDGE_CSV = OUTPUTS / "eval_judge.csv"
STRIP_DIR = OUTPUTS / "eval_judge"
W, H = 640, 360
STEP = 5  # 30 fps 영상에서 5 프레임마다 (6 fps) 판정
STRIP_SECONDS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
FIELDS = ["pair", "model", "order", "dataset_dir", "frames", "cube_w_px", "start_x", "start_y", "end_x", "end_y",
          "yellow_bbox", "end_in_square", "still_last2s", "occluded_frames", "reached_auto", "moved_auto", "placed_auto",
          "start_in_square", "reached", "grasped", "lifted", "placed", "exclude", "judged_by", "note"]


def hsv(a):
    """RGB 배열 (H, W, 3) 을 PIL HSV (각 0-255) 정수 배열로."""
    return np.asarray(Image.fromarray(a).convert("HSV")).astype(int)


def table_only(m):
    """작업 영역 (마우스 패드와 로봇 베이스 사이의 책상, x 150-400 · y 120-340) 밖을 마스크에서 뺀다."""
    m[:, :150] = False
    m[:, 400:] = False
    m[:120, :] = False
    m[340:, :] = False
    return m


def pink_mask(a):
    """분홍 큐브 픽셀 (살구색: R 이 크고 G 와 B 가 비슷). 윗면은 노출 과다로 거의 흰색이라 옆면 픽셀로 잡힌다."""
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return table_only((r > 170) & (r - g > 45) & (r - b > 45) & (b > 70) & (g > 70) & (np.abs(g - b) < 35))


def yellow_mask(a):
    """노란 테이프 픽셀 (G 가 B 보다 70 이상 큼 — 책상 나무결은 20-30)."""
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return table_only((r > 150) & (g > 170) & (b < 160) & (g - b > 70))


def purple_mask(a):
    """보라색 팔 · 그리퍼 픽셀 (H 175-205)."""
    h, s, v = hsv(a)[..., 0], hsv(a)[..., 1], hsv(a)[..., 2]
    return (h >= 175) & (h <= 205) & (s >= 60) & (v >= 80)


def blob(mask, min_px=30):
    """마스크의 중심 (중앙값) 과 bbox (x0, y0, x1, y1). bbox 는 2-98 백분위로 잡아 흩어진 픽셀을 무시한다. 픽셀이 적으면 None."""
    ys, xs = np.nonzero(mask)
    if len(xs) < min_px:
        return None
    x0, x1 = np.percentile(xs, [2, 98])
    y0, y1 = np.percentile(ys, [2, 98])
    return float(np.median(xs)), float(np.median(ys)), int(x0), int(y0), int(x1), int(y1)


def decode(path: Path):
    """영상을 (시각, 640x360 PIL 이미지) 로 전부 디코딩한다 (30 s 짜리 1개라 메모리에 둔다)."""
    out = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            out.append((float(fr.pts * st.time_base), fr.to_image().resize((W, H))))
    return out


def judge_one(rec: dict) -> dict:
    """실행 하나를 자동 판정하고 필름스트립을 저장한다."""
    d = Path(rec["dataset_dir"])
    over = decode(d / "videos/observation.images.camera1/chunk-000/file-000.mp4")
    wrist = decode(d / "videos/observation.images.camera2/chunk-000/file-000.mp4")
    first = np.asarray(over[0][1])
    yb = blob(yellow_mask(first), min_px=60)
    if yb is not None and not (25 <= yb[4] - yb[2] <= 90 and 20 <= yb[5] - yb[3] <= 80):
        yb = None  # 테이프 크기가 아니면 (책상 나무결 등) 미검출로 본다
    cube0 = blob(pink_mask(first))
    row = {"pair": rec["pair"], "model": rec["model"], "order": rec["order"], "dataset_dir": str(d), "frames": len(over), "start_in_square": None,
           "yellow_bbox": yb[2:] if yb else None, "cube_w_px": None, "start_x": None, "start_y": None, "end_x": None, "end_y": None,
           "end_in_square": None, "still_last2s": None, "occluded_frames": 0, "reached_auto": False, "moved_auto": False, "placed_auto": None}
    if cube0 is None or yb is None:
        row["note"] = "첫 프레임에서 큐브 또는 노란 칸 미검출 — 사람이 판정"
        cube_w = 17.0
    else:
        cube_w = max(cube0[4] - cube0[2], cube0[5] - cube0[3], 8)
        row.update({"cube_w_px": cube_w, "start_x": round(cube0[0], 1), "start_y": round(cube0[1], 1)})
        # degenerate 배치 (methodology §1.4): 시작 프레임에서 큐브 중심이 이미 노란 칸 안이면 그 쌍을 제외
        row["start_in_square"] = bool(yb[2] <= cube0[0] <= yb[4] and yb[3] <= cube0[1] <= yb[5])
    centers = []
    for i in range(0, len(over), STEP):
        a = np.asarray(over[i][1])
        c = blob(pink_mask(a))
        if c is None:
            row["occluded_frames"] += 1
            centers.append(None)
            continue
        centers.append((c[0], c[1]))
        if cube0 is not None:
            if np.hypot(c[0] - cube0[0], c[1] - cube0[1]) > cube_w:
                row["moved_auto"] = True
        # 큐브 bbox 를 0.5 변 넓힌 영역에 보라 픽셀이 있으면 reached
        x0, y0, x1, y1 = int(c[2] - 0.5 * cube_w), int(c[3] - 0.5 * cube_w), int(c[4] + 0.5 * cube_w), int(c[5] + 0.5 * cube_w)
        pm = purple_mask(a)[max(0, y0):y1, max(0, x0):x1]
        if pm.sum() > 20:
            row["reached_auto"] = True
    last = [c for c in centers[-max(1, 2 * 30 // STEP):] if c is not None]
    end = centers[-1] if centers else None
    if end is not None and yb is not None:
        row.update({"end_x": round(end[0], 1), "end_y": round(end[1], 1)})
        row["end_in_square"] = bool(yb[2] <= end[0] <= yb[4] and yb[3] <= end[1] <= yb[5])
        row["still_last2s"] = bool(len(last) >= 2 and max(np.hypot(p[0] - last[-1][0], p[1] - last[-1][1]) for p in last) < 0.3 * cube_w)
        row["placed_auto"] = bool(row["end_in_square"] and row["still_last2s"])
    elif end is None:
        row["note"] = (row.get("note", "") + " 마지막 프레임 큐브 미검출 (그리퍼에 들려 있거나 가려짐)").strip()
    # 필름스트립: 윗줄 전체 뷰, 아랫줄 손목 뷰
    strip = Image.new("RGB", (256 * len(STRIP_SECONDS), 288))
    dr = ImageDraw.Draw(strip)
    for k, sec in enumerate(STRIP_SECONDS):
        for rowi, frames in enumerate((over, wrist)):
            idx = min(int(sec * 30), len(frames) - 1)
            strip.paste(frames[idx][1].resize((256, 144)), (k * 256, rowi * 144))
        dr.text((k * 256 + 4, 2), f"{sec}s", fill=(255, 255, 0))
    dr.text((4, 150), f"pair {rec['pair']} {rec['model']}", fill=(255, 255, 0))
    STRIP_DIR.mkdir(parents=True, exist_ok=True)
    strip.save(STRIP_DIR / f"pair{rec['pair']:02d}_{rec['model']}.png")
    # 최종 판정 열: 자동값으로 채우고 (grasped · lifted 는 비움) 사람이 고친다
    row.update({"reached": row["reached_auto"], "grasped": "", "lifted": "", "placed": row["placed_auto"],
                "exclude": bool(row.get("start_in_square")), "judged_by": "auto"})
    row.setdefault("note", "")
    return row


def main():
    """기록된 정상 실행을 전부 판정한다. 이미 판정된 (CSV 에 있는) 실행은 건너뛴다."""
    rows = []
    done = set()
    if JUDGE_CSV.exists():
        with open(JUDGE_CSV) as f:
            rows = list(csv.DictReader(f))
        done = {(int(r["pair"]), r["model"]) for r in rows}
    recs = [json.loads(l) for l in RECORD.read_text().splitlines() if l.strip()]
    todo = [r for r in recs if "pair" in r and not r.get("aborted") and r.get("returncode") == 0 and r.get("dataset_dir")
            and (r["pair"], r["model"]) not in done]
    print(f"판정할 실행 {len(todo)}개 (이미 {len(done)}개)")
    for rec in todo:
        row = judge_one(rec)
        rows.append({k: row.get(k, "") for k in FIELDS})
        print(f"  pair {rec['pair']:2d} {rec['model']:9s} reached_auto={row['reached_auto']} moved={row['moved_auto']} placed_auto={row['placed_auto']} occluded={row['occluded_frames']}")
    with open(JUDGE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("저장:", JUDGE_CSV, "/ 필름스트립:", STRIP_DIR)


if __name__ == "__main__":
    main()
