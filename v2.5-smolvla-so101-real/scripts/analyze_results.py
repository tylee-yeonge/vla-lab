"""v2.5 실기 측정 집계 — zero-shot vs fine-tuned 의 4단계 성공 수, placed 의 Wilson 구간, 짝지은 exact McNemar.

방법의 정본은 Studies/Phase 4.5/week5/outputs/stat_method.md (v1.5 와 같다). 여기서 방법을 새로 고르지 않는다.
  - 1차 지표 placed: 쌍이 완성된 N쌍 기준 Wilson 95% 구간
  - reached / grasped / lifted: 기술 통계만 (구간 · 검정 없음)
  - 짝지은 비교: placed 불일치 쌍의 exact 이항검정, 양측, alpha 0.05. 불일치 쌍 6 미만이면 판정 불가
  - degenerate 배치 제외: 판정 CSV 의 exclude 가 참인 실행이 있는 쌍을 뺀다 (시작 프레임에 큐브가 이미 노란 칸 안 — methodology §1.4)
입력: outputs/eval_pairs.jsonl (실행 기록) + outputs/eval_judge.csv (최종 판정 — 사람이 고친 열 reached/grasped/lifted/placed)
출력: 표 (stdout, findings.md 에 그대로 붙인다) + outputs/plots/stage_counts.png

사용:
  /workspace/venvs/lerobot/bin/python /workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/scripts/analyze_results.py
"""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUTS = Path("/workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/outputs")
STAGES = ["reached", "grasped", "lifted", "placed"]
ALPHA = 0.05
MODELS = ["zeroshot", "finetuned"]


def truthy(v) -> bool:
    """CSV 의 'True'/'False'/'' 를 bool 로."""
    return str(v).strip().lower() in ("true", "1", "yes", "o")


def main():
    """판정 CSV 를 읽어 집계하고 표를 출력한다."""
    meta = json.loads(OUTPUTS.joinpath("eval_pairs.jsonl").read_text().splitlines()[0])["_meta"]
    with open(OUTPUTS / "eval_judge.csv") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(int(r["pair"]), {})[r["model"]] = r
    complete = sorted(p for p, d in by.items() if all(m in d for m in MODELS))
    # degenerate 배치 (methodology §1.4) — 한 실행이라도 exclude 면 그 쌍 전체를 뺀다
    excluded = [p for p in complete if any(truthy(by[p][m].get("exclude", "")) for m in MODELS)]
    pairs = [p for p in complete if p not in excluded]
    n = len(pairs)
    unjudged = [(p, m) for p in pairs for m in MODELS if by[p][m]["judged_by"] != "manual"]
    print(f"완성된 쌍 {len(complete)}, 제외 {excluded} -> 집계 N = {n} (사람이 확정하지 않은 실행 {len(unjudged)}개)")
    print(f"조건: task='{meta['task']}', 상한 {meta['limit']}, 순서 {meta['order']}")

    print("\n[1] 단계별 성공 수 (N쌍 기준)")
    print("| 단계 | zero-shot | fine-tuned |")
    print("|---|---|---|")
    counts = {}
    for st in STAGES:
        counts[st] = {m: sum(truthy(by[p][m][st]) for p in pairs) for m in MODELS}
        cells = []
        for m in MODELS:
            k = counts[st][m]
            if st == "placed" and n > 0:
                ci = stats.binomtest(k, n).proportion_ci(confidence_level=1 - ALPHA, method="wilson")
                cells.append(f"{k}/{n} (Wilson {100 * ci.low:.1f}-{100 * ci.high:.1f}%)")
            else:
                cells.append(f"{k}/{n}")
        print(f"| {st} | {cells[0]} | {cells[1]} |")

    print("\n[2] 짝지은 비교 — placed (같은 쌍 번호끼리)")
    both = only_z = only_f = neither = 0
    for p in pairs:
        z, f = truthy(by[p]["zeroshot"]["placed"]), truthy(by[p]["finetuned"]["placed"])
        both += z and f
        only_z += z and not f
        only_f += f and not z
        neither += (not z) and (not f)
    disc = only_z + only_f
    print(f"둘 다 {both} / zero-shot 만 {only_z} / fine-tuned 만 {only_f} / 둘 다 아님 {neither} — 불일치 쌍 {disc}")
    if disc == 0:
        print("판정 불가 — 불일치 쌍 0")
    else:
        pv = stats.binomtest(only_f, disc, p=0.5, alternative="two-sided").pvalue
        print(f"exact McNemar 양측 p = {pv:.2e} (alpha {ALPHA})" + ("" if disc >= 6 else f" — 판정 불가 (불일치 쌍 {disc} < 6, 검정력 부족)"))

    print("\n[3] 순서 효과 (기술 통계) — 쌍 안에서 1번째 vs 2번째 실행의 placed 수")
    for m in MODELS:
        first = sum(truthy(by[p][m]["placed"]) for p in pairs if by[p][m]["order"] == "1")
        second = sum(truthy(by[p][m]["placed"]) for p in pairs if by[p][m]["order"] == "2")
        print(f"  {m}: 1번째 {first} / 2번째 {second}")

    (OUTPUTS / "plots").mkdir(exist_ok=True)
    x = np.arange(len(STAGES))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, [counts[s]["zeroshot"] for s in STAGES], width=0.4, label="zero-shot")
    plt.bar(x + 0.2, [counts[s]["finetuned"] for s in STAGES], width=0.4, label="fine-tuned")
    plt.xticks(x, STAGES)
    plt.ylabel(f"count (N={n} pairs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "plots/stage_counts.png")
    print("\n저장:", OUTPUTS / "plots/stage_counts.png")


if __name__ == "__main__":
    main()
