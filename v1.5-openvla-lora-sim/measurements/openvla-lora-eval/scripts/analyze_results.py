"""
실습 1-3: week5 원시 결과를 집계하고 구간·짝지은 비교를 계산한다.

방법의 정본은 week5 outputs/stat_method.md 다. 여기서 방법을 새로 고르지 않는다.
  - 구간: Wilson score (stat_method.md 1.3)
  - 1차 지표: placed, 무행동 하한 seed {8, 58} 제외한 98쌍 기준 + 100쌍 원값 병기 (stat_method.md 3)
  - 단계별 (reached/grasped/lifted): 기술 통계만 -- 구간·검정을 붙이지 않는다 (stat_method.md 3)
  - 짝지은 비교: placed 의 불일치 쌍 exact 이항검정 (McNemar), 양측, alpha=0.05 (stat_method.md 2)
실행 위치: week6/ 를 cwd 로 (PRACTICE.md 0.4) -- python scripts/analyze_results.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

STAGES = ["reached", "grasped", "lifted", "placed"]
ALPHA = 0.05                       # 유의 수준 (stat_method.md 2.2)
NOOP_SEEDS = {8, 58}               # 무행동 하한 seed -- 초기 배치가 이미 성공 조건 충족 (eval_plan.md)

def load(path):
    """jsonl 을 읽어 메타와 seed 별 레코드로 나눈다."""
    with open(path) as f:
        lines = [json.loads(line) for line in f]
    return lines[0]["_meta"], lines[1:]

zero_meta, zero_records = load("../week5/outputs/eval_zeroshot.jsonl")
ft_meta, ft_records = load("../week5/outputs/eval_finetuned.jsonl")

# -- 1-1. 조건 재확인 (week5 무결성 검사를 분석 직전에 한 번 더) --
print("=" * 60)
print("[1-1] 메타 차이")
for key in set(zero_meta) | set(ft_meta):
    if zero_meta.get(key) != ft_meta.get(key):
        print(f"   {key}: {zero_meta.get(key)} -> {ft_meta.get(key)}")
assert [r["seed"] for r in zero_records] == [r["seed"] for r in ft_records], "seed 목록 불일치"
n = len(zero_records)
print(f"   N = {n}")

# 무행동 하한 seed 를 뺀 98쌍 -- 구간 추정의 분자·분모에만 적용 (짝지은 비교에는 영향 없음)
zero98 = [r for r in zero_records if r["seed"] not in NOOP_SEEDS]
ft98 = [r for r in ft_records if r["seed"] not in NOOP_SEEDS]
n98 = len(zero98)

# -- 1-2. 단계별 집계: 98쌍 기준 성공 수 + 100쌍 원값 병기, 구간은 1차 지표(placed)에만 --
print(f"\n[1-2] 단계별 성공 수 (98쌍 기준, 괄호는 100쌍 원값)")
summary = {}
for stage in STAGES:
    zero_hits = sum(r[stage] for r in zero98)
    ft_hits = sum(r[stage] for r in ft98)
    zero_raw = sum(r[stage] for r in zero_records)
    ft_raw = sum(r[stage] for r in ft_records)
    summary[stage] = {"zero": zero_hits, "ft": ft_hits}
    line = (f"   {stage:8s} zero {zero_hits:3d}/{n98} (raw {zero_raw}/{n})"
            f"  |  ft {ft_hits:3d}/{n98} (raw {ft_raw}/{n})")
    print(line)
    if stage == "placed":
        # Wilson 구간 -- 0/N 에서도 상한이 0 이 아닌 방법 (stat_method.md 1)
        zero_ci = stats.binomtest(zero_hits, n98).proportion_ci(
            confidence_level=1 - ALPHA, method="wilson")
        ft_ci = stats.binomtest(ft_hits, n98).proportion_ci(
            confidence_level=1 - ALPHA, method="wilson")
        print(f"            zero Wilson [{zero_ci.low:.4f}, {zero_ci.high:.4f}]"
              f"  |  ft Wilson [{ft_ci.low:.4f}, {ft_ci.high:.4f}]")

# -- 2-1. 짝지은 2x2 표: 칸 수는 전 단계 기술 통계, 검정은 1차 지표(placed)에만 --
print("\n[2-1] 짝지은 비교 (100쌍 전체 -- 무행동 seed 는 두 모델에 똑같이 작용)")
for stage in STAGES:
    both = only_zero = only_ft = neither = 0
    for zero_record, ft_record in zip(zero_records, ft_records):   # 같은 seed 끼리
        z, f = zero_record[stage], ft_record[stage]
        both += z and f
        only_zero += z and not f                 # 악화 방향
        only_ft += f and not z                   # 개선 방향
        neither += not z and not f
    discordant = only_zero + only_ft             # 판정에 실제로 쓰이는 관측 수
    print(f"\n   {stage}")
    print(f"      둘 다 {both} / zero만 {only_zero} / ft만 {only_ft} / 둘 다 아님 {neither}")
    print(f"      불일치 쌍 {discordant}개 (개선 {only_ft} / 악화 {only_zero})")
    if stage != "placed":
        continue                                 # 검정은 1차 지표에만 (다중 비교 회피)
    if discordant == 0:
        print("      판정 불가 -- 불일치 쌍 0")
    else:
        # 불일치 쌍만 대상으로 한 exact 이항검정, 양측 (stat_method.md 2.1-2.2)
        test = stats.binomtest(only_ft, discordant, p=0.5, alternative="two-sided")
        print(f"      exact McNemar 양측 p = {test.pvalue:.4f} (유의수준 {ALPHA})")
        if discordant < 6:
            # 5:0 몰빵도 p=0.0625 -- 이 검정으로 유의 판정이 구조적으로 불가 (stat_method.md 2.3)
            print(f"      판정 불가 -- 불일치 쌍 {discordant}건, 검정력 부족")

# -- 2-2. 효율 비교: 조작 성공 episode 의 소요 스텝 (무행동 seed 제외) --
print("\n[2-2] 조작 성공 episode 의 스텝 수 (seed 8, 58 제외)")
for records, name in [(zero98, "zero"), (ft98, "ft")]:
    steps = [r["steps"] for r in records if r["placed"]]
    if steps:
        print(f"   {name}: n={len(steps)} median={np.median(steps):.0f} "
              f"min={min(steps)} max={max(steps)}")
    else:
        print(f"   {name}: 조작 성공 episode 없음")

# -- 3-1. 단계별 성공 수 막대 (98쌍 기준 -- 보고 표와 같은 분모) --
x = np.arange(len(STAGES))
zero_values = [summary[s]["zero"] for s in STAGES]
ft_values = [summary[s]["ft"] for s in STAGES]
plt.figure(figsize=(7, 4))
plt.bar(x - 0.2, zero_values, width=0.4, label="zero-shot")
plt.bar(x + 0.2, ft_values, width=0.4, label="fine-tuned")
plt.xticks(x, STAGES)
plt.ylabel(f"count (N={n98}, noop seeds excluded)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/stage_counts.png")
print("\n저장: outputs/plots/stage_counts.png")
