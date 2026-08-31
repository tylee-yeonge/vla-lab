# 측정 방법 — openvla-lora-eval

> 이 문서는 "무엇을 어떻게 재었는가" 를 담는 모든 수치의 원본이다. 환경은 [`environment.md`](environment.md), 해석·판단은 [`findings.md`](findings.md).
> 원시 데이터: [`raw/`](raw/) — `eval_zeroshot.jsonl`, `eval_finetuned.jsonl` (각 메타 1줄 + episode 100줄) + 실행 로그 2개

## 1. 측정 설계

### 1.1 변인 통제

변인은 모델 하나다. 두 실행의 차이는 `model` 과 `unnorm_key` 두 항목뿐이며, 하네스가 실행 조건 전체를 결과 파일 첫 줄 `_meta` 에 기록한다. 분석 직전 두 메타를 코드로 대조해 차이 2항목을 재확인했다 (`scripts/analyze_results.py` [1-1]).

| 공통 조건 | 값 |
|---|---|
| 환경 / 지시문 | `PickCube-v1` / "pick up the cube" |
| seed 목록 | `list(range(100))` — week0 목록 (0-19) 포함, 학습 seed (1000-1099) 와 겹침 없음 (하네스 assert) |
| step 예산 | `max_episode_steps` 200, action repeat 4 (정책 결정 50회) |
| 성공 판정 | 환경 소유 `info["success"]` + 단계 판정 (reach_dist 0.05 m, lift_z 0.04 m) |
| 양자화 | nf4 + double quant + fp16 compute |

### 1.2 지표 — 4단계 부분 도달률

최종 성공 (placed) 하나만 보면 0 대 0 에서 아무것도 말할 수 없으므로, episode 를 reached / grasped / lifted / placed 4단계로 판정한다. 1차 지표는 placed, 나머지는 해석 보조 (기술 통계만).

### 1.3 N=100 의 결정 (결과 측정 전 확정)

스텝당 실측 310.8 ms (추론 300.3 + overhead 10.5) 에서 역산 — N=100 이면 95% 반폭 약 10%p 해상도, 두 모델 합계 약 52분. 근거 전문: `Studies/Phase 4.5/week5/outputs/eval_plan.md`.

### 1.4 무행동 하한 seed {8, 58}

zero action 만 넣는 probe 를 seed 0-99 전수에 돌린 결과 seed 8, 58 이 env step 1 에서 성공한다 (초기 배치가 성공 조건을 이미 충족하는 degenerate 배치). 두 seed 는 목록에서 빼지 않고 (두 모델에 똑같이 작용), 구간 추정의 분자·분모에서만 제외한다 (98쌍 기준). 근거: `eval_plan.md` §4.

### 1.5 통계 방법 (결과를 보기 전에 문서로 고정)

`Studies/Phase 4.5/week5/outputs/stat_method.md` 에서 확정한 방법을 그대로 적용했다.

| 항목 | 방법 |
|---|---|
| 구간 | Wilson score 95% (0/N 에서도 상한이 0 이 아님) |
| 짝지은 비교 | 같은 seed 끼리 2x2, 불일치 쌍의 exact 이항검정 (McNemar), 양측, alpha 0.05 |
| 판정 문구 | 불일치 쌍 0 이면 "판정 불가 (불일치 쌍 0)", 6 미만이면 "판정 불가 (검정력 부족)" |
| 다중 비교 회피 | 구간·검정은 1차 지표 placed 에만. 단계별 지표는 개수만 보고 |

## 2. 실행

```bash
# week5/scripts 에서 -- 두 실행은 인자 2개만 다르다
python eval_harness.py --model openvla/openvla-7b --unnorm-key bridge_orig \
  --out ../outputs/eval_zeroshot.jsonl 2>&1 | tee ../outputs/eval_zeroshot.log
python eval_harness.py --model /workspace/models/openvla-maniskill-ft --unnorm-key maniskill_pickcube \
  --out ../outputs/eval_finetuned.jsonl 2>&1 | tee ../outputs/eval_finetuned.log
```

집계는 `scripts/analyze_results.py` (cwd: `Studies/Phase 4.5/week6/`, GPU 불요).

## 3. 집계 결과 (원본 수치)

### 3.1 단계별 성공 수

98쌍 기준 (무행동 seed {8, 58} 제외), 괄호는 100쌍 원값.

| 단계 | zero-shot | fine-tuned |
|---|---|---|
| reached | 0/98 (0/100) | 92/98 (92/100) |
| grasped | 0/98 (0/100) | 75/98 (75/100) |
| lifted | 0/98 (0/100) | 1/98 (1/100) |
| placed | 0/98 (2/100) | 0/98 (2/100) |

- placed 95% Wilson 구간: zero-shot [0%, 3.77%], fine-tuned [0%, 3.77%] (둘 다 0/98)
- placed 의 100쌍 원값 2건은 양쪽 모두 무행동 seed {8, 58} 그 자체 — 조작 성공 0

### 3.2 짝지은 2x2 (100쌍 전체)

| 단계 | 둘 다 | zero 만 (악화) | ft 만 (개선) | 둘 다 아님 | 불일치 쌍 |
|---|---|---|---|---|---|
| reached | 0 | 0 | 92 | 8 | 92 |
| grasped | 0 | 0 | 75 | 25 | 75 |
| lifted | 0 | 0 | 1 | 99 | 1 |
| placed | 2 | 0 | 0 | 98 | 0 |

placed 불일치 쌍 0 — 검정을 하지 않고 "판정 불가 (불일치 쌍 0)" 으로 보고 (stat_method.md §2.3).

### 3.3 기타 관측

- 종료 사유: 양쪽 모두 `env_end` 98건 + `success` 2건 (무행동 seed)
- 조작 성공 episode 0건이므로 성공 스텝 수 (효율 지표) 비교는 성립하지 않음
- fine-tuned 의 grasped 75건 중 lifted 도달 1건 (seed 72, 이후 placed 실패)

그림: [`plots/stage_counts.png`](plots/stage_counts.png) — 단계별 성공 수 막대 (N=98).

## 4. 재생성 가능 여부

| 항목 | 구분 |
|---|---|
| eval 결과 (jsonl) | 재생성 가능 — 하네스 + seed 규칙 + 모델. 단 fine-tuned 가중치가 전제 |
| fine-tuned 가중치 | 재생성 가능 — base 캐시 + LoRA 어댑터 (week3 회수물) + 재머지 절차 (`week4/outputs/remerge_check.md`) |
| N·통계 방법의 판단 근거 | 재생성 불가 — `eval_plan.md`, `stat_method.md` 기록이 원본 |
