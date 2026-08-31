# 결과 해석·판단 근거 — openvla-lora-eval

> 측정일: 2026-08-30 (eval) / 집계·분석: 2026-08-31
> 수치의 원본: [`methodology.md`](methodology.md) — 이 문서의 숫자는 모두 그 문서 §3 에서 가져온다.

## 0. 세 문서의 분업

| 문서 | 답하는 질문 |
|---|---|
| [`environment.md`](environment.md) | 어떤 장비·어떤 버전에서 재었는가 |
| [`methodology.md`](methodology.md) | 무엇을 어떻게 재었는가 — 모든 수치의 원본 |
| 이 문서 | 그 수치가 무엇을 뜻하고, 그 위에서 무엇을 판단했는가 |

## 1. 결과 요약

```
zero-shot   placed 0/98 (95% Wilson 구간 0-3.8%) [전체 2/100]
fine-tuned  placed 0/98 (95% Wilson 구간 0-3.8%) [전체 2/100]
불일치 쌍   개선 c=0건 / 악화 b=0건 -> 판정 불가 (불일치 쌍 0)
단계별      reached 0/98 -> 92/98 / grasped 0/98 -> 75/98 / lifted 0/98 -> 1/98
```

**결론 (한 문장)**: 최종 성공 지표 placed 는 두 모델 모두 조작 성공 0/98 (95% Wilson 상한 3.8%), 불일치 쌍 0 으로 판정 불가이며, 부분 도달 지표는 fine-tuned 쪽에서 reached 0/98 -> 92/98, grasped 0/98 -> 75/98 로 이동했으나 lifted 1/98 이후로 이어지지 않았다.

## 2. 이 수치로 말할 수 있는 것과 없는 것

| 말할 수 있는 것 | 근거 |
|---|---|
| adaptation 이 접근·파지 행동을 만들었다 (reached 92/98, grasped 75/98 — zero-shot 은 둘 다 0/98) | 짝지은 개수, 기술 통계 |
| 최종 성공률의 차이는 이 N 으로 판정 불가 | 불일치 쌍 0 — "차이 없음" 과 다른 진술 |
| fine-tuned 의 실패는 파지 이후 (lift) 에 몰려 있다 | grasped 75건 중 lifted 1건 |

| 말할 수 없는 것 | 이유 |
|---|---|
| "성능이 개선됐다" (지표 이름 없는 요약) | 부분 도달률과 성공률은 다른 지표다 |
| real 로봇·다른 task 로의 일반화 | sim, PickCube 단일 task, 단일 embodiment |
| "LoRA 는 ~하다" 류 일반화 | 학습 1회, seed 반복 없음 |
| adaptation 이 원리적으로 무효라는 주장 | 부분 도달률이 크게 움직였다 |

## 3. 원인 분석 — 배제와 잔여

"성공률이 안 올랐다" 의 원인 후보 중, week0-5 의 검증 기록이 실제로 배제한 것과 배제하지 못한 것을 구분한다. 전문: `Studies/Phase 4.5/week6/outputs/causal_analysis.md`.

### 3.1 배제된 후보

| 배제된 후보 | 근거 (파일 + 판정) |
|---|---|
| 측정 장치 고장 (환경·루프·판정·step 예산) | week0 `harness_check.md` — scripted 상한 20/20, 최대 49/200 step |
| 라벨 변환의 수치 손실·규약 불일치 | week1 `roundtrip_check.md` — 왕복 항등 1.49e-8, 계약 표 실측 6/6. week2 `norm_check.md` §4 — gripper 규약 일치 |
| 회전 표현 불일치 (axis-angle vs euler) | week2 `format_spec.md` — 회전 표현 게이트 통과 (라벨 구성이 EEF_POS 와 일치) |
| 데이터가 학습에 안 들어감 | week2 `norm_check.md` §5.1 — 통계 100/3,760 이 라벨 원본과 일치. week3 `train_log.md` — 학습 정확도 0.12 -> 0.87 |
| 라벨 이중 정규화 | week2 `norm_check.md` §5.2 — 통계 실측이 물리 단위 (위치 ±0.015 m) |
| 학습이 안 돌았음 | week3 `train_log.md` — 2,000스텝 완주, loss 10.75 -> 0.40 |
| 체크포인트 손상·재머지 오류 | week4 `remerge_check.md` — 색인 일치. `compat_check.md` — 가중치 경고 0건, 결정성 bit 동일 |
| `unnorm_key` 오연결 | week4 `compat_check.md` §3 — 출력 대역이 자체 통계 대역 (±0.012) 과 일치, bridge_orig 대역 (±0.029-0.042) 과 구분 |
| 양자화 조건 차이 | 두 eval 메타의 quant 동일 (nf4+dq+fp16) |
| 측정 조건 누출 | 메타 차이 2항목 (model, unnorm_key), seed 목록 일치 (analyze_results.py [1-1]) |

### 3.2 남은 후보

| 남은 후보 | 왜 남는가 | 가르려면 |
|---|---|---|
| 데이터 규모 (100 ep / 3,760 transition) | 최소선 이상 시도 안 함 | 규모 늘린 2차 학습 |
| 상태 분포 협소 (expert-only) | 이탈 상태의 정답이 데이터에 없음 | DAgger 류, 파지 이후 로그 분석 |
| 도메인 갭 | 이번 Phase 의 전제 | 도메인 정합 환경 또는 real 데이터 |
| 학습 스텝·하이퍼파라미터 | 예산 역산 값, 탐색 없음 | 탐색 실험 |
| task 난이도 | PickCube 단일 | 다른 task |
| 학습의 무작위성 | 학습 1회 | 반복 학습 |
| 라벨 재생 무결성 (강한 기준) | week1 `roundtrip_check.md` 의 sim 재생 검증이 기록상 미측정 | 재생 스크립트 재실행 |

grasp 이후 실패 집중 (75건 파지, lift 1건) 은 상태 분포 협소나 파지 이후 라벨 품질 후보와 양립하지만, 이 데이터만으로 우선순위를 정할 수 없다 — 해석은 가설이지 판정이 아니다.

## 4. 이 측정이 확정하지 못한 것

- 남은 후보들 사이의 우선순위 (성공 관측 0 이라 판별력 없음)
- 배제 목록의 유효 범위는 이 조건 (PickCube-v1, N=100, nf4, 학습 1회) 한정 — real 이관 시 리셋되며, §3 의 표가 그때의 재검증 목록이 된다 (Phase 7)
- fine-tuned 정책의 정성적 실패 양상 (파지 후 정지인지, 파지 해제인지, 이동 방향 오류인지) — episode 영상·상태 로그를 남기지 않았다

## 5. 관련 코드·문서 소재

| 자산 | 위치 |
|---|---|
| eval harness | `scripts/eval_harness.py` (원본: `Studies/Phase 4.5/week5/scripts/`) |
| 분석 스크립트 | `scripts/analyze_results.py` (실행 cwd: `Studies/Phase 4.5/week6/`) |
| action 변환 (정/역) | `Studies/Phase 4.5/week1/action_transform.py` |
| RLDS 빌더·등록 패치 | `Studies/Phase 4.5/week2/` + `week2/outputs/openvla_registration.patch`, `openvla_base_commit.txt` |
| 학습 Dockerfile | `Studies/Phase 4.5/week3/` (`openvla-train:v2`, `image_build.md`) |
| 방법 확정 문서 | `Studies/Phase 4.5/week5/outputs/eval_plan.md`, `stat_method.md` |
| 공개 문서 초고 | `Studies/Phase 4.5/week6/outputs/blog_draft.md` (vla-lab 발행 대기) |
