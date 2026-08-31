# 재현 절차 — v1.5 OpenVLA LoRA adaptation (sim)

> 발행 시점 (2026-08-31) 사본. 살아있는 원본은 비공개 프로젝트 리포에 있으며, 이 디렉토리는 스냅샷이다.
> 버전 고정: [`requirements.txt`](requirements.txt) + 각 measurements/*/environment.md

## 실행 순서

| # | 단계 | 스크립트 | 검증 기준 |
|---|---|---|---|
| 0 | 환경 확인 | [`../measurements/openvla-maniskill-zeroshot/scripts/practice_env_check.py`](../measurements/openvla-maniskill-zeroshot/scripts/practice_env_check.py) | 렌더·seed 재현 확인 |
| 1 | 하네스 검증 (상한 대조) | [`harness_check.py`](../measurements/openvla-maniskill-zeroshot/scripts/harness_check.py) | scripted 해법 20/20, 최대 49/200 step |
| 2 | zero-shot baseline | [`practice_zeroshot_baseline.py`](../measurements/openvla-maniskill-zeroshot/scripts/practice_zeroshot_baseline.py) + [`openvla_action.py`](../measurements/openvla-maniskill-zeroshot/scripts/openvla_action.py) | 부분 도달률 병기 |
| 3 | LoRA 학습 (RunPod 24GB) | [`../measurements/openvla-lora-runpod/scripts/`](../measurements/openvla-lora-runpod/scripts/) (Dockerfile 2종 + patch + start.sh) | 2,000 스텝 완주, loss·token acc 곡선 |
| 4 | 재머지·4bit 적재·검증 (로컬 12GB) | 본 디렉토리 `practice_remerge.py` → `practice_load_4bit.py` → `practice_compat_check.py` → `practice_smoke_test.py` → `practice_unnorm_switch.py` | 색인 일치 / VRAM 4.38GB·경고 0 / bit 결정성 / unnorm 대역 구분 |
| 5 | before/after eval (N=100) | [`eval_harness.py`](../measurements/openvla-lora-eval/scripts/eval_harness.py) → [`analyze_results.py`](../measurements/openvla-lora-eval/scripts/analyze_results.py) | 메타 차이 2항목 확인 후 집계 (Wilson + exact McNemar, 사전 등록) |

## 재생성으로 대체하는 것

- **데이터셋 (100 ep / 3,760 transitions)**: expert 스크립트 + seed 1000-1099 로 재수집 (라벨 규약: measurements/openvla-lora-eval/methodology.md)
- **LoRA 가중치**: 위 3-4 단계 재실행 (94분 / 약 $1.18, RTX 4090 기준)
