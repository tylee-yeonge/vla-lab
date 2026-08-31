# 측정 환경 — openvla-lora-eval

> 측정일: 2026-08-30 (eval 실행) / 2026-08-31 (집계·분석)
> 이 문서는 "어떤 장비·어떤 버전에서 재었는가" 만 담는다. 방법은 [`methodology.md`](methodology.md), 해석은 [`findings.md`](findings.md).

## 1. eval 실행 환경 (성공률 측정)

| 항목 | 값 | 근거 |
|---|---|---|
| GPU | RTX 4070 12GB | `Studies/Phase 4.5/week5/outputs/eval_plan.md` 헤더 |
| venv | `Studies/Phase 4/.venv-vla` (python 3.12) | `raw/eval_*.log` 의 경로 |
| transformers | 4.40.1 | `Studies/Phase 4.5/week4/outputs/local_versions.txt` |
| torch | 2.12.0+cu130 | 같은 파일 |
| bitsandbytes | 0.49.2 | 같은 파일 |
| attention 구현 | eager (flash-attn 미설치) | `Studies/Phase 4.5/week4/outputs/compat_check.md` §1 |
| 양자화 | nf4 + double quant + fp16 compute (두 모델 동일) | `raw/eval_*.jsonl` 첫 줄 `_meta.quant` |
| sim | ManiSkill `PickCube-v1`, SAPIEN 래스터 렌더 224x224 | `raw/eval_*.jsonl` `_meta`, week0 `env_build.md` |
| 하네스 코드 버전 | commit f2cabad | `raw/eval_*.jsonl` `_meta.commit` |

## 2. 대상 모델

| 모델 | 소재 | unnorm_key |
|---|---|---|
| zero-shot | `openvla/openvla-7b` (HF 캐시 리비전 47a0ec7fc4ec123775a391911046cf33cf9ed83f) | `bridge_orig` |
| fine-tuned | `/workspace/models/openvla-maniskill-ft` — week4 재머지 산출물 (LoRA rank 32, 2,000스텝, 색인 대조 통과) | `maniskill_pickcube` |

fine-tuned 가중치의 이력: 학습은 `Measurements/openvla-lora-runpod/`, 재머지·호환성 검증은 `Studies/Phase 4.5/week4/outputs/remerge_check.md`, `compat_check.md`.

## 3. 집계·분석 환경 (GPU 불요)

| 항목 | 값 |
|---|---|
| venv | `Studies/Phase 4.5/.venv-sim` |
| python / numpy / scipy / matplotlib | 3.12.3 / 2.5.1 / 1.18.0 / 3.11.1 |
| 분석 스크립트 | `scripts/analyze_results.py` (실행 cwd: `Studies/Phase 4.5/week6/`) |
