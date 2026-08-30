# v1.5 — OpenVLA LoRA adaptation (sim, before/after 정량 분석)

> 상태: **작성 중** (2026.09 초 발행 예정). 아래는 문서 골격 — 수치·분석이 채워진 뒤 공개 본문이 된다.

## 문서 골격

1. **배경** — 왜 adaptation 인가 (3층 좌표: 배포·통합 위의 둘째 층 방어선)
2. **설계** — sim task 선정 (ManiSkill, 미학습 분포 vs 환경 정합의 상충), 데이터 생성·포맷팅 (RLDS), 컴퓨트 분업 (RunPod 24GB 학습 / 로컬 12GB 4-bit eval)
3. **측정 장치 검증** — scripted-solution 상한 20/20 · 하한 1/20 (성공률 0% 를 "정책의 성적"으로 읽을 수 있게 만드는 절차)
4. **실행** — LoRA 1사이클 실측: 94분 / $1.18 / VRAM 피크 18.5GB (batch 1 + grad accum 16 으로 27GB 하한을 24GB 에 넣은 방법), 비용의 23% 가 체크포인트 저장이라는 발견
5. **결과** — zero-shot vs fine-tuned, 동일 조건 N회: 성공률 + 부분 도달률 (reached/grasped/lifted/placed) + 분산 *(작성 예정)*
6. **분석** — 차이가 노이즈면 그 원인 (데이터 규모/분포/학습 설정) *(작성 예정)*
7. **필드 변화** — 선정 시점 (2026.06) 과 마감 시점 사이의 세대 교체 (π0.7 / Gemini Robotics 2 / GR00T N1.7), AR 토큰 300ms 한계와 action-chunk 계열, 다음 단계 = 같은 eval 논리를 실기 (SO-101 + SmolVLA) 로

## 동반 자료 (발행 시 추가)

- 재현 스크립트 (데이터 생성 → 포맷팅 → 학습 → 머지/양자화 → eval)
- 측정 조건 (environment / methodology / findings 3분할)
