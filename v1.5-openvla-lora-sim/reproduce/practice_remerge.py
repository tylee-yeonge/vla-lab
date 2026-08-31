"""
실습 1: LoRA 어댑터를 base 에 재머지해 pod 의 머지 가중치를 로컬에서 재구성
전용 venv(/workspace/venvs/remerge -- 학습과 같은 torch 2.2.0 / peft 0.11.1)에서 실행한다.
학습 스크립트가 체크포인트 저장 때 하던 것과 같은 절차다: bf16 base 에 어댑터를 합쳐 저장.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq


# 입력: week3 이 회수한 어댑터. 출력: 레포 밖 /workspace/models (15GB 를 커밋 대상에서 격리)
ADAPTER_PATH = (
    "/workspace/study/physical-ai-study/Studies/Phase 4.5/week3/outputs/recovered/"
    "adapter-tmp/"
    "openvla-7b+maniskill_pickcube_only+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
)
OUT_PATH = "/workspace/models/openvla-maniskill-ft"


# -- base 적재: 학습 때와 같은 bf16 (HF_HUB_OFFLINE=1 이라 로컬 캐시만 읽는다) --
base = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",                          # 캐시의 스냅샷 (1-1 에서 확인한 리비전)
    torch_dtype=torch.bfloat16,                    # 학습 저장과 같은 dtype
    low_cpu_mem_usage=True,                        # 15GB 를 RAM 에 한 번만 올린다
    trust_remote_code=True,                        # openvla 자체 모델 코드 허용
)

# -- 머지: 어댑터를 얹고 수정분을 가중치에 더해 일반 모델로 되돌린다 --
merged = PeftModel.from_pretrained(base, ADAPTER_PATH).merge_and_unload()

# -- 저장: 조각난 safetensors + config + 색인이 생긴다 --
merged.save_pretrained(OUT_PATH)
print("저장 완료:", OUT_PATH)
