"""
실습 4-1: OpenVLA 의 action 통계를 열어 단위·정규화 규약을 확인
"""
import torch
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(               # Phase 4 week6 과 동일한 4-bit 설정
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
vla = AutoModelForVision2Seq.from_pretrained(  # 로드에 수 분 걸린다
    "openvla/openvla-7b",
    attn_implementation="eager",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    quantization_config=bnb_config,
)


# -- 사용 가능한 unnorm_key 목록 --
print("\n[4-1a] 선택 가능한 unnorm_key")
print(list(vla.norm_stats.keys()))             # 학습 데이터셋별 통계 키 목록


# -- 특정 key 의 통계 내용 (여기서 단위·정규화 대상 차원을 읽는다) --
KEY = "bridge_orig"                            # <- 후보 key. 선택 근거를 표에 적을 것
print(f"\n[4-1b] {KEY} 통계")
for name, value in vla.norm_stats[KEY]["action"].items():
    print(f"   {name}: {value}")               # 분위수·평균·마스크 등이 그대로 나온다