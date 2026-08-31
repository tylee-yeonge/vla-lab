import torch
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig

MODEL_PATH = "/workspace/models/openvla-maniskill-ft"
BASELINE_GB = 4.38

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

torch.cuda.reset_peak_memory_stats()
before_gb = torch.cuda.memory_allocated()
print(f"적재 전: {before_gb:.2f} GB")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    attn_implementation="eager",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    quantization_config=bnb_config,
)
after_gb = torch.cuda.memory_allocated() / 1e9
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"적재 후: {after_gb:.2f} GB (피크 {peak_gb:.2f} GB)")

diff = after_gb - BASELINE_GB
print(f"baseline({BASELINE_GB} GB) 대비 차이: {diff:+.2f} GB")
if abs(diff) < 0.5:
    print("판정: 근사 일치 -- 양자화 적용됨")
else:
    print("판정: 벗어남 -- 양자화 설정 또는 적재 범위 확인 필요")