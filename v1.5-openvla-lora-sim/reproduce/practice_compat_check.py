import accelerate
import bitsandbytes
import numpy as np
import timm
import tokenizers
import torch
import transformers

print("추론 환경")

for module in [torch, transformers, tokenizers, timm, accelerate, bitsandbytes]:
    print(f"{module.__name__}: {module.__version__}")

outputs = []
for trial in range(3):
    with torch.no_grad():
        action = vla.predict_action(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            unnorm_key=DATASET_KEY,
            do_sample=False,
        )
    outputs.append(np.asarray(action))
    print(f"trial{trial}: {np.round(action, 5)}")

max_diff = max(np.abs(outputs[0] - other).max() for other in outputs[1:])
print(f"최대 편차: {max_diff:.2e}")
print("판정: ", "결정적" if max_diff < 1e-6 else "비결정 -- do_sample / dropout 설정 확인")