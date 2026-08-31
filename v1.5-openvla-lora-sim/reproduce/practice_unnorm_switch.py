import json
import numpy as np
import torch
from PIL import Image

STATS_PATH = "/workspace/models/openvla-maniskill-ft/dataset_statistics.json"
DATASET_KEY = "maniskill_pickcube"
INSTRUCTION = "pick up the cube"

print(f"현재 norm_stats 키: ", list(vla.norm_stats.keys()))

with open(STATS_PATH) as f:
    stats = json.load(f)
print("통계 파일 키: ", list(stats.keys()))
vla.norm_stats[DATASET_KEY] = stats[DATASET_KEY]
print("주입 후 키: ", list(vla.norm_stats.keys()))

image = Image.fromarray(
    (np.random.RandomState(0).rand(224, 224, 3) * 255).astype(np.uint8)
)
prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.float16)

for key in ["bridge_orig", DATASET_KEY]:
    with torch.no_grad():
        action = vla.predict_action(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            unnorm_key=key,
            do_sample=False,
        )
    print(f"unnorm_key={key}")
    print("action: ", np.round(action, 4))
    print("위치 3차원 크기: ", np.round(np.abs(action[:3]), 4))

action_stats = stats[DATASET_KEY]["action"]
for name, value in action_stats.items():
    value = np.asarray(value)    
    if value.dtype == bool:
        print(f"{name}: {value}")
    else:
        print(f"{name}: {np.round(value, 4)}")