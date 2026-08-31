import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from PIL import Image

ENV_ID = "PickCube-v1"
MAX_EPISODE_STEPS = 200
ACTION_REPEAT = 4
POLICY_STEPS = MAX_EPISODE_STEPS // ACTION_REPEAT
SMOKE_SEED = 500
INSTRUCTION = "pick up the cube"

MODEL_PATH = "/workspace/models/openvla-maniskill-ft"
UNNORM_KEY = "maniskill_pickcube"

env = gym.make(
    ENV_ID,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",
    render_mode="rgb_array",
    sensor_configs=dict(width=224, height=224),
    max_episode_steps=MAX_EPISODE_STEPS,
)
obs, info = env.reset(seed=SMOKE_SEED)
prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"

done = False
for policy_step in range(POLICY_STEPS):
    frame = obs["sensor_data"]["base_camera"]["rgb"].cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    image = Image.fromarray(frame.astype(np.uint8)).resize((224, 224))
    model_inputs = processor(prompt, image).to("cuda:0", dtype=torch.float16)
    with torch.no_grad():
        raw_action = vla.predict_action(
            input_ids=model_inputs["input_ids"],
            pixel_values=model_inputs["pixel_values"],
            unnorm_key=UNNORM_KEY,
            do_sample=False,
        )
    action = raw_action
    for _ in range(ACTION_REPEAT):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated or bool(info["success"].item()):
            done = True
            break
    if done:
        break

env.close()
print(f"루프 완주: 정책 결정 {policy_step + 1}회. 예외 없음")
print("성공/실패는 판정하지 않는다 -- week5의 N회 측정에서 다룬다")