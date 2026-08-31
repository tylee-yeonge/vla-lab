import argparse
import json
import os
import subprocess
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from PIL import Image
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig

ENV_ID = "PickCube-v1"
MAX_EPISODE_STEPS = 200
ACTION_REPEAT = 4
POLICY_STEPS = MAX_EPISODE_STEPS // ACTION_REPEAT
INSTRUCTION = "pick up the cube"
CAMERA_KEY = ("sensor_data", "base_camera", "rgb")
IMAGE_SIZE = 224
N_EPISODES = 100
POS_LIMIT = 0.1
ROT_SCALE = -0.1
REACH_DIST = 0.05
LIFT_Z = 0.04


def to_maniskill_action(raw_action):
    """OpenVLA 물리량 출력(7,)을 ManiSkill pd_ee_delta_pose 정규화 action(7,)으로 바꾼다.

    규칙의 근거는 week0 action_contract.md 계약 표. 두 모델 모두 물리량(m, rad,
    gripper 0-1)을 내므로 같은 정변환을 쓴다 (week1 역변환과 짝).

    Args:
        raw_action: vla.predict_action() 출력 (7,)

    Returns:
        ManiSkill action (7,) float32. 전 차원 [-1, 1] 정규화값
    """
    pos = raw_action[:3] / POS_LIMIT
    rot = raw_action[3:6] / ROT_SCALE
    rot_norm = np.linalg.norm(rot)
    if rot_norm > 1.0:
        rot = rot / rot_norm
    grip = 2.0 * raw_action[6] - 1.0
    action = np.concatenate([np.clip(pos, -1, 1), rot, [np.clip(grip, -1, 1)]])
    return action.astype(np.float32)


def to_vec(pose_field):
    """(1, 3) 형태 GPU 텐서 좌표를 (3,) numpy 벡터로 바꾼다.

    Args:
        pose_field: pose.p 같은 배치 텐서

    Returns:
        (3,) float numpy 배열
    """
    return np.asarray(pose_field.cpu())[0]


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--unnorm-key", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    args.model,
    attn_implementation="eager",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    quantization_config=bnb_config,
)

stats_path = os.path.join(args.model, "dataset_statistics.json")
if os.path.exists(stats_path):
    with open(stats_path) as f:
        vla.norm_stats.update(json.load(f))

prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"

with open("../../week0/outputs/zeroshot_baseline.json") as f:
    base_seeds = json.load(f)["seeds"]
eval_seeds = list(range(N_EPISODES))
assert set(base_seeds) <= set(eval_seeds), "week0 목록이 새 목록에 포함되어야 한다"
with open("../../week1/outputs/dataset/collect_meta.json") as f:
    train_seeds = set(json.load(f)["train_seeds"])
assert not (set(eval_seeds) & train_seeds), "eval seed 에 학습 seed 가 섞였다"

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], 
                        capture_output=True, text=True).stdout.strip()
meta = {"model": args.model, "unnorm_key": args.unnorm_key, "env_id": ENV_ID,
        "max_episode_steps": MAX_EPISODE_STEPS, "action_repeat": ACTION_REPEAT,
        "instruction": INSTRUCTION, "n_episodes": N_EPISODES,
        "quant": "nf4+dq+fp16", "reach_dist": REACH_DIST, "lift_z": LIFT_Z,
        "commit": commit}

env = gym.make(
    ENV_ID,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",
    render_mode="rgb_array",
    sensor_configs=dict(width=224, height=224),
    max_episode_steps=MAX_EPISODE_STEPS,
)
base = env.unwrapped

with open(args.out, "w") as out_file:
    out_file.write(json.dumps({"_meta": meta}) + "\n")
    for seed in eval_seeds:
        obs, info = env.reset(seed=seed)
        stages = {"reached": False, "grasped": False, "lifted": False, "placed": False}
        reason = "step_cap"
        done = False
        for policy_step in range(POLICY_STEPS):
            frame = obs[CAMERA_KEY[0]][CAMERA_KEY[1]][CAMERA_KEY[2]].cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            image = Image.fromarray(frame.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE))
            model_inputs = processor(prompt, image).to("cuda:0", dtype=torch.float16)
            with torch.no_grad():
                raw_action = vla.predict_action(
                    input_ids=model_inputs["input_ids"],
                    pixel_values=model_inputs["pixel_values"],
                    unnorm_key=args.unnorm_key,
                    do_sample=False,
                )
            action = to_maniskill_action(raw_action)
            for _ in range(ACTION_REPEAT):
                obs, reward, terminated, truncated, info = env.step(action)
                tcp = to_vec(base.agent.tcp.pose.p)
                cube = to_vec(base.cube.pose.p)
                stages["reached"] |= bool(np.linalg.norm(tcp - cube) < REACH_DIST)
                stages["grasped"] |= bool(info["is_grasped"].item())
                stages["lifted"] |= bool(cube[2] > LIFT_Z)
                stages["placed"] |= bool(info["success"].item())

                if stages["placed"]:
                    reason = "success"
                    done = True
                    break
                if terminated or truncated:
                    reason = "env_end"
                    done = True
                    break
            if done:
                break

        record = {"seed": seed, "steps": policy_step + 1, "reason": reason, **stages}
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()
        print(f"seed{seed:03d}: {record}")

env.close()
print(f"\n완료: {args.out}")