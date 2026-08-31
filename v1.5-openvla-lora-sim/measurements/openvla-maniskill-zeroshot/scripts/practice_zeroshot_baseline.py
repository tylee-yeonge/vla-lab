"""
실습 6: OpenVLA zero-shot baseline (N=20, 고정 seed, 부분 도달률 병기)
"""
import json                                    # 결과를 원시 형태로 저장하기 위해
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


ENV_ID = "PickCube-v1"                         # <- 실습 2 확정값
MAX_EPISODE_STEPS = 200                        # <- 실습 4-3: env step 예산 (실습 5 와 동일)
ACTION_REPEAT = 4                              # <- 실습 4-3: 5 Hz 정책을 20 Hz sim 에 맞춘다
POLICY_STEPS = MAX_EPISODE_STEPS // ACTION_REPEAT   # 정책 결정 횟수 = 50
SEEDS = list(range(20))                        # <- 실습 5 와 같은 목록 (변인 고정)
NOOP_SEEDS = [8]                               # <- 실습 5-3 무행동 하한. 이 seed 의 성공은 조작 성공이 아니다
UNNORM_KEY = "bridge_orig"                     # <- 실습 4 에서 근거와 함께 확정한 key
INSTRUCTION = "pick up the cube"               # 영어 단문 고정 (OpenVLA prompt 틀)
POS_LIMIT = 0.1                                # pd_ee_delta_pose 의 위치 한계 (m). action 1.0 = 0.1 m
ROT_SCALE = -0.1                               # 회전 스케일 (rad). 부호가 반전돼 있다 (계약 표 4번)
REACH_DIST = 0.05                              # reached 임계값 (m). 근거는 아래 판정식 주석
LIFT_Z = 0.04                                  # lifted 임계값 (m). 근거는 아래 판정식 주석


def to_maniskill_action(raw_action):
    """OpenVLA 역정규화 출력(7,) 을 ManiSkill pd_ee_delta_pose action(7,) 으로 변환한다.

    변환 규칙의 근거는 `action_contract.md` 의 계약 표에 있다. OpenVLA 는 물리량(m, rad)을
    돌려주고 ManiSkill 은 전 차원 [-1, 1] 정규화값을 받으므로, 이 함수를 거치지 않으면
    위치 명령이 의도의 1/10 로 줄고 회전이 반대로 돈다.

    Args:
        raw_action: `vla.predict_action()` 출력 numpy 배열 (7,).
            [dx,dy,dz](m), [drx,dry,drz](rad), gripper[0,1]

    Returns:
        ManiSkill action (7,) float32. 전 차원 [-1, 1] 정규화값
    """
    pos = raw_action[:3] / POS_LIMIT           # 미터 -> 정규화 (±0.1 m 가 ±1)
    rot = raw_action[3:6] / ROT_SCALE          # 라디안 -> 정규화, rot_lower 곱셈 때문에 부호 반전
    rot_norm = np.linalg.norm(rot)             # 회전은 축별이 아니라 3벡터 노름으로 제한된다
    if rot_norm > 1.0:                         # 노름이 1 을 넘으면 방향을 유지한 채 축소
        rot = rot / rot_norm
    grip = 2.0 * raw_action[6] - 1.0           # [0,1](0=닫힘) -> [-1,1](-1=닫힘)
    action = np.concatenate([np.clip(pos, -1, 1), rot, [np.clip(grip, -1, 1)]])
    return action.astype(np.float32)           # env.step 이 받는 dtype 으로 맞춘다


def to_vec(pose_field):
    """배치 텐서로 오는 좌표를 (3,) numpy 벡터로 바꾼다.

    Args:
        pose_field: `pose.p` 같은 (1, 3) 형태의 GPU 텐서

    Returns:
        (3,) float numpy 배열
    """
    return np.asarray(pose_field.cpu())[0]     # GPU -> CPU -> numpy, 배치 차원 제거


print("=" * 60)
print("실습 6: zero-shot baseline")
print("=" * 60)


# -- 6-1. 모델 로드 (Phase 4 week6 과 동일 설정) --
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="eager",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    quantization_config=bnb_config,
)
prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"   # OpenVLA 가 학습된 문장 틀


# -- 6-2. 환경 생성 (실습 5 와 동일 조건 + 관측 카메라 해상도 고정) --
env = gym.make(
    ENV_ID,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",           # OpenVLA 출력 형식과 대응 (실습 2)
    render_mode="rgb_array",
    sensor_configs=dict(width=224, height=224),   # 모델 입력 크기와 일치 -> 리사이즈 열화 없음
    max_episode_steps=MAX_EPISODE_STEPS,       # 실습 5 와 같은 env step 예산
)
base = env.unwrapped                           # 큐브·TCP 좌표는 wrapper 를 벗겨야 보인다


# -- 6-3. episode 루프 --
records = []                                   # episode 별 결과를 모은다
grip_raw = []                                  # raw_action[6] 전량. 계약 표 5번의 [0,1] 가정 검증용
for seed in SEEDS:
    obs, info = env.reset(seed=seed)           # 고정 seed 로 초기 배치 재현
    stages = {"reached": False, "grasped": False, "lifted": False, "placed": False}
    done = False                               # 이 episode 를 끝낼지 여부
    for policy_step in range(POLICY_STEPS):    # 정책 결정 50회
        # (a) 관측에서 카메라 이미지 추출 — 키 경로는 실습 2 의 2-3 출력대로
        frame = obs["sensor_data"]["base_camera"]["rgb"]
        frame = np.asarray(frame.cpu() if hasattr(frame, "cpu") else frame)   # GPU 텐서면 내린다
        if frame.ndim == 4:                    # (1, H, W, 3) 이면 첫 장만
            frame = frame[0]
        image = Image.fromarray(frame.astype(np.uint8))       # 이미 224x224 이므로 리사이즈 불필요

        # (b) 추론 — attention_mask 는 넘기지 않는다 (Phase 4 week6 의 크래시 회피)
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.float16)
        with torch.no_grad():                  # 학습이 아니므로 기울기 계산을 끈다
            raw_action = vla.predict_action(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                unnorm_key=UNNORM_KEY,         # 실습 4 에서 확정
                do_sample=False,               # 결정적 출력 (제어에서는 무작위성 배제)
            )
        grip_raw.append(float(raw_action[6]))  # gripper 차원 원값을 남긴다 (계약 표 5번 검증)

        # (c) 변환 — 실습 4 의 action_contract.md 에 쓴 변환 함수를 그대로 쓴다
        action = to_maniskill_action(raw_action)

        # (d) 실행 — 같은 action 을 ACTION_REPEAT 번 넣어 실효 주기를 5 Hz 로 맞춘다
        for _ in range(ACTION_REPEAT):
            obs, reward, terminated, truncated, info = env.step(action)

            # (e) 부분 도달률 갱신 — 판정 경로는 실습 3 의 3-4 출력에서 확정한 것
            tcp = to_vec(base.agent.tcp.pose.p)               # 그리퍼 끝 현재 위치
            cube = to_vec(base.cube.pose.p)                   # 큐브 현재 위치
            # reached: 실습 5 scripted 정책의 접근 고도(5 cm) 와 같은 값 -> 상한 대조와 같은 척도
            stages["reached"] |= bool(np.linalg.norm(tcp - cube) < REACH_DIST)
            # grasped: 환경이 이미 계산해 info 로 실어 보낸다 -> 판정식을 새로 만들지 않는다
            stages["grasped"] |= bool(info["is_grasped"].item())
            # lifted: 큐브가 모서리로 기울어 서 있을 때 중심 높이가 최대 0.02*sqrt(3)=0.0346 m 이므로,
            #         그보다 위인 0.04 m 를 기준으로 잡으면 "기울어짐"을 "들림"으로 오판하지 않는다
            stages["lifted"] |= bool(cube[2] > LIFT_Z)
            stages["placed"] |= bool(info["success"].item())

            if stages["placed"] or terminated or truncated:   # 성공 또는 환경이 끝냄
                done = True
                break                          # repeat 루프 탈출
        if done:
            break                              # 정책 루프 탈출

    records.append({"seed": seed, "policy_steps": policy_step + 1, **stages})
    print(f"   seed{seed:02d}: {records[-1]}")


env.close()


# -- 6-4. 요약 + 원시 결과 저장 --
print("\n[6-4] 요약")
for stage in ["reached", "grasped", "lifted", "placed"]:
    hit = sum(record[stage] for record in records)          # 해당 단계 도달 episode 수
    print(f"   {stage}: {hit}/{len(SEEDS)}")
# 무행동 하한 seed 의 성공은 조작 능력의 증거가 아니므로 걷어낸 값을 함께 적는다 (실습 5-3)
earned = sum(record["placed"] for record in records if record["seed"] not in NOOP_SEEDS)
print(f"   placed 중 조작으로 얻은 것: {earned}/{len(SEEDS)} (하한 seed {NOOP_SEEDS} 제외)")


# gripper 원값 분포 — 계약 표 5번이 가정한 [0,1] 이산값인지 확인한다
print("\n[6-4] raw_action[6] (gripper) 분포")
print(f"   최소 {min(grip_raw):.3f} / 최대 {max(grip_raw):.3f} / 평균 {sum(grip_raw) / len(grip_raw):.3f}")
print(f"   0.1 미만 {sum(g < 0.1 for g in grip_raw)} / 0.9 초과 {sum(g > 0.9 for g in grip_raw)}"
      f" / 그 사이 {sum(0.1 <= g <= 0.9 for g in grip_raw)} (전체 {len(grip_raw)})")


with open("outputs/zeroshot_baseline.json", "w") as f:
    json.dump({"env_id": ENV_ID, "max_episode_steps": MAX_EPISODE_STEPS,
               "action_repeat": ACTION_REPEAT, "unnorm_key": UNNORM_KEY,
               "reach_dist": REACH_DIST, "lift_z": LIFT_Z, "noop_seeds": NOOP_SEEDS,
               "seeds": SEEDS, "records": records, "gripper_raw": grip_raw}, f, indent=2)
print("저장 완료: outputs/zeroshot_baseline.json")
