"""
실습 2: 환경을 직접 생성해 관측 구조를 확인하고 렌더 이미지 1장을 저장
"""
import gymnasium as gym                        # ManiSkill 은 gymnasium 규약을 따른다
import mani_skill.envs                         # import 만으로 ManiSkill task 들이 gym 에 등록된다
import numpy as np                             # 관측 배열 처리
from PIL import Image                          # 렌더 결과 저장


print("=" * 60)
print("실습 2: 환경 생성 + 관측 구조")
print("=" * 60)


# -- 2-1. 등록된 task 이름 확인 (문서를 믿지 않고 설치된 버전에서 직접 본다) --
print("\n[2-1] PickCube 관련 등록 환경")
for env_id in gym.registry.keys():             # gym 에 등록된 모든 환경 이름을 훑는다
    if "PickCube" in env_id:                   # 이름에 PickCube 가 포함된 것만
        print("  ", env_id)                    # 실제 사용할 정확한 id 를 여기서 얻는다


# -- 2-2. 환경 생성 (ENV_ID 를 2-1 출력의 실제 이름으로 교체) --
ENV_ID = "PickCube-v1"                         # <- 2-1 출력의 실제 id 로 교체한다
print(f"\n[2-2] 환경 생성: {ENV_ID}")
env = gym.make(
    ENV_ID,                                    # 사용할 task
    obs_mode="rgb",                            # 관측에 카메라 RGB 이미지를 포함시킨다
    control_mode="pd_ee_delta_pose",           # action 을 EEF 델타 pose 로 해석 (OpenVLA 출력과 대응)
    render_mode="rgb_array",                   # GUI 없이 배열로 렌더 (헤드리스)
    sensor_configs=dict(width=224, height=224), # 관측 카메라를 OpenVLA 입력 크기로 (기본 128 이면 업샘플 열화가 섞인다)
)
print("환경 생성 성공 — Vulkan 렌더 컨텍스트가 붙었다")


# -- 2-3. 관측 구조 확인 (키 이름을 추측하지 않고 출력해서 본다) --
print("\n[2-3] 관측 구조")
obs, info = env.reset(seed=0)                  # seed 를 주면 초기 배치가 재현된다
for key in obs:                                # 관측 최상위 키
    value = obs[key]
    if isinstance(value, dict):                # 한 겹 더 들어가는 항목 (카메라 등)
        for sub_key in value:
            inner = value[sub_key]
            if isinstance(inner, dict):        # 카메라별로 한 겹 더 있는 경우
                for leaf in inner:
                    print(f"   {key}/{sub_key}/{leaf}: {getattr(inner[leaf], 'shape', '')}")
            else:
                print(f"   {key}/{sub_key}: {getattr(inner, 'shape', '')}")
    else:
        print(f"   {key}: {getattr(value, 'shape', type(value).__name__)}")
print("\n[2-3b] info 키:", list(info.keys()))  # 성공 플래그가 어디 실리는지 단서


# -- 2-4. 렌더 이미지 저장 (여기가 통과 판정) --
print("\n[2-4] 렌더 이미지 저장")
frame = env.render().cpu().numpy()             # 현재 장면 렌더 — GPU 텐서로 오므로 cpu 로 내린 뒤 numpy 로 바꾼다
if frame.ndim == 4:                            # (batch, H, W, 3) 형태면 첫 장만 쓴다
    frame = frame[0]
print("frame shape:", frame.shape)             # 사람이 보는 뷰의 해상도. 모델 입력은 2-3 의 sensor_data 쪽이다
Image.fromarray(frame.astype(np.uint8)).save("outputs/env_check.png")
print("저장 완료: outputs/env_check.png")


env.close()                                    # 렌더 컨텍스트 해제
print("\n실습 2 완료")