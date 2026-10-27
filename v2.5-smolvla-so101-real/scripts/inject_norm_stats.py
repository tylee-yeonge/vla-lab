"""smolvla_base 의 로컬 사본에 SO-101 데이터셋의 정규화 통계를 넣고, 데이터셋 프레임 하나로 원본과 사본의 출력을 비교한다.

왜 필요한가:
  lerobot 0.6.2 의 smolvla_base 체크포인트는 normalizer · unnormalizer 통계를 so100 계열 키 (`so100.buffer.action.*` 등) 로만
  가지고 있다. SO-101 rollout 의 키는 `observation.state` · `action` 이라 통계가 적용되지 않아, 입력이 정규화되지 않고
  출력이 도 단위로 펴지지 않는다 (6 관절 명령이 0 근처). lerobot-rollout 은 이 통계를 덮어쓰지 않는다.
  (lerobot-train 은 학습 데이터셋 통계로 덮어쓰므로 파인튜닝 쪽은 영향이 없다.)

키 규칙은 '<feature>.<stat>' 이고, 기존 so100.* 키는 그대로 둔다.

사용 (GPU 1개, 팔 불필요):
  python inject_norm_stats.py
"""

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from safetensors.torch import save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class
from lerobot.policies import make_pre_post_processors

# HF 캐시의 smolvla_base 스냅샷 (lerobot-rollout 을 한 번 돌렸거나 hf download 로 받아 둔 상태)
SNAP = sorted(Path("/root/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots").glob("*"))[0]
# 통계를 넣은 사본을 둘 곳
OUT = Path("/workspace/study/physical-ai-study/Studies/Hardware-Arm/v25/outputs/smolvla_base_so101stats")
# 통계를 가져올 SO-101 데이터셋 (로컬 폴더)
DS_REPO = "tylee-yeonge/so101-pick-cube-v25"
DS_ROOT = "/root/.cache/huggingface/lerobot/tylee-yeonge/so101-pick-cube-v25"
TASK = "Pick up the pink cube and place it in the yellow square."
PRE = "policy_preprocessor_step_5_normalizer_processor.safetensors"
POST = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"


def build_copy() -> None:
    """스냅샷을 실제 파일로 복사하고 normalizer · unnormalizer 에 state · action 의 mean · std 를 더한다."""
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SNAP, OUT, symlinks=False)  # HF 캐시는 심링크라 실제 파일로 푼다
    stats = json.load(open(f"{DS_ROOT}/meta/stats.json"))
    add = {f"{f}.{s}": torch.tensor(stats[f][s], dtype=torch.float32) for f in ("observation.state", "action") for s in ("mean", "std")}
    for fname, keys in ((PRE, add), (POST, {k: v for k, v in add.items() if k.startswith("action.")})):
        path = OUT / fname
        with safe_open(str(path), "pt") as f:
            meta = f.metadata()  # 메타데이터를 보존해 다시 쓴다
        tensors = load_file(str(path))
        tensors.update(keys)
        save_file(tensors, str(path), metadata=meta)


def first_action(path: str, frame: dict) -> list:
    """체크포인트 path 로 전처리 -> 정책 -> 후처리를 한 번 돌려 첫 action (도) 을 돌려준다."""
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).eval().to("cuda")
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path, dataset_stats=None,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"},
                                                                 "rename_observations_processor": {"rename_map": {}}})
    # rollout 과 같은 키로 넣는다 (전체 뷰 = camera1, 손목 = camera2)
    obs = {"observation.state": frame["observation.state"], "observation.images.camera1": frame["observation.images.front"],
           "observation.images.camera2": frame["observation.images.wrist"], "task": TASK}
    with torch.no_grad():
        policy.reset()
        action = post(policy.select_action(pre(obs)))
    return [round(float(x), 1) for x in action.squeeze().cpu()]


def main() -> None:
    """사본을 만들고 에피소드 0 의 300 번째 프레임 (팔이 움직이는 중) 으로 원본과 사본을 비교한다."""
    build_copy()
    frame = LeRobotDataset(DS_REPO, root=DS_ROOT)[300]
    print("state          :", [round(float(x), 1) for x in frame["observation.state"]])
    print("smolvla_base   :", first_action("lerobot/smolvla_base", frame))  # 2026-10-22: [-0.8, 0.6, -0.3, 0.6, 2.1, -0.5]
    print("so101stats 사본:", first_action(str(OUT), frame))  # 2026-10-22: [34.5, 49.5, -7.2, 102.5, 26.7, 18.7]


if __name__ == "__main__":
    main()
