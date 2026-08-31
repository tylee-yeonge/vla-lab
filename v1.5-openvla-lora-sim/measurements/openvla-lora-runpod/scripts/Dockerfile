# 학습 측 이미지. sim 은 넣지 않는다 (Vulkan 요구로 난이도가 다르고, eval 은 로컬에서 돈다)
# 베이스의 python 은 openvla 핀에 휠이 있는 버전이어야 한다.
# week2 outputs/env_rlds.md 가 기록했듯 tensorflow==2.15.0 은 python 3.12 용 휠이 없다 -- 3.11 이면 핀이 그대로 선다
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# 시스템 의존성 (git 은 리포 클론과 패치 적용에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git rsync && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# OpenVLA 본체. 기준 커밋을 고정한다 -- week2 outputs/openvla_base_commit.txt 의 해시
ARG OPENVLA_COMMIT=c8f03f48af692657d3060c19588038c7220e9af9
RUN git clone https://github.com/openvla/openvla.git && \
    cd openvla && git checkout ${OPENVLA_COMMIT}

# week2 의 등록 변경(3파일)을 패치로 적용. 손으로 다시 고치지 않는다
COPY openvla_registration.patch /tmp/
RUN cd openvla && git apply /tmp/openvla_registration.patch

# 의존성 설치 + 데이터 경로 패키지를 week2 가 검증한 조합으로 덮어쓰기.
# 덮어쓰는 이유: openvla 의 tensorflow==2.15.0 핀이 protobuf 를 4.x 로 묶는데,
#   핀이 없는 전이 의존성 tensorflow-metadata 는 최신이 들어와 protobuf 5.27+ 를 요구한다.
#   그대로 두면 빌드는 성공하고 import 에서 깨진다. 값의 출처는 week2 outputs/pip_freeze_rlds.txt.
# 한 RUN 으로 묶는 이유: 레이어를 나누면 버려질 TF 2.15 가 앞 레이어에 그대로 남아 이미지가 커진다
RUN cd openvla && pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
        "tensorflow==2.21.0" \
        "tensorflow-datasets==4.9.10" \
        "tensorflow-metadata==1.21.0" \
        "protobuf==6.33.6" && \
    pip uninstall -y tensorflow-addons tensorflow-estimator

# flash-attn. pyproject 45행에 주석으로만 적혀 있어 `pip install -e .` 로는 들어오지 않는다.
# 소스 빌드라 30-90분 걸린다 -- pod 가 아니라 여기서 굽는다 (베이스가 -devel 이라 nvcc 가 있다)
RUN pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir "flash-attn==2.5.5" --no-build-isolation

# 데이터셋과 체크포인트는 이미지에 넣지 않는다 -- 실행 시 volume 으로 붙인다
