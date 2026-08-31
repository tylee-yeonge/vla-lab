#!/bin/bash
# pod 컨테이너가 뜰 때 실행되는 스크립트
set -e

# SSH 로 들어온 셸은 Docker 의 환경변수를 물려받지 않는다. 파일로 남겨야 보인다.
#   PATH 가 빠지면 conda 의 python 을 못 찾아 `python: command not found` 가 나고,
#   LD_LIBRARY_PATH / CUDA_HOME 이 빠지면 학습에서 CUDA 라이브러리를 못 찾으며,
#   HF_HOME 이 빠지면 15GB 짜리 모델이 volume 이 아니라 pod 기본 디스크로 떨어진다
# `|| true` 가 필요한 이유: grep 은 매치가 하나도 없으면 실패를 반환하고,
#   set -e 가 그것을 보고 스크립트를 여기서 끝내 버린다. 그러면 sshd 가 뜨지 않는다
env | grep -E '^(PATH|LD_LIBRARY_PATH|CUDA_HOME|HF_HOME|HF_TOKEN|HUGGINGFACE)=' \
    >> /etc/environment || true

# RunPod 이 주입한 공개키를 등록해야 내 키로 접속할 수 있다
mkdir -p /root/.ssh
echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys

# 호스트 키 생성 후 sshd 기동
ssh-keygen -A
/usr/sbin/sshd

# 마지막 줄이 컨테이너를 살려 둔다.
# sshd 는 백그라운드로 빠지므로 이 줄이 없으면 시작 명령이 끝나고 pod 가 죽는다
sleep infinity