# 측정 조건·절차 — openvla-lora-runpod

> 측정 대상: OpenVLA 7B 를 `maniskill_pickcube` 로 LoRA 파인튜닝했을 때의 **소요 시간·VRAM·비용**, 그리고 학습 지표의 추이
> 측정 코드: upstream `vla-scripts/finetune.py` (기준 커밋 `c8f03f48`, [`scripts/openvla_registration.patch`](scripts/openvla_registration.patch) 적용)
> 원본 데이터: [`raw/loss_history.csv`](raw/loss_history.csv) (201 표본) · [`raw/run-eoy46jsv.wandb`](raw/run-eoy46jsv.wandb) (그 원본) · [`raw/train.log`](raw/train.log) (진행률·저장 이력)
> 실행 환경: [`environment.md`](environment.md)

## 0. 이 문서가 답하는 질문

> **"이 숫자는 정확히 무엇을 잰 값인가."**

"LoRA 1사이클이 2시간 걸렸다" 는 그 자체로 의미가 확정되지 않는다. 스텝을 무엇으로 세는지, 그 안에 체크포인트 저장이 포함되는지, 배치 조합이 무엇인지에 따라 같은 GPU 에서 전혀 다른 값이 나온다. 이 문서는 그 조건을 전부 적는다.

문서 구성:

| 절 | 내용 |
|---|---|
| §1 | 스텝의 정의 — 이 측정에서 가장 오해하기 쉬운 지점 |
| §2 | probe 측정 (예산 역산용 사전 측정) |
| §3 | 본 사이클 실행 조건 |
| §4 | 시간 측정 방법과 분해 |
| §5 | 학습 지표의 정의와 추출 방법 |
| §6 | 산출물 회수와 검증 절차 |
| §7 | 이 측정이 재지 못한 것 |

---

## 1. 스텝의 정의

`finetune.py` 에는 두 종류의 "스텝" 이 있고 값이 16배 차이 난다.

| 이름 | 코드 | 뜻 |
|---|---|---|
| micro-batch | `batch_idx` | 데이터 배치 하나의 순전파·역전파 |
| **gradient step** | `gradient_step_idx = batch_idx // grad_accumulation_steps` | 옵티마이저가 가중치를 갱신하는 단위 |

**`--max_steps` 와 진행률 표시(`s/it`)는 모두 gradient step 기준이다.** `progress.update()` 가 optimizer step 블록 안에 있기 때문이다(`finetune.py:315-318`). 따라서 이 문서의 "스텝당 2.53초" 는 micro-batch 16개를 처리하고 가중치를 한 번 갱신하는 데 걸린 시간이며, micro-batch 하나는 그 1/16 인 약 0.16초다.

유효 배치는 `batch_size x grad_accumulation_steps x GPU 수` 이고 이 측정에서는 `1 x 16 x 1 = 16` 이다.

데이터셋이 3,760 프레임이므로 **1 epoch = 3760 / 16 = 235 gradient step** 이다.

---

## 2. probe 측정 (사전 측정)

본 학습 전에 20 스텝만 실행해 스텝당 시간과 VRAM 피크를 재고, 거기서 총 스텝 수와 비용을 역산했다.

```bash
export WANDB_MODE=offline
nvidia-smi --query-gpu=memory.used --format=csv -l 2 > probe_vram.csv &
SMI_PID=$!

time torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /workspace/data \
  --dataset_name maniskill_pickcube_only \
  --run_root_dir /workspace/runs \
  --adapter_tmp_dir /workspace/adapter-tmp \
  --lora_rank 32 --batch_size 1 --grad_accumulation_steps 16 \
  --max_steps 20 --save_steps 1000

kill $SMI_PID
sort -t, -k1 -n probe_vram.csv | tail -1
```

측정 설계에서 중요한 선택 셋:

| 선택 | 이유 |
|---|---|
| `--save_steps` 를 `max_steps` 보다 크게 | **저장이 시간 측정을 오염시킨다.** 저장 한 번이 스텝 수십 개분 시간을 쓴다 (§4.2) |
| VRAM 을 2초 간격 백그라운드 기록 | 학습 후 `nvidia-smi` 를 한 번 찍으면 피크가 이미 지나가 있다 |
| `time` 의 real 이 아니라 tqdm 의 `s/it` 를 채택 | 총 시간에는 모델 로딩(약 93초)이 섞인다. 그 값은 스텝 수에 비례하지 않는다 |

### 2.1 probe 결과

| 항목 | 값 |
|---|---|
| gradient step 당 | **2.67초** |
| 20스텝 학습 구간 | 58초 |
| 전체 (`time` real) | 2분 31초 |
| 모델 로딩 등 고정 비용 | 약 93초 |
| **VRAM 피크** | **18,470MiB / 24,564MiB** (75%) |

`Max step 20 reached! Stopping training...` 출력으로 `--max_steps` 가 인식됐음을 확인했다. 기본값은 200,000 이다.

VRAM 원본(`probe_vram.csv`)은 container disk 에 있어 회수하지 못했다. 위 피크 값은 실행 직후 화면에서 읽은 값이다.

---

## 3. 본 사이클 실행 조건

```bash
export WANDB_MODE=offline
export WANDB_DIR=/workspace/wandb

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /workspace/data \
  --dataset_name maniskill_pickcube_only \
  --run_root_dir /workspace/runs \
  --adapter_tmp_dir /workspace/adapter-tmp \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 16 \
  --max_steps 2000 \
  --save_steps 1000 \
  2>&1 | tee /workspace/train.log
```

| 인자 | 값 | 근거 |
|---|---|---|
| `batch_size` / `grad_accumulation_steps` | 1 / 16 | probe 에서 OOM 없이 성립한 조합 |
| `max_steps` | 2,000 | 8.5 epoch. probe 의 스텝당 시간과 예산에서 역산 |
| `save_steps` | 1,000 | 저장 비용이 커서 2회로 제한 (§4.2) |
| `lora_rank` | 32 | upstream 기본값 |
| `learning_rate` / `image_aug` / `lora_dropout` | 5e-4 / True / 0.0 | 스크립트 기본값 |

**`max_steps` 는 `save_steps` 의 배수로 잡았다.** 저장 조건(`finetune.py:321`)이 `gradient_step_idx % save_steps == 0` 이고 종료 조건(`367`)이 그 뒤에 평가되므로, 배수가 아니면 마지막 학습 상태가 저장되지 않고 끝난다.

실행은 tmux 세션 안에서 했다. SSH 연결이 끊기면 그 자식 프로세스가 함께 죽기 때문이다. 스크립트는 `set -eo pipefail` 로 시작한다 — `set -e` 만으로는 `| tee` 뒤의 실패를 잡지 못한다(파이프의 종료 상태는 마지막 명령의 것이다).

---

## 4. 시간 측정

### 4.1 측정 방법

세 출처를 쓴다.

| 출처 | 무엇을 주는가 |
|---|---|
| tqdm 의 `s/it` (`raw/train.log`) | gradient step 당 시간. 저장 중에는 갱신되지 않으므로 **순수 학습 속도** |
| tqdm 의 경과 시간 `[H:MM:SS<...]` | 특정 스텝 도달 시각. 저장 시간이 포함된다 |
| wandb `_runtime` (`raw/loss_history.csv`) | 실행 시작 이후 경과 초 |

저장 시간은 직접 재지 않고 **차이로 구했다**: 스텝 A 와 스텝 B 의 경과 시간 차이에서 (B-A) x 스텝당 시간을 뺀 값.

### 4.2 결과

| 항목 | 값 | 출처 |
|---|---|---|
| gradient step 당 | **2.51-2.58초** | tqdm `s/it`, 전 구간 |
| 총 소요 | **94.3분** | wandb `_runtime` 최종값 5,656초 |
| 스텝 1000 저장 | 약 7분 | 경과 시간 차이 계산 |
| 스텝 2000 저장 | 약 30초 | 종료 직전 구간 |

구간별 표본:

```
 45%|  900/2000 [39:22<46:02, 2.51s/it]
 60%| 1200/2000 [59:00<33:43, 2.53s/it]
 90%| 1800/2000 [1:24:20<08:25, 2.53s/it]
 98%| 1950/2000 [1:30:40<02:09, 2.58s/it]
```

### 4.3 저장 동작의 두 가지 성질

**반복된다.** 저장 조건이 배치 루프 안에 있고 `gradient_step_idx` 가 `grad_accumulation_steps` 개의 배치 동안 같은 값을 유지하므로, **한 번의 저장 이벤트가 16회로 늘어난다.** 매 회 base 3 shard 를 읽어 어댑터를 머지하고 15GB 를 network volume 에 쓴다.

```
$ grep -c "Saved Model Checkpoint for Step 1000" raw/train.log
16
$ grep -c "Saved Model Checkpoint for Step 2000" raw/train.log
1
```

**마지막 저장은 반복되지 않는다.** `max_steps` 에 도달하면 첫 저장 직후 `break` 하기 때문이다.

저장 이벤트 1회의 소요는 **실행마다 7-25분으로 변동**했다. network volume 이 MooseFS 공유 파일시스템이라 쓰기 속도가 클러스터 상태에 좌우되는 것으로 보인다. 계획에는 보수적인 값(25분)을 쓰고 실측은 매 실행 기록한다.

---

## 5. 학습 지표

### 5.1 세 지표의 정의

`finetune.py` 가 10 gradient step 마다 세 값을 기록한다. 각각 직전 구간의 micro-batch 들에 대한 평균이다.

| 지표 | 뜻 |
|---|---|
| `train_loss` | action 을 토큰으로 이산화해 예측하는 교차 엔트로피. 정답 토큰에 부여한 확률이 낮을수록 크다 |
| `action_accuracy` | 예측 토큰이 정답 토큰과 일치한 비율 |
| `l1_loss` | 토큰을 되돌린 연속값과 정답의 평균 절대 오차 (정규화 공간) |

**세 값 모두 학습 데이터에 대한 것이다.** 스크립트는 `train[:95%]` 만 사용하고 나머지로 검증하지 않는다. held-out 평가가 없다.

`l1_loss` 는 정규화 공간의 값이므로, 미터·라디안으로 환산하려면 [`raw/dataset_statistics.json`](raw/dataset_statistics.json) 의 `std` 를 곱해야 한다.

### 5.2 추출 방법

loss 는 **`train.log` 에 없다.** `finetune.py:305` 가 `wandb.log` 로만 보내고 stdout 에 출력하지 않는다. offline 모드였으므로 기록은 protobuf 스트림([`raw/run-eoy46jsv.wandb`](raw/run-eoy46jsv.wandb))으로 남았고, 다음으로 읽어 [`raw/loss_history.csv`](raw/loss_history.csv) 로 변환했다.

```python
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

ds = DataStore()
ds.open_for_scan("run-eoy46jsv.wandb")
while True:
    data = ds.scan_data()
    if data is None:
        break
    rec = pb.Record()
    rec.ParseFromString(data)
    if rec.HasField("history"):
        for item in rec.history.item:
            key = item.key or ".".join(item.nested_key)   # <- 이 줄이 핵심
            value = json.loads(item.value_json)
```

**`item.key` 만 보면 키가 전부 빈 문자열로 나와 값이 뒤섞인다.** 실제 키는 `nested_key` 에 들어 있다.

CSV 는 201행(스텝 0-2000, 10 스텝 간격)이고 열은 `_step, train_loss, action_accuracy, l1_loss, _runtime` 이다.

---

## 6. 산출물 회수와 검증

### 6.1 회수 대상

| 대상 | 크기 | 회수 |
|---|---|---|
| LoRA 어댑터 `adapter_model.safetensors` | 462MB | 예 |
| `adapter_config.json` | 988B | 예 ([`raw/`](raw/adapter_config.json)) |
| `dataset_statistics.json` | 2.3KB | 예 ([`raw/`](raw/dataset_statistics.json)) |
| processor / tokenizer 설정 8개 | 약 2.4MB | 예 |
| wandb 기록 | 755KB | 예 ([`raw/`](raw/run-eoy46jsv.wandb)) |
| `train.log` | 149KB | 예 ([`raw/`](raw/train.log)) |
| 머지 가중치 4 shard | 약 15GB | **아니오** |

머지 가중치를 받지 않은 것은 **머지 가중치 = base + 어댑터**이고 base 15GB 가 로컬 캐시에 있기 때문이다. 어댑터로 다시 합치면 같은 모델이 된다.

### 6.2 전송 경로

세 경로를 시도했고 하나만 동작했다.

| 경로 | 결과 |
|---|---|
| rsync over 프록시 SSH | **불가.** 프록시가 원격 명령 실행을 거부한다(`Your SSH client doesn't support PTY`). rsync 는 원격에서 rsync 를 실행하는 구조라 성립하지 않는다 |
| `runpodctl` (P2P) | **불가.** pod -> 로컬 방향은 로컬이 고포트로 아웃바운드 연결을 걸어야 하는데 그것이 막혀 데이터가 한 바이트도 오지 않았다. 반대 방향은 성공했다 |
| **network volume 의 S3 호환 API** | **가능.** HTTPS 만 쓰고 멀티파트라 재개된다. pod 없이도 접근된다 |

S3 경로의 제약: `aws s3 cp` 는 사전 `HeadObject` 호출이 403 으로 거부되어 실패한다. 저수준 명령을 쓴다.

```bash
aws configure set default.s3.addressing_style path
aws s3api get-object --bucket <volume-id> --key recover.tar.gz \
    --endpoint-url https://s3api-eu-ro-1.runpod.io recover.tar.gz
```

버킷 이름은 **volume ID** 이고 볼륨 이름이나 pod ID 가 아니다. S3 API 는 일부 데이터센터에서만 제공되므로, **볼륨을 만들 때 지원 데이터센터를 고르는 것이 회수 경로를 미리 확보하는 방법이다.**

### 6.3 검증 항목

| 항목 | 확인한 것 | 결과 |
|---|---|---|
| 파일 수·바이트 합 (데이터 전송 시) | 로컬과 pod 양쪽에서 `find -type f` 개수와 파일 바이트 합 | 5 / 5, 261,606,336 / 261,606,336 |
| 데이터 로드 재현 | week2 와 **같은 스크립트**를 pod 에서 실행 (경로만 심볼릭 링크로 맞춤) | 구조·통계 해시 일치 |
| 어댑터 | `r`, `base_model_name_or_path`, base 리비전 | 32, `openvla/openvla-7b`, `47a0ec7f...` |
| base 리비전 대조 | 로컬 HuggingFace 캐시 스냅샷 해시 | 일치 |
| 통계 파일 | `action.mean[6]` 을 week1 실측과 대조 | 0.4319 vs 0.432 |

로드 재현 검증에서 **dim 별 min/max 값은 대조 항목이 아니다.** 로더에 `shuffle_buffer_size` 가 걸려 있어 실행마다 다른 샘플이 나온다. 판정은 구조 일치·통계 해시 동일·값의 범위로 한다.

---

## 7. 이 측정이 재지 못한 것

- **held-out 성능.** `train[:95%]` 만 사용하고 검증 분할을 계산하지 않는다. 모든 지표가 학습 데이터에 대한 값이다
- **정책의 실제 성공률.** 학습이 끝났을 뿐 시뮬레이터에서 돌려 보지 않았다
- **VRAM 피크의 원본 시계열.** `probe_vram.csv` 가 container disk 에 있어 회수하지 못했다. 피크 값 18,470MiB 는 실행 직후 화면에서 읽은 값이다
- **머지 가중치의 로드 가능성.** volume 에 15GB 가 저장된 것은 확인했으나 로드해 보지 않았다
- **재머지 결과가 원본 머지와 동일한지.** 리비전과 버전 조합은 맞췄으나 비교하지 않았다
- **batch 2 조합.** VRAM 여유가 6GB 남았으나 시도하지 않았다
- **volume 재기동 잔존.** pod 정지 -> 재시작 검증은 수행하지 않았다
