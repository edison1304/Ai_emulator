# SMgNO (Spherical Multigrid Neural Operator) Integration

이 문서는 ACE 프로젝트에서 SMgNO 모델을 사용하는 방법을 설명합니다.

## 개요

SMgNO (Spherical Multigrid Neural Operator)는 기존의 SFNO (Spherical Fourier Neural Operator)와 다른 아키텍처를 사용하는 모델입니다. 주요 차이점은 다음과 같습니다:

### SFNO vs SMgNO

| 특징 | SFNO | SMgNO |
|------|------|-------|
| 아키텍처 | Spherical Fourier Neural Operator | Spherical Multigrid Neural Operator |
| 핵심 기술 | Spectral convolution | Multigrid + CSHFs (Convolution based on Spherical Harmonic Functions) |
| 주요 파라미터 | `filter_type`, `operator_type` | `max_levels`, `smoothing_iterations`, `use_cshfs` |
| 논문 | FourCastNet | "A spherical multigrid neural operator for global weather forecasting" |

## 파일 구조

```
fme/ace/models/modulus/
├── sfnonet.py          # 기존 SFNO 모델
├── smgnonet.py         # 새로운 SMgNO 모델
└── ...

fme/ace/registry/
├── sfno.py             # SFNO 빌더 등록
├── smgnonet.py         # SMgNO 빌더 등록
└── __init__.py         # 모듈 import

train_output/
├── config.yaml         # 기존 SFNO 설정
├── config_smgnonet.yaml # SMgNO 설정
└── ...

run_smgnonet.sh         # SMgNO 실행 스크립트
```

## 설정 파일

### SMgNO Config (`config_smgnonet.yaml`)

주요 설정 차이점:

```yaml
stepper:
  step:
    config:
      builder:
        type: SphericalMultigridNeuralOperatorNet  # SFNO에서 변경
        config:
          # 공통 파라미터
          embed_dim: 384
          filter_type: linear
          hard_thresholding_fraction: 1.0
          use_mlp: true
          normalization_layer: instance_norm
          num_layers: 8
          operator_type: diagonal  # dhconv에서 변경
          scale_factor: 1
          separable: false
          
          # SMgNO 전용 파라미터
          max_levels: 3                    # Multigrid 레벨 수
          smoothing_iterations: 2          # 스무딩 반복 횟수
          use_cshfs: true                  # CSHFs 사용 여부
          mlp_ratio: 2.0                   # MLP 비율
          drop_rate: 0.0                   # Dropout 비율
          drop_path_rate: 0.0              # Drop path 비율
          activation_function: gelu        # 활성화 함수
          big_skip: true                   # Big skip connection
          pos_embed: true                  # Positional embedding
          encoder_layers: 1                # Encoder 레이어 수
          checkpointing: 0                 # Checkpointing 레벨
```

## 사용법

### 1. SMgNO 모델로 훈련 실행

```bash
# 단일 GPU로 실행
./run_smgnonet.sh 1

# 다중 GPU로 실행 (예: 4개 GPU)
./run_smgnonet.sh 4
```

### 2. 기존 SFNO와 비교

```bash
# SFNO 실행 (기존)
torchrun --nproc_per_node 1 -m fme.ace.train train_output/config.yaml

# SMgNO 실행 (새로운)
torchrun --nproc_per_node 1 -m fme.ace.train train_output/config_smgnonet.yaml
```

### 3. 설정 파일 수정

필요에 따라 `config_smgnonet.yaml`에서 다음 파라미터를 조정할 수 있습니다:

- `max_levels`: Multigrid 레벨 수 (기본값: 3)
- `smoothing_iterations`: 스무딩 반복 횟수 (기본값: 2)
- `use_cshfs`: CSHFs 사용 여부 (기본값: true)
- `embed_dim`: 임베딩 차원 (기본값: 384)
- `num_layers`: 레이어 수 (기본값: 8)

## 모델 아키텍처

### SMgNO의 주요 구성 요소

1. **CSHFs (Convolution based on Spherical Harmonic Functions)**
   - 구면 데이터 왜곡 문제 해결
   - 학습 가능한 절단 오차 보상

2. **Multigrid Framework**
   - V-cycle 구조로 계층적 처리
   - 계산 효율성 향상

3. **Semi-iterative Smoothing**
   - 잔차 보정을 반복적 접근으로 대체

4. **Periodic Padding**
   - 경도 방향에서 동서 경계 연속성 보장

5. **Pixel Shuffle**
   - 체커보드 아티팩트 방지를 위한 업샘플링

## 문제 해결

### Python 3.9 호환성

Python 3.9에서는 `TypeAlias`를 지원하지 않습니다. `fme/core/metrics.py`에서 다음과 같이 수정했습니다:

```python
# Python 3.9 호환
from typing import Union
Dimension = Union[int, Iterable[int]]
Array = Union[np.ndarray, torch.Tensor]
```

### Import 오류

모델을 import할 때 오류가 발생하면 다음을 확인하세요:

1. Python 경로 설정: `export PYTHONPATH="/home/yjlee/ace-main:$PYTHONPATH"`
2. 의존성 설치: `torch_harmonics`, `torch` 등
3. Python 버전: 3.9 이상 권장

## 성능 비교

SMgNO와 SFNO의 성능을 비교하려면:

1. 동일한 데이터셋으로 두 모델 훈련
2. 동일한 하이퍼파라미터 사용 (가능한 경우)
3. 메트릭 비교 (MSE, MAE, RMSE 등)

## 참고 자료

- [SMgNO 논문](https://doi.org/10.1038/s41598-025-96208-y)
- [FourCastNet 논문](https://arxiv.org/abs/2202.11214)
- [Modulus 프로젝트](https://github.com/NVIDIA/modulus)
