#!/usr/bin/env python3
"""
임시 스크립트: 체크포인트를 불러와서 inference_one_epoch을 수행
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import fme
import dacite
from fme.ace.train.train import TrainBuilders, TrainConfig
from fme.core.generics.trainer import Trainer
from fme.core.cli import prepare_config


def load_config_from_yaml(config_path: str) -> TrainConfig:
    """YAML 설정 파일에서 TrainConfig 로드"""
    config_data = prepare_config(config_path, override=None)
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_data, config=dacite.Config(strict=True)
    )
    return config


def run_inference_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
    ema_checkpoint_path: str = None,
    output_dir: str = None
):
    """
    체크포인트를 불러와서 inference_one_epoch을 수행하는 함수
    
    Args:
        config_path: 훈련 설정 YAML 파일 경로
        checkpoint_path: 체크포인트 파일 경로
        ema_checkpoint_path: EMA 체크포인트 파일 경로 (선택사항)
        output_dir: 출력 디렉토리 (선택사항)
    """
    
    # 설정 로드
    print(f"설정 파일 로드 중: {config_path}")
    config = load_config_from_yaml(config_path)
    
    # 출력 디렉토리 설정
    if output_dir:
        config.experiment_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # 로깅 설정
    config.logging.configure_logging(config.experiment_dir, log_filename="inference.log")
    logging.info("체크포인트에서 inference 시작")
    
    # 분산 환경 설정
    dist = fme.Distributed.get_instance()
    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True
    
    # TrainBuilders 초기화
    print("TrainBuilders 초기화 중...")
    builders = TrainBuilders(config)
    
    # Trainer 초기화
    print("Trainer 초기화 중...")
    trainer = Trainer(
        stepper=builders.stepper,
        optimization=builders.optimization,
        train_data=builders.train_data,
        validation_data=builders.validation_data,
        inference_data=builders.inference_data,
        train_aggregator=builders.get_train_aggregator(),
        validation_aggregator=builders.get_validation_aggregator(),
        aggregator_builder=builders,
        config=config,
        paths=builders.paths,
    )
    
    # 체크포인트 복원
    print(f"체크포인트 복원 중: {checkpoint_path}")
    if ema_checkpoint_path and os.path.exists(ema_checkpoint_path):
        print(f"EMA 체크포인트 복원 중: {ema_checkpoint_path}")
        trainer.restore_checkpoint(checkpoint_path, ema_checkpoint_path)
    else:
        # EMA 체크포인트가 없으면 None으로 전달
        trainer.restore_checkpoint(checkpoint_path, None)
    
    logging.info(f"체크포인트 복원 완료. Epoch: {trainer._epoch}")
    
    # inference_one_epoch 수행
    print("inference_one_epoch 수행 중...")
    try:
        inference_logs = trainer.inference_one_epoch()
        logging.info("inference_one_epoch 완료")
        print("inference_one_epoch 완료!")
        
        # 결과 출력
        if inference_logs:
            print("\n=== Inference 결과 ===")
            for key, value in inference_logs.items():
                print(f"{key}: {value}")
        
        return inference_logs
        
    except Exception as e:
        logging.error(f"inference_one_epoch 실행 중 오류 발생: {e}")
        print(f"오류 발생: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="체크포인트에서 inference 수행")
    parser.add_argument("--config", required=True, help="훈련 설정 YAML 파일 경로")
    parser.add_argument("--checkpoint", required=True, help="체크포인트 파일 경로")
    parser.add_argument("--ema-checkpoint", help="EMA 체크포인트 파일 경로")
    parser.add_argument("--output-dir", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.config):
        print(f"설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        sys.exit(1)
    
    if args.ema_checkpoint and not os.path.exists(args.ema_checkpoint):
        print(f"EMA 체크포인트 파일을 찾을 수 없습니다: {args.ema_checkpoint}")
        sys.exit(1)
    
    try:
        run_inference_from_checkpoint(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            ema_checkpoint_path=args.ema_checkpoint,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
사용 예시:

1. 기본 사용법:
python temp_inference_from_checkpoint.py \
    --config train_output/config.yaml \
    --checkpoint train_output/training_checkpoints/ckpt.tar \
    --ema-checkpoint train_output/training_checkpoints/ema_ckpt.tar \
    --output-dir inference_output

2. EMA 체크포인트 없이 사용:
python temp_inference_from_checkpoint.py \
    --config train_output/config.yaml \
    --checkpoint train_output/training_checkpoints/ckpt.tar \
    --output-dir inference_output

3. 최고 성능 체크포인트 사용:
python temp_inference_from_checkpoint.py \
    --config train_output/config.yaml \
    --checkpoint train_output/training_checkpoints/best_ckpt.tar \
    --output-dir inference_output
"""
