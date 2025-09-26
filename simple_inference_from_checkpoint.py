#!/usr/bin/env python3
"""
간단한 체크포인트 inference 스크립트
기존 inference 모듈을 활용하여 체크포인트에서 inference 수행
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def create_inference_config_from_train_config(train_config_path: str, checkpoint_path: str, output_dir: str) -> str:
    """
    훈련 설정에서 inference 설정을 생성
    """
    import yaml
    
    # 훈련 설정 로드
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # validation_loader에서 데이터 경로 추출
    validation_loader = train_config.get('validation_loader', {})
    dataset_config = validation_loader.get('dataset', {})
    
    # inference 설정 생성 (올바른 구조)
    inference_config = {
        'experiment_dir': output_dir,
        'n_forward_steps': train_config.get('inference', {}).get('n_forward_steps', 2),
        'checkpoint_path': checkpoint_path,
        'logging': {
            'level': 'INFO',
            'log_to_screen': True,
            'log_to_file': True,
            'log_to_wandb': False
        },
        'initial_condition': {
            'path': '/data1/DATA_ARCHIVE/Reanalysis/ERA5_new/data_360x180/daily/multi_lev/t/ERA5_dy_t_195001.nc',  # 실제 NetCDF 파일 사용
            'engine': 'netcdf4',
            'start_indices': {
                'n_initial_conditions': 1,  # 단일 inference를 위해 1로 설정
                'first': 0,
                'interval': 1
            }
        },
        'forcing_loader': {
            'dataset': {
                'data_path': '/data1/DATA_ARCHIVE/Reanalysis/ERA5_new/data_360x180/daily/multi_lev/t',
                'file_pattern': '*.nc'
            },
            'num_data_workers': 0
        },
        'forward_steps_in_memory': 10,
        'data_writer': {
            'save_prediction_files': True,
            'save_monthly_files': True,
            'save_histogram_files': False
        },
        'aggregator': {
            'log_global_mean_time_series': False
        }
    }
    
    # inference 설정 파일 저장
    inference_config_path = os.path.join(output_dir, 'inference_config.yaml')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(inference_config_path, 'w') as f:
        yaml.dump(inference_config, f, default_flow_style=False)
    
    return inference_config_path

def run_inference_with_checkpoint(
    train_config_path: str,
    checkpoint_path: str,
    output_dir: str = "inference_output"
):
    """
    체크포인트를 사용하여 inference 실행
    """
    
    print(f"훈련 설정 로드: {train_config_path}")
    print(f"체크포인트: {checkpoint_path}")
    print(f"출력 디렉토리: {output_dir}")
    
    # 파일 존재 확인
    if not os.path.exists(train_config_path):
        raise FileNotFoundError(f"훈련 설정 파일을 찾을 수 없습니다: {train_config_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
    
    # inference 설정 생성
    print("inference 설정 생성 중...")
    inference_config_path = create_inference_config_from_train_config(
        train_config_path, checkpoint_path, output_dir
    )
    print(f"inference 설정 저장: {inference_config_path}")
    
    # inference 실행
    print("inference 실행 중...")
    try:
        # 기존 inference 모듈 사용
        cmd = [
            sys.executable, "-m", "fme.ace.inference",
            inference_config_path
        ]
        
        print(f"실행 명령: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("inference 성공적으로 완료!")
            print("출력:")
            print(result.stdout)
        else:
            print("inference 실행 중 오류 발생:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"inference 실행 중 예외 발생: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="체크포인트에서 inference 수행")
    parser.add_argument("--train-config", required=True, help="훈련 설정 YAML 파일 경로")
    parser.add_argument("--checkpoint", required=True, help="체크포인트 파일 경로")
    parser.add_argument("--output-dir", default="inference_output", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    try:
        success = run_inference_with_checkpoint(
            train_config_path=args.train_config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir
        )
        
        if success:
            print("\n=== Inference 완료 ===")
            sys.exit(0)
        else:
            print("\n=== Inference 실패 ===")
            sys.exit(1)
            
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
