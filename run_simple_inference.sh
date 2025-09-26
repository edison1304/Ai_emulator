#!/bin/bash

# 간단한 체크포인트 inference 실행 스크립트
# 사용법: ./run_simple_inference.sh

# 기본 경로 설정 (필요에 따라 수정하세요)
TRAIN_CONFIG_PATH="train_output/config.yaml"
CHECKPOINT_PATH="train_output/training_checkpoints/ckpt.tar"
OUTPUT_DIR="inference_output"

echo "=== 간단한 체크포인트 Inference 수행 ==="
echo "훈련 설정: $TRAIN_CONFIG_PATH"
echo "체크포인트: $CHECKPOINT_PATH"
echo "출력 디렉토리: $OUTPUT_DIR"
echo ""

# 파일 존재 확인
if [ ! -f "$TRAIN_CONFIG_PATH" ]; then
    echo "오류: 훈련 설정 파일을 찾을 수 없습니다: $TRAIN_CONFIG_PATH"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "오류: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT_PATH"
    exit 1
fi

# 간단한 inference 실행
echo "간단한 inference 실행 중..."
python simple_inference_from_checkpoint.py \
    --train-config "$TRAIN_CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Inference 완료 ==="
