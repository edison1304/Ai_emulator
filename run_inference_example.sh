#!/bin/bash

# 체크포인트에서 inference를 수행하는 예시 스크립트
# 사용법: ./run_inference_example.sh

# 기본 경로 설정 (필요에 따라 수정하세요)
CONFIG_PATH="train_output/config.yaml"
CHECKPOINT_PATH="train_output/training_checkpoints/ckpt.tar"
EMA_CHECKPOINT_PATH="train_output/training_checkpoints/ema_ckpt.tar"
OUTPUT_DIR="inference_output"

echo "=== 체크포인트에서 Inference 수행 ==="
echo "설정 파일: $CONFIG_PATH"
echo "체크포인트: $CHECKPOINT_PATH"
echo "EMA 체크포인트: $EMA_CHECKPOINT_PATH"
echo "출력 디렉토리: $OUTPUT_DIR"
echo ""

# 파일 존재 확인
if [ ! -f "$CONFIG_PATH" ]; then
    echo "오류: 설정 파일을 찾을 수 없습니다: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "오류: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT_PATH"
    exit 1
fi

# EMA 체크포인트가 있으면 사용, 없으면 생략
if [ -f "$EMA_CHECKPOINT_PATH" ]; then
    echo "EMA 체크포인트를 사용합니다."
    python temp_inference_from_checkpoint.py \
        --config "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --ema-checkpoint "$EMA_CHECKPOINT_PATH" \
        --output-dir "$OUTPUT_DIR"
else
    echo "EMA 체크포인트가 없습니다. 일반 체크포인트만 사용합니다."
    python temp_inference_from_checkpoint.py \
        --config "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "=== Inference 완료 ==="
