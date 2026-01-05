#!/bin/bash
# Enhanced RLOO训练脚本
# 使用所有创新点进行训练

echo "=========================================="
echo "Enhanced RLOO Training"
echo "=========================================="

# 设置GPU（可选，如果不设置则使用所有GPU）
# export CUDA_VISIBLE_DEVICES=0

# 训练参数
EPOCHS=3
DATA_LIMIT=500
LR=1e-7
BATCH_SIZE=1
NUM_SAMPLES=4

# 运行训练（使用所有创新点）
python RLOO_enhanced_train.py \
    --epochs $EPOCHS \
    --data-limit $DATA_LIMIT \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --num-samples $NUM_SAMPLES \
    --use-value-head \
    --use-chain-credit \
    --lambda-baseline 0.7

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Evaluate: python evaluate_rloo_enhanced.py"
echo "2. View logs: tensorboard --logdir /root/Train-RL/outputs/rloo_enhanced/tensorboard"
echo ""

