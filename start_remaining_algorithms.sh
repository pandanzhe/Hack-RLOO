#!/bin/bash
# 启动剩余三个算法（PPO, GRPO, RLOO）
# DPO已经在运行，不需要重复启动

cd /root/Train-RL/clean_dir

echo "=========================================="
echo "启动剩余三个算法（PPO, GRPO, RLOO）"
echo "=========================================="
echo ""
echo "GPU分配:"
echo "  GPU 1: PPO"
echo "  GPU 2: GRPO"
echo "  GPU 3: RLOO"
echo ""
echo "训练参数:"
echo "  Epochs: 3"
echo "  Data limit: 500"
echo "  Batch size: 1"
echo ""
echo "=========================================="
echo ""

# 创建日志目录
mkdir -p logs

# PPO on GPU 1
echo "🚀 启动 PPO 训练 (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python PPO_train.py \
    --epochs 3 \
    --data-limit 500 \
    --lr 1e-6 \
    --batch-size 1 \
    > logs/ppo.log 2>&1 &
PPO_PID=$!
echo "  ✓ PPO PID: $PPO_PID"

# GRPO on GPU 2
echo "🚀 启动 GRPO 训练 (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 python GRPO_train.py \
    --epochs 3 \
    --data-limit 500 \
    --lr 1e-6 \
    --batch-size 1 \
    > logs/grpo.log 2>&1 &
GRPO_PID=$!
echo "  ✓ GRPO PID: $GRPO_PID"

# RLOO on GPU 3
echo "🚀 启动 RLOO 训练 (GPU 3)..."
CUDA_VISIBLE_DEVICES=3 python RLOO_train.py \
    --epochs 3 \
    --data-limit 500 \
    --lr 1e-6 \
    --batch-size 1 \
    > logs/rloo.log 2>&1 &
RLOO_PID=$!
echo "  ✓ RLOO PID: $RLOO_PID"

echo ""
echo "=========================================="
echo "✅ 三个算法已启动！"
echo "=========================================="
echo ""
echo "进程ID:"
echo "  PPO:  $PPO_PID"
echo "  GRPO: $GRPO_PID"
echo "  RLOO: $RLOO_PID"
echo ""
echo "📊 监控命令:"
echo "  查看日志: tail -f logs/ppo.log logs/grpo.log logs/rloo.log"
echo "  查看GPU:  watch -n 1 nvidia-smi"
echo "  查看进程: ps aux | grep python"
echo ""
echo "🛑 停止训练:"
echo "  kill $PPO_PID $GRPO_PID $RLOO_PID"
echo ""

