#!/bin/bash
# 四算法并行训练启动脚本
# 使用方法: ./run_all_algorithms.sh

cd /root/Train-RL/clean_dir

echo "=========================================="
echo "启动4个算法的并行训练"
echo "=========================================="
echo ""
echo "GPU分配:"
echo "  GPU 0: DPO"
echo "  GPU 1: PPO"
echo "  GPU 2: GRPO"
echo "  GPU 3: RLOO"
echo ""
echo "训练参数:"
echo "  Epochs: 3"
echo "  Data limit: 500"
echo "  Batch size: 1 (单卡运行，减少显存使用)"
echo ""
echo "=========================================="
echo ""

# 创建日志目录
mkdir -p logs

# DPO on GPU 0 (batch_size=1 以减少显存使用)
echo "🚀 启动 DPO 训练 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python DPO_train.py \
    --epochs 3 \
    --data-limit 500 \
    --lr 1e-5 \
    --beta 1.0 \
    --batch-size 1 \
    > logs/dpo.log 2>&1 &
DPO_PID=$!
echo "  ✓ DPO PID: $DPO_PID"

# PPO on GPU 1 (batch_size=1 以减少显存使用)
echo "🚀 启动 PPO 训练 (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python PPO_train.py \
    --epochs 3 \
    --data-limit 500 \
    --lr 1e-6 \
    --batch-size 1 \
    > logs/ppo.log 2>&1 &
PPO_PID=$!
echo "  ✓ PPO PID: $PPO_PID"

# GRPO on GPU 2 (batch_size=1 以减少显存使用)
echo "🚀 启动 GRPO 训练 (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 python GRPO_train.py \
    --epochs 3 \
    --data-limit 500 \
    --lr 1e-6 \
    --batch-size 1 \
    > logs/grpo.log 2>&1 &
GRPO_PID=$!
echo "  ✓ GRPO PID: $GRPO_PID"

# RLOO on GPU 3 (batch_size=1 以减少显存使用)
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
echo "✅ 所有训练任务已启动！"
echo "=========================================="
echo ""
echo "进程ID:"
echo "  DPO:  $DPO_PID"
echo "  PPO:  $PPO_PID"
echo "  GRPO: $GRPO_PID"
echo "  RLOO: $RLOO_PID"
echo ""
echo "📊 监控命令:"
echo "  查看日志: tail -f logs/dpo.log logs/ppo.log logs/grpo.log logs/rloo.log"
echo "  查看GPU:  watch -n 1 nvidia-smi"
echo "  查看进程: ps aux | grep python"
echo ""
echo "🛑 停止训练:"
echo "  kill $DPO_PID $PPO_PID $GRPO_PID $RLOO_PID"
echo ""
echo "📈 TensorBoard (在另一个终端运行):"
echo "  tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard --port 6006"
echo "  tensorboard --logdir /root/Train-RL/outputs/ppo/tensorboard --port 6007"
echo "  tensorboard --logdir /root/Train-RL/outputs/grpo/tensorboard --port 6008"
echo "  tensorboard --logdir /root/Train-RL/outputs/rloo/tensorboard --port 6009"
echo ""

