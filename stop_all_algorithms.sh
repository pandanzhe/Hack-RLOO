#!/bin/bash
# 停止所有训练任务

echo "=========================================="
echo "停止所有训练任务"
echo "=========================================="
echo ""

# 查找所有训练进程
DPO_PID=$(ps aux | grep "DPO_train.py" | grep -v grep | awk '{print $2}')
PPO_PID=$(ps aux | grep "PPO_train.py" | grep -v grep | awk '{print $2}')
GRPO_PID=$(ps aux | grep "GRPO_train.py" | grep -v grep | awk '{print $2}')
RLOO_PID=$(ps aux | grep "RLOO_train.py" | grep -v grep | awk '{print $2}')

# 停止进程
if [ ! -z "$DPO_PID" ]; then
    echo "🛑 停止 DPO (PID: $DPO_PID)..."
    kill $DPO_PID
    echo "  ✓ DPO 已停止"
else
    echo "  ⚠️  DPO 未运行"
fi

if [ ! -z "$PPO_PID" ]; then
    echo "🛑 停止 PPO (PID: $PPO_PID)..."
    kill $PPO_PID
    echo "  ✓ PPO 已停止"
else
    echo "  ⚠️  PPO 未运行"
fi

if [ ! -z "$GRPO_PID" ]; then
    echo "🛑 停止 GRPO (PID: $GRPO_PID)..."
    kill $GRPO_PID
    echo "  ✓ GRPO 已停止"
else
    echo "  ⚠️  GRPO 未运行"
fi

if [ ! -z "$RLOO_PID" ]; then
    echo "🛑 停止 RLOO (PID: $RLOO_PID)..."
    kill $RLOO_PID
    echo "  ✓ RLOO 已停止"
else
    echo "  ⚠️  RLOO 未运行"
fi

echo ""
echo "=========================================="
echo "✅ 所有训练任务已停止"
echo "=========================================="

