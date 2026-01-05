#!/bin/bash
# 监控后台训练进度

echo "════════════════════════════════════════════════════════════════"
echo "  DPO 后台训练监控 (进程 PID: 22382)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查进程
echo -e "${BLUE}[进程状态]${NC}"
ps aux | grep "DPO_train.py" | grep -v grep || echo "❌ 训练进程未运行"
echo ""

# 查看训练日志最后20行
echo -e "${BLUE}[最新日志]${NC}"
tail -20 /root/Train-RL/clean_dir/training.log
echo ""

# 计数 Epoch 和 Batch
echo -e "${BLUE}[进度统计]${NC}"
EPOCHS=$(grep -c "Epoch" /root/Train-RL/clean_dir/training.log || echo "0")
BATCHES=$(grep -c "Batch" /root/Train-RL/clean_dir/training.log || echo "0")
echo "✓ Epochs: $EPOCHS"
echo "✓ Batches: $BATCHES"
echo ""

# 显示损失值
echo -e "${BLUE}[损失统计]${NC}"
grep "Loss:" /root/Train-RL/clean_dir/training.log | tail -5 || echo "还未开始记录损失"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}💡 实时查看日志:${NC} tail -f /root/Train-RL/clean_dir/training.log"
echo -e "${GREEN}💡 启动 TensorBoard:${NC} tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard${NC}"
echo "════════════════════════════════════════════════════════════════"
