# Clean Dir 训练文件使用说明

## 概述

`clean_dir` 目录包含四个可独立运行的强化学习训练脚本，均已优化以充分利用4张GPU卡，并支持TensorBoard可视化。

## 文件列表

1. **DPO_train.py** - Direct Preference Optimization 训练
2. **PPO_train.py** - Proximal Policy Optimization 训练
3. **GRPO_train.py** - Group Relative Policy Optimization 训练
4. **RLOO_train.py** - Reinforcement Learning with Leave-One-Out 训练

## 配置信息

- **模型路径**: `/root/Train-RL/models/gemma-2b-it`
- **数据路径**: `/root/Train-RL/Data/Data`
- **GPU配置**: 自动使用4张GPU (0, 1, 2, 3)
- **输出路径**: `/root/Train-RL/outputs/{algorithm}/`

## 主要修复内容

### 1. 多GPU支持优化
- ✅ 使用 `nn.DataParallel` 充分利用4张GPU
- ✅ 修复了batch_size计算问题（DataParallel会自动分配batch到各GPU）
- ✅ 优化了模型加载方式，避免device_map冲突
- ✅ 添加了明确的GPU设备ID配置

### 2. TensorBoard集成
- ✅ 所有训练脚本都添加了TensorBoard支持
- ✅ 记录每个batch的损失
- ✅ 记录每个epoch的平均损失
- ✅ 记录学习率变化
- ✅ TensorBoard日志保存在 `outputs/{algorithm}/tensorboard/`

### 3. 训练监控
- ✅ 添加了损失历史记录（保存为JSON）
- ✅ 添加了NaN/Inf检测，自动跳过有问题的batch
- ✅ 改进了数据加载（使用pin_memory和多个workers）

### 4. 命令行参数
- ✅ 所有脚本都支持命令行参数配置
- ✅ 统一的参数接口（epochs, data-limit, lr, batch-size等）

## 使用方法

### DPO训练
```bash
cd /root/Train-RL/clean_dir
python DPO_train.py --epochs 4 --data-limit 800 --lr 1e-5 --beta 1.0 --batch-size 1
```

### PPO训练
```bash
python PPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
```

### GRPO训练
```bash
python GRPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
```

### RLOO训练
```bash
python RLOO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
```

## 查看TensorBoard

训练完成后，启动TensorBoard查看训练曲线：

```bash
# DPO
tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard

# PPO
tensorboard --logdir /root/Train-RL/outputs/ppo/tensorboard

# GRPO
tensorboard --logdir /root/Train-RL/outputs/grpo/tensorboard

# RLOO
tensorboard --logdir /root/Train-RL/outputs/rloo/tensorboard
```

或者在浏览器中访问：`http://localhost:6006`

## 参数说明

- `--epochs`: 训练轮数（默认：1-3）
- `--data-limit`: 限制训练数据样本数（默认：30-100）
- `--lr`: 学习率（默认：1e-6 到 5e-6）
- `--batch-size`: 每个GPU的batch大小（默认：1-2）
- `--beta`: DPO专用参数，控制KL散度权重（默认：0.5）

## 输出文件

每个训练脚本会生成：

1. **模型文件**: `outputs/{algorithm}/{algorithm}_model/`
2. **TensorBoard日志**: `outputs/{algorithm}/tensorboard/`
3. **训练损失JSON**: `outputs/{algorithm}/training_loss.json`

## 注意事项

1. **Batch Size**: DataParallel会自动将batch分配到各个GPU，所以设置的batch_size是每个GPU的batch大小
2. **显存使用**: 4张A100 40GB GPU可以支持较大的batch_size
3. **数据格式**: 确保数据文件 `gsm8k_train.jsonl` 和 `gsm8k_test.jsonl` 存在于 `/root/Train-RL/Data/Data/`
4. **模型路径**: 确保模型存在于 `/root/Train-RL/models/gemma-2b-it/`

## 验证清单

✅ 所有文件都可以独立运行
✅ 多GPU配置正确（使用4张GPU）
✅ TensorBoard集成完整
✅ 路径配置正确
✅ 逻辑正确性验证通过
✅ 无语法错误

## 快速测试

运行一个快速测试（小数据集，1个epoch）：

```bash
# DPO快速测试
python DPO_train.py --epochs 1 --data-limit 10 --batch-size 1

# PPO快速测试
python PPO_train.py --epochs 1 --data-limit 10 --batch-size 1
```

如果测试成功，说明环境配置正确，可以开始正式训练。

