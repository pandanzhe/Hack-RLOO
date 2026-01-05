# Enhanced RLOO: 推理感知的强化学习优化框架

## 🎯 核心创新点

本项目实现了增强版的RLOO（Reinforcement Learning with Leave-One-Out）算法，专门针对数学推理任务（GSM8K）进行了优化。主要创新点包括：

### 1. 结构化Reasoning Reward（非神经RM）

**创新点**：不使用传统的神经网络Reward Model，而是使用结构化的、可解释的reward组件：

- **AnswerCorrect** (α=1.0): 答案正确性奖励
- **StepCountReward** (β=0.1): 推理步骤数量奖励（鼓励3-6步的合理推理）
- **SymbolCoverage** (γ=0.05): 运算符覆盖奖励（鼓励使用多种数学运算符）
- **ReasoningConsistency** (δ=0.1): 推理一致性奖励（检查中间值是否自洽）

**优势**：
- 无需训练额外的Reward Model，节省时间和算力
- 可解释性强，每个组件都有明确的含义
- 适合比赛场景（72小时内完成）

### 2. 多采样RLOO（Leave-One-Out Baseline）

**创新点**：为每个prompt生成多个推理路径（默认4个），使用leave-one-out方法计算baseline：

```
baseline_i = (total_reward - reward_i) / (n - 1)
advantage_i = reward_i - baseline_i
```

**优势**：
- 利用同一prompt下的推理多样性作为baseline
- 不需要额外的Value Network
- 天然适合推理任务（同一问题可以有多种解法）

### 3. Chain-level Credit Assignment（思维链级别归因）

**创新点**：为推理链的每一步分配reward，而不是只给最终答案：

- 提取推理步骤（基于正则表达式）
- 为每个步骤分配基础reward
- 根据步骤质量调整（引入中间量、避免重复等）

**优势**：
- 让模型学会"如何思考"，而不只是"答案是什么"
- 更细粒度的学习信号
- 提高推理质量

### 4. Prompt-level Value Head（可选）

**创新点**：添加一个轻量级的Value Head，作为global baseline：

- 输入：prompt的embedding
- 输出：expected reward
- 与RLOO baseline融合：`combined_baseline = λ * RLOO_baseline + (1-λ) * Value_baseline`

**优势**：
- 提供跨prompt的长期估计
- 与RLOO baseline互补
- 轻量级（只有两层MLP）

## 📊 算法对比

| 特性 | PPO | DPO | 标准RLOO | Enhanced RLOO |
|------|-----|-----|----------|---------------|
| 需要Value Network | ✅ | ❌ | ❌ | ❌ (可选) |
| 需要Reward Model | ✅ | ❌ | ❌ | ❌ |
| On-policy | ✅ | ❌ | ✅ | ✅ |
| Baseline来源 | 学出来的V(s) | 对比样本 | 同一prompt的其他采样 | 同一prompt的其他采样 + Value Head |
| Reward类型 | 标量 | Pairwise | 标量 | 结构化（多组件） |
| Chain-level归因 | ❌ | ❌ | ❌ | ✅ |
| 适合推理任务 | 中 | 中 | 高 | **很高** |

## 🚀 使用方法

### 1. 训练Enhanced RLOO模型

**基础训练（不使用Value Head和Chain Credit）**：
```bash
python RLOO_enhanced_train.py \
    --epochs 3 \
    --data-limit 500 \
    --lr 1e-7 \
    --batch-size 1 \
    --num-samples 4
```

**完整训练（使用所有创新点）**：
```bash
python RLOO_enhanced_train.py \
    --epochs 3 \
    --data-limit 500 \
    --lr 1e-7 \
    --batch-size 1 \
    --num-samples 4 \
    --use-value-head \
    --use-chain-credit \
    --lambda-baseline 0.7
```

**参数说明**：
- `--epochs`: 训练轮数（推荐3-5）
- `--data-limit`: 训练数据量（推荐500-1000）
- `--lr`: 学习率（推荐1e-7，已自动限制最大值为1e-7）
- `--batch-size`: 批次大小（推荐1，避免OOM）
- `--num-samples`: 每个prompt生成的样本数（推荐4-8）
- `--use-value-head`: 启用Prompt-level Value Head
- `--use-chain-credit`: 启用Chain-level Credit Assignment
- `--lambda-baseline`: RLOO baseline和Value Head的融合权重（0-1，推荐0.7）

### 2. 评估模型

```bash
python evaluate_rloo_enhanced.py
```

评估脚本会：
- 对比Baseline模型和Enhanced RLOO模型
- 计算Accuracy和Pass@1
- 显示改进幅度
- 保存评估结果到JSON文件

### 3. 查看训练曲线

```bash
tensorboard --logdir /root/Train-RL/outputs/rloo_enhanced/tensorboard
```

## 📈 预期效果

基于创新点的设计，预期改进：

- **Accuracy提升**: 5-15%（相比Baseline）
- **推理质量**: 更合理的推理步骤
- **稳定性**: 数值稳定，无梯度爆炸
- **可解释性**: 每个reward组件都有明确含义

## 🔬 技术细节

### Reward计算公式

```
R_total = α·R_answer + β·R_steps + γ·R_symbols + δ·R_consistency + R_format

其中：
- R_answer: 答案正确性（0/0.5/1.0）
- R_steps: 步骤数量奖励（3-6步最优）
- R_symbols: 运算符覆盖（+/−/×/÷）
- R_consistency: 推理一致性（中间值自洽）
- R_format: 格式奖励（包含####）
```

### RLOO Advantage计算

```
对于每个prompt的n个样本：
  baseline_i = (Σ rewards - reward_i) / (n - 1)
  advantage_i = reward_i - baseline_i
```

### Chain-level Credit Assignment

```
对于推理链的每个步骤：
  base_reward = total_reward / num_steps
  step_reward = base_reward * quality_multiplier
  
quality_multiplier:
  - 包含中间量（<<value>>）: ×1.2
  - 重复/空洞步骤: ×0.8
```

## 📝 论文/答辩表述建议

### 英文版本

> **We propose a reasoning-aware RLOO framework that optimizes large language models via prompt-level leave-one-out baselines and structured reward decomposition, enabling stable and interpretable reasoning optimization without value networks.**

### 中文版本

> **我们提出了一个推理感知的RLOO框架，通过prompt级别的leave-one-out baseline和结构化reward分解来优化大语言模型，实现了无需Value Network的稳定且可解释的推理优化。**

### 核心创新点总结

1. **结构化Reward设计**：不使用神经网络RM，而是使用可解释的多组件reward
2. **多采样RLOO**：利用同一prompt的推理多样性作为baseline
3. **思维链归因**：为推理步骤分配reward，学习"如何思考"
4. **轻量级Value Head**：可选的global baseline，与RLOO互补

## 🐛 已知问题和限制

1. **生成速度**：多采样生成可能较慢，可以通过减少`--num-samples`来加速
2. **内存占用**：batch_size=1时内存占用较小，但训练速度较慢
3. **Reward权重**：当前权重是启发式设置的，可能需要根据任务调整

## 🔄 未来改进方向

1. **自适应Reward权重**：根据训练进度动态调整权重
2. **更精细的Chain Credit**：使用attention权重来分配credit
3. **多任务支持**：扩展到其他推理任务（如代码生成、逻辑推理）

## 📚 参考文献

- RLOO: [Reinforcement Learning with Leave-One-Out](相关论文)
- DPO: Direct Preference Optimization
- PPO: Proximal Policy Optimization

## 📧 联系方式

如有问题或建议，请提交Issue或Pull Request。

