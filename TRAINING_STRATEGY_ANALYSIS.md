# 四算法训练策略分析 - A100 40GB

## 📊 显存需求分析

### 单卡显存占用（Gemma-2B-IT）
```
模型权重:        ~4 GB  (BF16/FP16)
优化器状态:      ~2 GB  (AdamW)
梯度:            ~1 GB
激活值:          ~2 GB  (batch_size=2)
批处理数据:      ~1 GB
其他开销:        ~1 GB
-------------------
总计:            ~11 GB / 40 GB (27.5%利用率)
```

**结论**: A100 40GB 显存充足，单卡完全可以运行一个算法。

---

## 🎯 方案对比

### 方案1: 四张卡分别跑四个实验（同时跑）⭐ **推荐**

**配置**:
```bash
# 终端1 - GPU 0
CUDA_VISIBLE_DEVICES=0 python DPO_train.py --epochs 3 --data-limit 500 --lr 1e-5 --batch-size 2

# 终端2 - GPU 1  
CUDA_VISIBLE_DEVICES=1 python PPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2

# 终端3 - GPU 2
CUDA_VISIBLE_DEVICES=2 python GRPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2

# 终端4 - GPU 3
CUDA_VISIBLE_DEVICES=3 python RLOO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
```

**优势**:
- ✅ **总时间最短**: 4个算法同时跑，总时间 = 单个算法时间
- ✅ **算法对比完整**: 可以同时对比4种算法的训练过程
- ✅ **风险分散**: 一个算法失败不影响其他
- ✅ **显存利用率高**: 4张卡都充分利用（每张约11GB/40GB）
- ✅ **配置简单**: 无需修改代码，直接设置环境变量
- ✅ **科学价值高**: 完整的算法对比实验

**劣势**:
- ⚠️ 每个算法训练速度较慢（单卡）
- ⚠️ 需要开4个终端或使用进程管理

**预计时间**:
- 单个算法: 约 2-3 小时
- **总时间: 约 2-3 小时**（并行）

---

### 方案2: 四张卡跑一个实验，然后依次跑完所有实验

**配置**:
```bash
# 实验1: DPO (使用4张卡)
python DPO_train.py --epochs 3 --data-limit 500 --lr 1e-5 --batch-size 2
# 等待完成...

# 实验2: PPO (使用4张卡)
python PPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
# 等待完成...

# 实验3: GRPO (使用4张卡)
python GRPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
# 等待完成...

# 实验4: RLOO (使用4张卡)
python RLOO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
```

**优势**:
- ✅ 每个算法训练速度快（4倍加速，DataParallel）
- ✅ 单次实验显存利用率高（4张卡并行）

**劣势**:
- ❌ **总时间最长**: 需要依次跑4次，总时间 = 单个算法时间 × 4
- ❌ 无法同时对比算法
- ❌ 显存浪费（4张卡只跑一个算法，每张卡只用11GB）
- ❌ 如果某个算法失败，需要重新开始

**预计时间**:
- 单个算法（4卡并行）: 约 0.5-1 小时
- **总时间: 约 2-4 小时**（串行）

---

## 🏆 推荐方案：方案1（四张卡分别跑四个实验）

### 理由

1. **时间效率最高**
   - 方案1: 2-3小时完成所有实验
   - 方案2: 2-4小时完成所有实验
   - **节省 33-50% 时间**

2. **显存利用率合理**
   - 方案1: 4张卡 × 11GB = 44GB 总使用（每张卡27.5%）
   - 方案2: 4张卡 × 11GB = 44GB 总使用（但串行，利用率低）
   - **方案1显存利用率更高**

3. **科学价值**
   - 方案1: 可以同时观察4个算法的训练曲线对比
   - 方案2: 只能依次查看，无法实时对比
   - **方案1更适合算法研究**

4. **风险控制**
   - 方案1: 一个失败不影响其他
   - 方案2: 如果第4个失败，前面3个已经完成，但无法同时对比

---

## 🚀 实施方案1的脚本

### 方法1: 使用4个终端（最简单）

打开4个终端，分别运行：

```bash
# 终端1
cd /root/Train-RL/clean_dir
CUDA_VISIBLE_DEVICES=0 python DPO_train.py --epochs 3 --data-limit 500 --lr 1e-5 --beta 1.0 --batch-size 2

# 终端2
cd /root/Train-RL/clean_dir
CUDA_VISIBLE_DEVICES=1 python PPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2

# 终端3
cd /root/Train-RL/clean_dir
CUDA_VISIBLE_DEVICES=2 python GRPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2

# 终端4
cd /root/Train-RL/clean_dir
CUDA_VISIBLE_DEVICES=3 python RLOO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
```

### 方法2: 使用后台进程（推荐）

创建启动脚本 `run_all_algorithms.sh`:

```bash
#!/bin/bash
cd /root/Train-RL/clean_dir

# DPO on GPU 0
CUDA_VISIBLE_DEVICES=0 python DPO_train.py --epochs 3 --data-limit 500 --lr 1e-5 --beta 1.0 --batch-size 2 > dpo.log 2>&1 &

# PPO on GPU 1
CUDA_VISIBLE_DEVICES=1 python PPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2 > ppo.log 2>&1 &

# GRPO on GPU 2
CUDA_VISIBLE_DEVICES=2 python GRPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2 > grpo.log 2>&1 &

# RLOO on GPU 3
CUDA_VISIBLE_DEVICES=3 python RLOO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2 > rloo.log 2>&1 &

echo "所有训练任务已启动！"
echo "查看日志: tail -f dpo.log ppo.log grpo.log rloo.log"
echo "查看GPU使用: watch -n 1 nvidia-smi"
```

运行:
```bash
chmod +x run_all_algorithms.sh
./run_all_algorithms.sh
```

### 方法3: 使用 screen/tmux（适合远程服务器）

```bash
# 创建4个screen会话
screen -S dpo
CUDA_VISIBLE_DEVICES=0 python DPO_train.py --epochs 3 --data-limit 500 --lr 1e-5 --beta 1.0 --batch-size 2
# Ctrl+A, D 退出

screen -S ppo
CUDA_VISIBLE_DEVICES=1 python PPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
# Ctrl+A, D 退出

screen -S grpo
CUDA_VISIBLE_DEVICES=2 python GRPO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
# Ctrl+A, D 退出

screen -S rloo
CUDA_VISIBLE_DEVICES=3 python RLOO_train.py --epochs 3 --data-limit 500 --lr 1e-6 --batch-size 2
# Ctrl+A, D 退出

# 查看所有会话
screen -ls

# 重新连接
screen -r dpo
```

---

## 📈 监控训练进度

### 查看GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看TensorBoard（4个算法同时）
```bash
# 终端1: DPO
tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard --port 6006

# 终端2: PPO
tensorboard --logdir /root/Train-RL/outputs/ppo/tensorboard --port 6007

# 终端3: GRPO
tensorboard --logdir /root/Train-RL/outputs/grpo/tensorboard --port 6008

# 终端4: RLOO
tensorboard --logdir /root/Train-RL/outputs/rloo/tensorboard --port 6009
```

然后在浏览器中打开:
- DPO: http://localhost:6006
- PPO: http://localhost:6007
- GRPO: http://localhost:6008
- RLOO: http://localhost:6009

---

## ✅ 总结

**推荐方案**: **方案1 - 四张卡分别跑四个实验（同时跑）**

**核心优势**:
1. ⏱️ 总时间最短（2-3小时 vs 2-4小时）
2. 📊 可以同时对比4个算法
3. 💾 显存利用率合理（每张卡27.5%）
4. 🛡️ 风险分散，一个失败不影响其他
5. 🔬 科学价值高，完整的算法对比

**适用场景**: 
- ✅ 需要对比多个算法
- ✅ 显存充足（A100 40GB）
- ✅ 时间有限，需要快速完成所有实验

