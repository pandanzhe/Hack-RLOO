#!/usr/bin/env python3
"""
TensorBoard 使用指南
"""

def print_guide():
    guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    TensorBoard 实时监控训练指南                            ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 三种监控方式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方式 1: TensorBoard (推荐) ⭐ - 最可视化
───────────────────────────────────────────────────────────────────────────

1️⃣  启动训练并自动记录:
    python DPO_train.py --epochs 3 --data-limit 500 --lr 1e-5 --beta 1.0

2️⃣  启动 TensorBoard (新终端):
    python tensorboard_launcher.py
    或者:
    tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard

3️⃣  打开浏览器访问:
    http://localhost:6006

4️⃣  查看实时曲线:
    • Scalars: 查看损失和学习率变化
    • 自动刷新显示最新数据


方式 2: 实时文本监控 - 轻量级
───────────────────────────────────────────────────────────────────────────

启动监控 (训练时):
    python monitor_training.py

查看 Epoch 对比:
    python monitor_training.py --compare

自定义更新间隔:
    python monitor_training.py --interval 5


方式 3: 训练后可视化 - 离线分析
───────────────────────────────────────────────────────────────────────────

训练完成后绘制曲线:
    python plot_training_curves.py

输出:
    • training_curves.png - 4 合 1 曲线图
    • 详细的损失统计信息


🚀 快速开始
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

步骤 1: 运行训练 (20 分钟)
────────────────────────────────────────────────────────
cd /root/Train-RL/clean_dir
python DPO_train.py --epochs 3 --data-limit 500 --lr 1e-5 --beta 1.0

输出示例:
  ✓ TensorBoard 日志已保存到: /root/Train-RL/outputs/dpo/tensorboard
  启动 TensorBoard: tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard


步骤 2: 启动 TensorBoard (新终端)
────────────────────────────────────────────────────────
python /root/Train-RL/clean_dir/tensorboard_launcher.py --port 6006

或手动启动:
  tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard --port 6006


步骤 3: 打开浏览器
────────────────────────────────────────────────────────
http://localhost:6006

你会看到:
  • Scalars 标签页 - 实时损失曲线
  • 自动刷新显示最新数据


📈 TensorBoard 界面说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

顶部标签:
  📊 Scalars - 查看标量值（损失、学习率等）✓ 主要用这个
  📊 Distributions - 权重分布变化
  📊 Histograms - 参数直方图
  📊 Graphs - 计算图

Scalars 标签下:
  • Loss/batch - 每个 batch 的损失曲线
  • Loss/epoch - 每个 epoch 的平均损失
  • Learning_rate - 学习率曲线

右侧控制面板:
  • Smoothing - 平滑曲线（拖动滑块）
  • Horizontal Axis - 改变 X 轴（Step/Relative/Wall time）
  • Y-axis scale - Y 轴放大/缩小

💡 使用技巧:
  1. 拖动 "Smoothing" 滑块平滑曲线，更清楚看趋势
  2. 点击图例可隐藏/显示曲线
  3. 可以并列比较多个训练运行
  4. 点击曲线点可查看具体数值


🔄 实时监控工作流程
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

终端 1: 运行训练
  $ python DPO_train.py --epochs 3 --data-limit 500 --lr 1e-5 --beta 1.0
  训练进行中...

终端 2: 启动 TensorBoard
  $ python tensorboard_launcher.py
  ⏳ 启动中...
  🌐 访问地址: http://localhost:6006

浏览器: 实时查看
  1. 打开 http://localhost:6006
  2. 点击 "Scalars" 标签
  3. 看实时更新的曲线

✓ 优势:
  • 实时查看，不影响训练
  • 可视化效果好
  • 自动保存数据


📊 TensorBoard 记录内容
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当前记录的指标:
  
  Loss/batch:
    • 每个 batch 的 DPO 损失
    • 显示训练过程的细粒度变化
    • 通常会有波动但总体下降趋势
  
  Loss/epoch:
    • 每个 epoch 的平均损失
    • 显示每轮训练的整体表现
    • 应该单调递减或保持稳定
  
  Learning_rate:
    • 当前学习率
    • 如果使用学习率调度会显示变化


⚙️  自定义 TensorBoard 参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

改变端口:
  tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard --port 8888
  访问: http://localhost:8888

在网络上公开:
  tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard --host 0.0.0.0
  访问: http://<你的IP>:6006

刷新时间间隔:
  TensorBoard 默认每 30 秒检查一次新数据
  点击刷新按钮手动刷新


🐛 常见问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: TensorBoard 无法连接?
A: 
  1. 检查 TensorBoard 是否启动: ps aux | grep tensorboard
  2. 检查防火墙设置
  3. 尝试其他端口: --port 8888

Q: 数据不更新?
A: 
  1. 检查训练是否仍在运行
  2. 浏览器刷新一下
  3. TensorBoard 默认 30 秒检查一次

Q: 如何对比多个训练运行?
A: 
  1. TensorBoard 会自动检测 logdir 下的所有 run
  2. 运行多个不同参数的训练
  3. 在 TensorBoard 中会并列显示

Q: 存储空间太大?
A: 
  TensorBoard 数据存储在: /root/Train-RL/outputs/dpo/tensorboard
  可以安全删除来释放空间（但会丢失历史数据）


📝 完整工作流示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 终端 1: 开始训练
cd /root/Train-RL/clean_dir
python DPO_train.py --epochs 3 --data-limit 500 --lr 1e-5 --beta 1.0

# 等待训练开始输出，然后...

# 终端 2: 启动 TensorBoard
cd /root/Train-RL/clean_dir
python tensorboard_launcher.py --port 6006

# 浏览器: 访问
http://localhost:6006

# 实时观察:
- 看 Loss/batch 曲线下降
- 看 Loss/epoch 的平均值变化
- 看学习率曲线

# 训练完成后:
python plot_training_curves.py


🔗 文件位置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

训练脚本:
  /root/Train-RL/clean_dir/DPO_train.py

TensorBoard 启动脚本:
  /root/Train-RL/clean_dir/tensorboard_launcher.py

其他监控脚本:
  /root/Train-RL/clean_dir/monitor_training.py (实时文本)
  /root/Train-RL/clean_dir/plot_training_curves.py (离线可视化)

TensorBoard 日志:
  /root/Train-RL/outputs/dpo/tensorboard/


✅ 快速命令速查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 启动训练 (记录到 TensorBoard)
python DPO_train.py --epochs 3 --data-limit 500

# 启动 TensorBoard 服务
python tensorboard_launcher.py

# 或使用 tensorboard 命令
tensorboard --logdir /root/Train-RL/outputs/dpo/tensorboard --port 6006

# 查看信息
python tensorboard_launcher.py --info

# 列出所有日志文件
python tensorboard_launcher.py --list

# 实时文本监控
python monitor_training.py

# 离线可视化
python plot_training_curves.py


---

总结: TensorBoard 是最推荐的方式，提供实时、可视化的训练监控！
"""
    print(guide)

if __name__ == '__main__':
    print_guide()
