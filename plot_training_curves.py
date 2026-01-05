#!/usr/bin/env python3
"""
DPO 训练曲线可视化 - 绘制Loss曲线
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = '/root/Train-RL/outputs/dpo'

def plot_training_curves():
    """绘制训练曲线"""
    
    loss_file = Path(OUTPUT_DIR) / 'training_loss.json'
    
    if not loss_file.exists():
        print(f"❌ 损失数据文件不存在: {loss_file}")
        print("请先运行训练脚本: python DPO_train.py")
        return
    
    # 读取数据
    with open(loss_file, 'r') as f:
        data = json.load(f)
    
    batch_losses = data['batch_losses']
    epoch_losses = data['epoch_losses']
    num_epochs = data['total_epochs']
    
    if not batch_losses:
        print("❌ 没有损失数据")
        return
    
    print(f"\n📊 训练统计")
    print(f"{'='*80}")
    print(f"总 Epochs: {num_epochs}")
    print(f"总 Batches: {len(batch_losses)}")
    print(f"Batches/Epoch: {len(batch_losses) // num_epochs}")
    print(f"初始损失: {batch_losses[0]:.6f}")
    print(f"最终损失: {batch_losses[-1]:.6f}")
    print(f"损失改进: {(batch_losses[0] - batch_losses[-1]):.6f} ({(batch_losses[0] - batch_losses[-1])/batch_losses[0]*100:.2f}%)")
    print(f"\nEpoch 平均损失:")
    for i, loss in enumerate(epoch_losses, 1):
        print(f"  Epoch {i}: {loss:.6f}")
    print(f"{'='*80}\n")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DPO Training Loss Curves', fontsize=16, fontweight='bold')
    
    # 1. Batch损失曲线
    ax = axes[0, 0]
    ax.plot(batch_losses, linewidth=1, alpha=0.8)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Batch Loss Over Time')
    ax.grid(True, alpha=0.3)
    
    # 2. 平滑的损失曲线（移动平均）
    ax = axes[0, 1]
    window_size = max(1, len(batch_losses) // 20)
    smoothed = np.convolve(batch_losses, np.ones(window_size)/window_size, mode='valid')
    ax.plot(smoothed, linewidth=2, color='orange', label='Smoothed (MA)')
    ax.plot(batch_losses, linewidth=0.5, alpha=0.3, label='Raw')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Smoothed Loss (Window={window_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Epoch平均损失
    ax = axes[1, 0]
    ax.bar(range(1, len(epoch_losses)+1), epoch_losses, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_title('Average Loss Per Epoch')
    ax.set_xticks(range(1, len(epoch_losses)+1))
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, loss in enumerate(epoch_losses):
        ax.text(i+1, loss, f'{loss:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Loss分布（直方图）
    ax = axes[1, 1]
    ax.hist(batch_losses, bins=30, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax.axvline(np.mean(batch_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(batch_losses):.4f}')
    ax.axvline(np.median(batch_losses), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(batch_losses):.4f}')
    ax.set_xlabel('Loss Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Loss Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(OUTPUT_DIR) / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 曲线图已保存: {output_path}\n")
    
    # 显示图表
    try:
        plt.show()
    except:
        print("(无法显示图表，但已保存到文件)\n")

def print_loss_statistics():
    """打印损失统计信息"""
    
    loss_file = Path(OUTPUT_DIR) / 'training_loss.json'
    
    if not loss_file.exists():
        return
    
    with open(loss_file, 'r') as f:
        data = json.load(f)
    
    batch_losses = data['batch_losses']
    epoch_losses = data['epoch_losses']
    
    print(f"\n📈 详细统计")
    print(f"{'='*80}")
    print(f"\nBatch Loss 统计:")
    print(f"  最小值: {np.min(batch_losses):.6f}")
    print(f"  最大值: {np.max(batch_losses):.6f}")
    print(f"  平均值: {np.mean(batch_losses):.6f}")
    print(f"  中位数: {np.median(batch_losses):.6f}")
    print(f"  标准差: {np.std(batch_losses):.6f}")
    
    print(f"\nEpoch Loss 统计:")
    print(f"  最小值: {np.min(epoch_losses):.6f}")
    print(f"  最大值: {np.max(epoch_losses):.6f}")
    print(f"  平均值: {np.mean(epoch_losses):.6f}")
    
    # 计算每个epoch的改进
    print(f"\n每个 Epoch 的损失改进:")
    for i in range(len(epoch_losses)):
        if i == 0:
            print(f"  Epoch 1: {epoch_losses[0]:.6f} (基础)")
        else:
            improvement = epoch_losses[i-1] - epoch_losses[i]
            percent = improvement / epoch_losses[i-1] * 100 if epoch_losses[i-1] != 0 else 0
            print(f"  Epoch {i+1}: {epoch_losses[i]:.6f} (改进: {improvement:+.6f}, {percent:+.2f}%)")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    plot_training_curves()
    print_loss_statistics()
