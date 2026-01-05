#!/usr/bin/env python3
"""
实时训练监控脚本
在训练进行中实时显示损失曲线和进度
"""

import json
import time
import sys
from pathlib import Path
from collections import deque

OUTPUT_DIR = '/root/Train-RL/outputs/dpo'

def clear_screen():
    """清除屏幕"""
    print('\033[2J\033[H', end='')

def format_loss_bar(loss, width=20):
    """创建损失条形图"""
    # 标准化到 0-1 范围，使用 sigmoid 函数
    normalized = min(loss / 1.0, 1.0)  # 假设最大损失为 1.0
    filled = int(normalized * width)
    bar = '█' * filled + '░' * (width - filled)
    return bar

def real_time_monitor(update_interval=2, history_size=10):
    """实时监控训练进度"""
    
    loss_file = Path(OUTPUT_DIR) / 'training_loss.json'
    
    print("🔄 实时训练监控")
    print(f"{'='*80}")
    print(f"损失文件: {loss_file}")
    print(f"更新间隔: {update_interval} 秒")
    print(f"显示历史: 最近 {history_size} 个样本")
    print(f"{'='*80}\n")
    print("按 Ctrl+C 停止监控\n")
    
    last_batch_count = 0
    loss_history = deque(maxlen=history_size)
    
    try:
        while True:
            if loss_file.exists():
                with open(loss_file, 'r') as f:
                    data = json.load(f)
                
                batch_losses = data['batch_losses']
                epoch_losses = data['epoch_losses']
                num_epochs = data['total_epochs']
                
                # 清除屏幕
                clear_screen()
                
                print("🔄 实时训练监控")
                print(f"{'='*80}\n")
                
                # 基本统计
                print(f"📊 训练进度:")
                print(f"  总 Epochs: {num_epochs}")
                print(f"  当前 Batch: {len(batch_losses)}")
                print(f"  完成 Epochs: {len(epoch_losses)}")
                print(f"  当前 Epoch Batch: {len(batch_losses) % max(1, len(batch_losses) // len(epoch_losses)) if epoch_losses else len(batch_losses)}\n")
                
                # 当前损失
                if batch_losses:
                    latest_loss = batch_losses[-1]
                    print(f"📉 当前损失: {latest_loss:.6f}")
                    print(f"   {format_loss_bar(latest_loss)}\n")
                    
                    # 最近的损失变化
                    if len(batch_losses) > 1:
                        prev_loss = batch_losses[-2]
                        change = latest_loss - prev_loss
                        direction = "↓ 改进" if change < 0 else "↑ 增加"
                        print(f"   变化: {change:+.6f} {direction}\n")
                
                # Epoch统计
                if epoch_losses:
                    print(f"📈 Epoch 平均损失:")
                    for i, loss in enumerate(epoch_losses, 1):
                        print(f"   Epoch {i}: {loss:.6f} {format_loss_bar(loss, 15)}")
                    print()
                
                # 损失范围
                if batch_losses:
                    print(f"📊 损失统计:")
                    print(f"   最小: {min(batch_losses):.6f}")
                    print(f"   最大: {max(batch_losses):.6f}")
                    print(f"   平均: {sum(batch_losses)/len(batch_losses):.6f}\n")
                
                # 检查新 batches
                if len(batch_losses) > last_batch_count:
                    new_batches = len(batch_losses) - last_batch_count
                    loss_history.extend(batch_losses[-new_batches:])
                    last_batch_count = len(batch_losses)
                
                # 最近的损失历史
                if loss_history:
                    print(f"📝 最近 {len(loss_history)} 个 Batch 损失:")
                    for i, loss in enumerate(loss_history, 1):
                        idx = len(batch_losses) - len(loss_history) + i
                        print(f"   [{idx:4d}] {loss:.6f} {format_loss_bar(loss, 12)}")
                
                print(f"\n{'='*80}")
                print(f"⏱️  下次更新: {update_interval} 秒后")
                print(f"💡 提示: 按 Ctrl+C 停止监控")
                
                time.sleep(update_interval)
            else:
                print(f"⏳ 等待训练开始...")
                time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n\n✓ 监控已停止")
        if loss_file.exists():
            with open(loss_file, 'r') as f:
                data = json.load(f)
            print(f"\n最终统计:")
            print(f"  总 Batches: {len(data['batch_losses'])}")
            print(f"  总 Epochs: {len(data['epoch_losses'])}")
            print(f"  最终损失: {data['batch_losses'][-1]:.6f}")

def compare_epochs():
    """对比不同 Epochs 的损失"""
    
    loss_file = Path(OUTPUT_DIR) / 'training_loss.json'
    
    if not loss_file.exists():
        print("❌ 损失数据文件不存在")
        return
    
    with open(loss_file, 'r') as f:
        data = json.load(f)
    
    batch_losses = data['batch_losses']
    epoch_losses = data['epoch_losses']
    num_epochs = len(epoch_losses)
    
    if num_epochs < 2:
        print("⚠️  需要至少 2 个 Epochs 来对比")
        return
    
    print(f"\n📊 Epoch 对比分析")
    print(f"{'='*80}\n")
    
    batches_per_epoch = len(batch_losses) // num_epochs
    
    for epoch_id in range(num_epochs):
        start_idx = epoch_id * batches_per_epoch
        end_idx = (epoch_id + 1) * batches_per_epoch if epoch_id < num_epochs - 1 else len(batch_losses)
        
        epoch_batch_losses = batch_losses[start_idx:end_idx]
        
        if epoch_batch_losses:
            min_loss = min(epoch_batch_losses)
            max_loss = max(epoch_batch_losses)
            avg_loss = sum(epoch_batch_losses) / len(epoch_batch_losses)
            
            print(f"Epoch {epoch_id + 1}:")
            print(f"  Batches: {len(epoch_batch_losses)}")
            print(f"  平均损失: {avg_loss:.6f}")
            print(f"  最小损失: {min_loss:.6f}")
            print(f"  最大损失: {max_loss:.6f}")
            
            # 对比上一个 Epoch
            if epoch_id > 0:
                prev_avg = epoch_losses[epoch_id - 1]
                curr_avg = epoch_losses[epoch_id]
                improvement = prev_avg - curr_avg
                percent = improvement / prev_avg * 100 if prev_avg != 0 else 0
                print(f"  vs Epoch {epoch_id}: {improvement:+.6f} ({percent:+.2f}%)")
            
            print()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='实时训练监控')
    parser.add_argument('--interval', type=int, default=2, help='更新间隔（秒）')
    parser.add_argument('--history', type=int, default=10, help='显示的历史大小')
    parser.add_argument('--compare', action='store_true', help='只显示 Epoch 对比')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_epochs()
    else:
        real_time_monitor(update_interval=args.interval, history_size=args.history)
