#!/usr/bin/env python3
"""
TensorBoard 启动和管理脚本
提供易用的 TensorBoard 服务器启动方式
"""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = '/root/Train-RL/outputs/dpo'
TB_DIR = Path(OUTPUT_DIR) / 'tensorboard'

def start_tensorboard(port=6006):
    """启动 TensorBoard 服务器"""
    
    if not TB_DIR.exists():
        print(f"❌ TensorBoard 日志目录不存在: {TB_DIR}")
        print("请先运行训练: python DPO_train.py")
        return
    
    print(f"\n📊 启动 TensorBoard")
    print(f"{'='*80}")
    print(f"日志目录: {TB_DIR}")
    print(f"端口: {port}")
    print(f"{'='*80}\n")
    print(f"⏳ 启动中...")
    print(f"🌐 访问地址: http://localhost:{port}\n")
    print(f"💡 提示: 按 Ctrl+C 停止 TensorBoard\n")
    
    try:
        subprocess.run(
            ['tensorboard', '--logdir', str(TB_DIR), '--port', str(port)],
            check=False
        )
    except KeyboardInterrupt:
        print(f"\n✓ TensorBoard 已停止")
    except FileNotFoundError:
        print(f"❌ TensorBoard 未安装")
        print(f"请运行: pip install tensorboard")
        sys.exit(1)

def list_events():
    """列出所有 TensorBoard 事件"""
    
    if not TB_DIR.exists():
        print(f"❌ TensorBoard 日志目录不存在: {TB_DIR}")
        return
    
    print(f"\n📋 TensorBoard 事件文件")
    print(f"{'='*80}\n")
    
    events = list(TB_DIR.glob('events.out.tfevents.*'))
    
    if not events:
        print("❌ 未找到任何事件文件")
        return
    
    for event_file in events:
        size = event_file.stat().st_size / 1024 / 1024  # MB
        print(f"✓ {event_file.name}")
        print(f"  大小: {size:.2f} MB")
        print()
    
    print(f"{'='*80}\n")

def show_tensorboard_info():
    """显示 TensorBoard 信息"""
    
    if not TB_DIR.exists():
        print(f"❌ TensorBoard 日志目录不存在: {TB_DIR}")
        print("请先运行训练: python DPO_train.py --epochs 2 --data-limit 100")
        return
    
    print(f"\n📊 TensorBoard 信息")
    print(f"{'='*80}\n")
    
    print(f"日志目录: {TB_DIR}")
    print(f"目录大小: {sum(f.stat().st_size for f in TB_DIR.rglob('*')) / 1024 / 1024:.2f} MB")
    
    events = list(TB_DIR.glob('events.out.tfevents.*'))
    print(f"事件文件数: {len(events)}")
    
    print(f"\n记录的指标:")
    print(f"  • Loss/batch - 每个 batch 的损失")
    print(f"  • Loss/epoch - 每个 epoch 的平均损失")
    print(f"  • Learning_rate - 学习率")
    
    print(f"\n启动命令:")
    print(f"  tensorboard --logdir {TB_DIR}")
    
    print(f"\n或使用此脚本:")
    print(f"  python tensorboard_launcher.py")
    
    print(f"\n访问地址:")
    print(f"  http://localhost:6006")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorBoard 管理工具')
    parser.add_argument('--port', type=int, default=6006, help='TensorBoard 端口（默认: 6006）')
    parser.add_argument('--info', action='store_true', help='显示 TensorBoard 信息')
    parser.add_argument('--list', action='store_true', help='列出事件文件')
    
    args = parser.parse_args()
    
    if args.info:
        show_tensorboard_info()
    elif args.list:
        list_events()
    else:
        start_tensorboard(port=args.port)
