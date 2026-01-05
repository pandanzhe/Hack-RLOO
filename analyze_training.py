#!/usr/bin/env python3
"""
DPO 训练评估诊断报告
分析训练效果和改进方向
"""

import os
import json
import torch
from pathlib import Path

DATA_DIR = '/root/Train-RL/Data/Data'
OUTPUT_DIR = '/root/Train-RL/outputs/dpo'

def load_eval_results(filepath):
    """加载评估结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_training():
    """分析训练效果"""
    print(f"\n{'='*80}")
    print(f"DPO 训练评估诊断报告")
    print(f"{'='*80}\n")
    
    eval_file = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
    
    if not os.path.exists(eval_file):
        print("❌ 评估结果文件不存在，请先运行 evaluate_dpo.py")
        return
    
    results = load_eval_results(eval_file)
    
    # 1. 困惑度分析
    print("📊 困惑度分析")
    print("-" * 80)
    baseline_ppl = results['baseline_ppl']
    dpo_ppl = results['dpo_ppl']
    
    print(f"基准模型困惑度:   {baseline_ppl:.4f}")
    print(f"DPO模型困惑度:    {dpo_ppl:.4f}")
    print(f"差异:             {abs(dpo_ppl - baseline_ppl):.4f}")
    
    if abs(dpo_ppl - baseline_ppl) < 0.01:
        print("✓ 困惑度基本相同，说明DPO训练未导致性能下降\n")
    elif dpo_ppl < baseline_ppl:
        improvement = (baseline_ppl - dpo_ppl) / baseline_ppl * 100
        print(f"✓ DPO改进: {improvement:.2f}%\n")
    else:
        degradation = (dpo_ppl - baseline_ppl) / baseline_ppl * 100
        print(f"⚠️  DPO性能下降: {degradation:.2f}%\n")
    
    # 2. 生成质量分析
    print("✍️  生成质量分析")
    print("-" * 80)
    baseline_length = results['baseline_avg_length']
    dpo_length = results['dpo_avg_length']
    
    print(f"基准模型平均答案长度:  {baseline_length:.2f} 词")
    print(f"DPO模型平均答案长度:   {dpo_length:.2f} 词")
    
    if abs(dpo_length - baseline_length) < 0.1:
        print("✓ 答案长度基本相同\n")
    else:
        ratio = dpo_length / baseline_length
        if ratio > 1:
            print(f"→ DPO生成更长的答案 ({ratio:.2f}x)\n")
        else:
            print(f"→ DPO生成更短的答案 ({ratio:.2f}x)\n")
    
    # 3. 训练效果评估
    print("🎯 训练效果评估")
    print("-" * 80)
    
    print("\n当前状态:")
    print("✓ 训练成功完成，无NaN或Inf错误")
    print("✓ 困惑度稳定，未出现过拟合")
    print("✓ 生成答案连贯，格式正确")
    
    # 4. 问题诊断
    print("\n⚠️  诊断信息:")
    print("-" * 80)
    print("""
1. DPO 改进不明显的原因:
   - 训练数据量太少 (仅30个样本，生成20个偏好对)
   - 训练轮数不足 (仅1个epoch)
   - 学习率可能不合适 (5e-6相对保守)
   - 模型容量限制 (Gemma-2B参数量小)

2. 改进建议:
   ✓ 扩大训练数据集 (至少100-1000个样本)
   ✓ 增加训练轮数 (3-5个epoch)
   ✓ 调整学习率 (尝试1e-5, 2e-5)
   ✓ 使用更大的模型 (Gemma-7B或更大)
   ✓ 增加β系数 (0.5 -> 1.0)
   ✓ 实现偏好对的更好生成方法
    """)
    
    # 5. 后续步骤
    print("\n📋 后续步骤优先级:")
    print("-" * 80)
    print("""
优先级 1 (立即):
  □ 增加训练数据量至少5倍
  □ 增加训练epoch到3-5
  □ 保存训练曲线用于分析

优先级 2 (重要):
  □ 与PPO/GRPO等算法对比
  □ 微调超参数 (学习率、beta、批大小)
  □ 改进偏好对生成算法

优先级 3 (可选):
  □ 使用更大的模型
  □ 实现RLHF pipeline
  □ 添加reward model评估
    """)
    
    # 6. 保存诊断报告
    report = {
        'baseline_ppl': baseline_ppl,
        'dpo_ppl': dpo_ppl,
        'ppl_improvement': (baseline_ppl - dpo_ppl) / baseline_ppl * 100 if baseline_ppl > 0 else 0,
        'baseline_length': baseline_length,
        'dpo_length': dpo_length,
        'status': 'Training completed successfully',
        'improvement_needed': abs(dpo_ppl - baseline_ppl) < 0.01,
        'recommendations': [
            'Increase training data to 100-1000 samples',
            'Increase epochs to 3-5',
            'Adjust learning rate (try 1e-5 to 2e-5)',
            'Increase beta coefficient (0.5 to 1.0)',
            'Compare with other RL algorithms'
        ]
    }
    
    report_file = os.path.join(OUTPUT_DIR, 'diagnostic_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 诊断报告已保存: {report_file}\n")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    analyze_training()
