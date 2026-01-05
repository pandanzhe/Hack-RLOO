#!/usr/bin/env python3
"""
快速测试Enhanced RLOO的关键组件
"""

import torch
from RLOO_enhanced_train import (
    ReasoningRewardCalculator,
    ChainCreditAssigner,
    PromptValueHead
)

def test_reward_calculator():
    """测试Reward计算器"""
    print("=" * 80)
    print("测试: ReasoningRewardCalculator")
    print("=" * 80)
    
    calculator = ReasoningRewardCalculator()
    
    # 测试案例1：正确答案
    generated1 = "Step 1: 48/2 = <<48/2=24>>24\nStep 2: 48+24 = <<48+24=72>>72\n#### 72"
    target1 = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"
    
    reward1 = calculator.compute_reward(generated1, target1)
    print(f"\n测试案例1（正确答案）:")
    print(f"  AnswerCorrect: {reward1['answer_correct']}")
    print(f"  StepCount: {reward1['step_count']}")
    print(f"  SymbolCoverage: {reward1['symbol_coverage']}")
    print(f"  ReasoningConsistency: {reward1['reasoning_consistency']}")
    print(f"  Total Reward: {reward1['total']}")
    
    # 测试案例2：错误答案
    generated2 = "Step 1: 48+24 = 70\n#### 70"
    reward2 = calculator.compute_reward(generated2, target1)
    print(f"\n测试案例2（错误答案）:")
    print(f"  AnswerCorrect: {reward2['answer_correct']}")
    print(f"  Total Reward: {reward2['total']}")
    
    print("\n✓ Reward计算器测试通过\n")

def test_chain_credit_assigner():
    """测试Chain Credit Assigner"""
    print("=" * 80)
    print("测试: ChainCreditAssigner")
    print("=" * 80)
    
    assigner = ChainCreditAssigner()
    
    text = "Step 1: 48/2 = <<48/2=24>>24\nStep 2: 48+24 = <<48+24=72>>72\n#### 72"
    total_reward = 1.5
    
    step_rewards = assigner.assign_step_rewards(text, total_reward)
    print(f"\n推理文本: {text[:50]}...")
    print(f"总Reward: {total_reward}")
    print(f"步骤Reward分配:")
    for i, (step_idx, reward) in enumerate(step_rewards):
        print(f"  步骤 {step_idx+1}: {reward:.4f}")
    
    print("\n✓ Chain Credit Assigner测试通过\n")

def test_value_head():
    """测试Value Head"""
    print("=" * 80)
    print("测试: PromptValueHead")
    print("=" * 80)
    
    hidden_size = 2048
    value_head = PromptValueHead(hidden_size)
    
    # 创建模拟的prompt embedding
    batch_size = 2
    prompt_embedding = torch.randn(batch_size, hidden_size)
    
    expected_reward = value_head(prompt_embedding)
    print(f"\n输入形状: {prompt_embedding.shape}")
    print(f"输出形状: {expected_reward.shape}")
    print(f"输出值: {expected_reward.tolist()}")
    
    print("\n✓ Value Head测试通过\n")

def test_rloo_advantage():
    """测试RLOO Advantage计算"""
    print("=" * 80)
    print("测试: RLOO Advantage计算")
    print("=" * 80)
    
    # 模拟4个样本的rewards
    rewards = torch.tensor([1.5, 0.8, 1.2, 0.5])
    
    # RLOO baseline计算
    total = torch.sum(rewards)
    baselines = (total - rewards) / (rewards.shape[0] - 1)
    advantages = rewards - baselines
    
    print(f"\nRewards: {rewards.tolist()}")
    print(f"Baselines: {baselines.tolist()}")
    print(f"Advantages: {advantages.tolist()}")
    print(f"Advantages总和: {advantages.sum().item():.4f} (应该接近0)")
    
    print("\n✓ RLOO Advantage计算测试通过\n")

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Enhanced RLOO 组件测试")
    print("=" * 80 + "\n")
    
    try:
        test_reward_calculator()
        test_chain_credit_assigner()
        test_value_head()
        test_rloo_advantage()
        
        print("=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

