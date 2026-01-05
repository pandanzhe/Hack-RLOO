#!/usr/bin/env python3
"""
DPO 模型推理测试 - 与基准模型对比
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print(f"\n{'='*80}")
print(f"DPO 模型推理测试")
print(f"{'='*80}\n")

# 路径配置
MODEL_DIR = '/root/Train-RL/models/gemma-2b-it'
DPO_MODEL_DIR = '/root/Train-RL/outputs/dpo/dpo_model'

# 测试问题集
TEST_QUESTIONS = [
    "What is 2+2?",
    "If John has 5 apples and Mary has 3 apples, how many do they have in total?",
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and uses eight in her bakery each day. How many eggs are left over at the end of the day?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does the robe take?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many meters does James run per week?",
]

def load_models():
    """加载基准模型和DPO模型"""
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR,
                                              trust_remote_code=True,
                                              local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 基准模型
    baseline_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        device_map='cuda:0' if num_gpus > 0 else None,
    )
    
    # DPO模型
    dpo_model = AutoModelForCausalLM.from_pretrained(
        DPO_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        device_map='cuda:0' if num_gpus > 0 else None,
    )
    
    # 多卡包装
    if num_gpus > 1:
        baseline_model = nn.DataParallel(baseline_model)
        dpo_model = nn.DataParallel(dpo_model)
    
    baseline_model.eval()
    dpo_model.eval()
    
    print(f"✓ 模型加载完成\n")
    return tokenizer, baseline_model, dpo_model

def generate_with_timing(model, tokenizer, prompt: str, max_length: int = 200):
    """生成答案并记录时间"""
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    
    start_time = time.time()
    
    # 处理 DataParallel 包装
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    with torch.no_grad():
        output_ids = actual_model.generate(
            input_ids,
            max_length=max_length,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )
    
    inference_time = time.time() - start_time
    
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = answer.replace(prompt, '').strip()
    
    return answer, inference_time, output_ids[0].shape[0]

def test_models(tokenizer, baseline_model, dpo_model):
    """测试模型性能"""
    print(f"{'='*80}")
    print(f"推理性能对比 (5个测试样本)")
    print(f"{'='*80}\n")
    
    baseline_times = []
    dpo_times = []
    baseline_tokens = []
    dpo_tokens = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        prompt = f"Question: {question}\nAnswer: "
        
        print(f"[问题 {i}] {question[:60]}...")
        print("-" * 80)
        
        # 基准模型推理
        baseline_answer, baseline_time, baseline_token_count = generate_with_timing(
            baseline_model, tokenizer, prompt
        )
        baseline_times.append(baseline_time)
        baseline_tokens.append(baseline_token_count)
        
        # DPO模型推理
        dpo_answer, dpo_time, dpo_token_count = generate_with_timing(
            dpo_model, tokenizer, prompt
        )
        dpo_times.append(dpo_time)
        dpo_tokens.append(dpo_token_count)
        
        # 显示对比
        print(f"基准模型 ({baseline_time:.3f}s):")
        print(f"  {baseline_answer[:100]}...")
        print(f"\nDPO模型 ({dpo_time:.3f}s):")
        print(f"  {dpo_answer[:100]}...")
        print()
    
    # 统计信息
    print(f"\n{'='*80}")
    print(f"推理统计")
    print(f"{'='*80}\n")
    
    import numpy as np
    
    print(f"推理时间 (秒):")
    print(f"  基准模型: {np.mean(baseline_times):.4f}±{np.std(baseline_times):.4f}")
    print(f"  DPO模型:  {np.mean(dpo_times):.4f}±{np.std(dpo_times):.4f}")
    
    speedup = np.mean(baseline_times) / np.mean(dpo_times)
    print(f"  加速比:   {speedup:.2f}x")
    
    print(f"\n生成tokens数:")
    print(f"  基准模型: {np.mean(baseline_tokens):.1f}±{np.std(baseline_tokens):.1f}")
    print(f"  DPO模型:  {np.mean(dpo_tokens):.1f}±{np.std(dpo_tokens):.1f}")
    
    print(f"\n{'='*80}\n")

def main():
    tokenizer, baseline_model, dpo_model = load_models()
    test_models(tokenizer, baseline_model, dpo_model)

if __name__ == '__main__':
    main()
