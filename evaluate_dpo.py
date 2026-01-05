#!/usr/bin/env python3
"""
DPO 模型评估脚本
评估指标：困惑度、生成质量、性能对比
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import json
import torch
import torch.nn as nn
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print(f"{'='*80}")
print(f"DPO 模型评估")
print(f"{'='*80}")
print(f"Device: {device}, GPUs: {num_gpus}\n")

# 路径配置
DATA_DIR = '/root/Train-RL/Data/Data'
MODEL_DIR = '/root/Train-RL/models/gemma-2b-it'
DPO_MODEL_DIR = '/root/Train-RL/outputs/dpo/dpo_model'
BASELINE_MODEL_DIR = MODEL_DIR

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL数据"""
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
    return data

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
        BASELINE_MODEL_DIR,
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

def compute_perplexity(model, tokenizer, texts: List[str]) -> float:
    """计算困惑度（Perplexity）"""
    total_loss = 0.0
    total_tokens = 0
    
    # 处理 DataParallel 包装
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            outputs = actual_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item() * input_ids.shape[1]
                total_tokens += input_ids.shape[1]
    
    if total_tokens == 0:
        return float('inf')
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity

def generate_answer(model, tokenizer, prompt: str, max_length: int = 256) -> str:
    """生成答案"""
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    
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
    
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.replace(prompt, '').strip()

def evaluate_on_test_set(tokenizer, baseline_model, dpo_model, test_data: List[Dict], limit: int = 10):
    """在测试集上评估"""
    print(f"{'='*80}")
    print(f"测试集评估 (前 {limit} 个样本)")
    print(f"{'='*80}\n")
    
    test_data = test_data[:limit]
    
    # 1. 困惑度评估
    print("1️⃣  困惑度评估")
    print("-" * 80)
    
    test_prompts = [f"数学问题: {ex.get('question', '')}\n答案: " for ex in test_data]
    
    baseline_ppl = compute_perplexity(baseline_model, tokenizer, test_prompts)
    dpo_ppl = compute_perplexity(dpo_model, tokenizer, test_prompts)
    
    print(f"基准模型困惑度:  {baseline_ppl:.4f}")
    print(f"DPO模型困惑度:   {dpo_ppl:.4f}")
    
    if dpo_ppl < baseline_ppl:
        improvement = (baseline_ppl - dpo_ppl) / baseline_ppl * 100
        print(f"✓ DPO 改进: {improvement:.2f}%\n")
    else:
        print(f"✗ DPO 未改进\n")
    
    # 2. 生成质量对比
    print("2️⃣  生成答案对比")
    print("-" * 80)
    
    results = []
    for i, ex in enumerate(test_data[:5]):  # 只显示前5个
        question = ex.get('question', '')
        gold_answer = ex.get('answer', '')
        prompt = f"数学问题: {question}\n答案: "
        
        print(f"\n[样本 {i+1}]")
        print(f"问题: {question[:80]}...")
        print(f"标准答案: {gold_answer[:100]}...")
        
        baseline_answer = generate_answer(baseline_model, tokenizer, prompt)
        dpo_answer = generate_answer(dpo_model, tokenizer, prompt)
        
        print(f"基准生成: {baseline_answer[:100]}...")
        print(f"DPO生成:  {dpo_answer[:100]}...")
        
        results.append({
            'question': question,
            'gold_answer': gold_answer,
            'baseline_answer': baseline_answer,
            'dpo_answer': dpo_answer
        })
    
    # 3. 答案长度对比
    print(f"\n\n3️⃣  答案统计")
    print("-" * 80)
    
    baseline_lengths = []
    dpo_lengths = []
    
    for ex in test_data:
        question = ex.get('question', '')
        prompt = f"数学问题: {question}\n答案: "
        
        baseline_answer = generate_answer(baseline_model, tokenizer, prompt)
        dpo_answer = generate_answer(dpo_model, tokenizer, prompt)
        
        baseline_lengths.append(len(baseline_answer.split()))
        dpo_lengths.append(len(dpo_answer.split()))
    
    print(f"基准模型平均答案长度: {np.mean(baseline_lengths):.2f} 词")
    print(f"DPO模型平均答案长度:  {np.mean(dpo_lengths):.2f} 词")
    print(f"标准差:")
    print(f"  基准: {np.std(baseline_lengths):.2f}")
    print(f"  DPO:  {np.std(dpo_lengths):.2f}")
    
    return {
        'baseline_ppl': baseline_ppl,
        'dpo_ppl': dpo_ppl,
        'baseline_avg_length': np.mean(baseline_lengths),
        'dpo_avg_length': np.mean(dpo_lengths),
        'examples': results
    }

def print_summary(eval_results: Dict):
    """打印评估总结"""
    print(f"\n{'='*80}")
    print(f"评估总结")
    print(f"{'='*80}\n")
    
    baseline_ppl = eval_results['baseline_ppl']
    dpo_ppl = eval_results['dpo_ppl']
    
    print(f"困惑度对比:")
    print(f"  基准模型: {baseline_ppl:.4f}")
    print(f"  DPO模型:  {dpo_ppl:.4f}")
    
    if dpo_ppl < baseline_ppl:
        improvement = (baseline_ppl - dpo_ppl) / baseline_ppl * 100
        print(f"  ✓ 改进: {improvement:.2f}%")
    else:
        degradation = (dpo_ppl - baseline_ppl) / baseline_ppl * 100
        print(f"  ✗ 下降: {degradation:.2f}%")
    
    print(f"\n答案长度对比:")
    print(f"  基准模型: {eval_results['baseline_avg_length']:.2f} 词")
    print(f"  DPO模型:  {eval_results['dpo_avg_length']:.2f} 词")
    
    length_ratio = eval_results['dpo_avg_length'] / eval_results['baseline_avg_length']
    print(f"  比率: {length_ratio:.2f}x")
    
    print(f"\n{'='*80}\n")

def main():
    # 加载数据
    test_data = load_jsonl(os.path.join(DATA_DIR, 'gsm8k_test.jsonl'))
    print(f"✓ 加载测试数据: {len(test_data)} 个样本\n")
    
    # 加载模型
    tokenizer, baseline_model, dpo_model = load_models()
    
    # 评估
    eval_results = evaluate_on_test_set(tokenizer, baseline_model, dpo_model, test_data, limit=10)
    
    # 打印总结
    print_summary(eval_results)
    
    # 保存结果
    output_file = '/root/Train-RL/outputs/dpo/evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        # 转换为可序列化的格式
        results_to_save = {
            'baseline_ppl': float(eval_results['baseline_ppl']),
            'dpo_ppl': float(eval_results['dpo_ppl']),
            'baseline_avg_length': float(eval_results['baseline_avg_length']),
            'dpo_avg_length': float(eval_results['dpo_avg_length']),
            'examples': eval_results['examples']
        }
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 评估结果已保存: {output_file}\n")

if __name__ == '__main__':
    main()
