#!/usr/bin/env python3
"""
Enhanced RLOO评估脚本
对比Baseline模型和Enhanced RLOO模型在GSM8K测试集上的表现
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import numpy as np
from typing import List, Dict, Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = '/root/Train-RL/Data/Data'
MODEL_DIR = '/root/Train-RL/models/gemma-2b-it'
ENHANCED_MODEL_DIR = '/root/Train-RL/outputs/rloo_enhanced/rloo_enhanced_model'
OUTPUT_DIR = '/root/Train-RL/outputs/rloo_enhanced'

def load_jsonl(file_path: str) -> List[Dict]:
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

def parse_answer(text: str) -> Optional[float]:
    """解析答案中的数值"""
    if not text:
        return None
    try:
        match = re.search(r'####\s*([-+]?[0-9]*\.?[0-9]+)', text)
        if match:
            return float(match.group(1))
        for line in reversed(text.split('\n')):
            nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line)
            if nums:
                return float(nums[-1])
        return None
    except:
        return None

def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 256, 
                   temperature: float = 0.7) -> str:
    """生成答案"""
    model_to_use = model.module if hasattr(model, 'module') else model
    
    prompt_tokens = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model_to_use.generate(
            **prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0][len(prompt_tokens['input_ids'][0]):], 
                                         skip_special_tokens=True)
    return generated_text

def evaluate_model(model, tokenizer, test_data: List[Dict], num_samples: int = 1) -> Dict:
    """评估模型"""
    correct = 0
    total = 0
    all_results = []
    
    for item in tqdm(test_data, desc="评估中"):
        prompt = item.get('prompt', '')
        target = item.get('target', '')
        question = item.get('question', '')
        
        gt_answer = parse_answer(target)
        if gt_answer is None:
            continue
        
        # 生成多个答案（用于pass@k评估）
        generated_answers = []
        for _ in range(num_samples):
            generated = generate_answer(model, tokenizer, prompt)
            generated_answers.append(generated)
        
        # 检查是否有正确答案
        found_correct = False
        for gen_answer in generated_answers:
            pred_answer = parse_answer(gen_answer)
            if pred_answer is not None and abs(pred_answer - gt_answer) < 1e-6:
                found_correct = True
                break
        
        if found_correct:
            correct += 1
        total += 1
        
        all_results.append({
            'question': question,
            'target': target,
            'generated': generated_answers[0],
            'correct': found_correct,
            'gt_answer': gt_answer,
            'pred_answer': parse_answer(generated_answers[0])
        })
    
    accuracy = correct / total if total > 0 else 0.0
    pass_at_k = accuracy  # 简化：pass@1
    
    return {
        'accuracy': accuracy,
        'pass_at_1': pass_at_k,
        'correct': correct,
        'total': total,
        'results': all_results
    }

def main():
    print(f"{'='*80}")
    print(f"Enhanced RLOO 模型评估")
    print(f"{'='*80}\n")
    
    # 加载测试数据
    test_data = load_jsonl(os.path.join(DATA_DIR, 'gsm8k_test.jsonl'))
    # 限制测试数量以便快速评估
    test_data = test_data[:100]
    print(f"✓ 测试数据: {len(test_data)} 条\n")
    
    results = {}
    
    # 评估Baseline模型
    print("1. 评估Baseline模型...")
    baseline_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    if baseline_tokenizer.pad_token is None:
        baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
    
    baseline_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True
    ).to(device)
    baseline_model.eval()
    
    baseline_results = evaluate_model(baseline_model, baseline_tokenizer, test_data)
    results['baseline'] = baseline_results
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  Pass@1: {baseline_results['pass_at_1']:.4f}")
    print(f"  Correct: {baseline_results['correct']}/{baseline_results['total']}\n")
    
    # 评估Enhanced RLOO模型
    if os.path.exists(ENHANCED_MODEL_DIR):
        print("2. 评估Enhanced RLOO模型...")
        enhanced_tokenizer = AutoTokenizer.from_pretrained(ENHANCED_MODEL_DIR, trust_remote_code=True, local_files_only=True)
        if enhanced_tokenizer.pad_token is None:
            enhanced_tokenizer.pad_token = enhanced_tokenizer.eos_token
        
        enhanced_model = AutoModelForCausalLM.from_pretrained(
            ENHANCED_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True
        ).to(device)
        enhanced_model.eval()
        
        enhanced_results = evaluate_model(enhanced_model, enhanced_tokenizer, test_data)
        results['enhanced_rloo'] = enhanced_results
        print(f"  Accuracy: {enhanced_results['accuracy']:.4f}")
        print(f"  Pass@1: {enhanced_results['pass_at_1']:.4f}")
        print(f"  Correct: {enhanced_results['correct']}/{enhanced_results['total']}\n")
        
        # 计算改进
        improvement = enhanced_results['accuracy'] - baseline_results['accuracy']
        improvement_pct = (improvement / baseline_results['accuracy'] * 100) if baseline_results['accuracy'] > 0 else 0
        print(f"{'='*80}")
        print(f"改进结果:")
        print(f"  Baseline Accuracy: {baseline_results['accuracy']:.4f}")
        print(f"  Enhanced RLOO Accuracy: {enhanced_results['accuracy']:.4f}")
        print(f"  绝对改进: {improvement:+.4f}")
        print(f"  相对改进: {improvement_pct:+.2f}%")
        print(f"{'='*80}\n")
    else:
        print("⚠️  Enhanced RLOO模型不存在，请先训练模型")
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
    
    # 只保存统计信息，不保存所有结果（避免文件过大）
    summary = {
        'baseline': {
            'accuracy': results['baseline']['accuracy'],
            'pass_at_1': results['baseline']['pass_at_1'],
            'correct': results['baseline']['correct'],
            'total': results['baseline']['total']
        }
    }
    
    if 'enhanced_rloo' in results:
        summary['enhanced_rloo'] = {
            'accuracy': results['enhanced_rloo']['accuracy'],
            'pass_at_1': results['enhanced_rloo']['pass_at_1'],
            'correct': results['enhanced_rloo']['correct'],
            'total': results['enhanced_rloo']['total']
        }
        summary['improvement'] = {
            'absolute': enhanced_results['accuracy'] - baseline_results['accuracy'],
            'relative_pct': improvement_pct
        }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ 评估结果已保存: {results_file}")
    
    # 显示一些示例
    if 'enhanced_rloo' in results:
        print(f"\n示例结果（前5个）:")
        for i, result in enumerate(results['enhanced_rloo']['results'][:5]):
            print(f"\n问题 {i+1}: {result['question'][:60]}...")
            print(f"  正确答案: {result['gt_answer']}")
            print(f"  预测答案: {result['pred_answer']}")
            print(f"  是否正确: {'✓' if result['correct'] else '✗'}")
            print(f"  生成文本: {result['generated'][:100]}...")

if __name__ == '__main__':
    main()

