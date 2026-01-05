#!/usr/bin/env python3
"""
多RL算法对比评估 - DPO vs PPO vs GRPO vs RLOO
对多个推理任务进行对比测试
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print(f"\n{'='*80}")
print(f"多RL算法对比评估")
print(f"{'='*80}\n")

# 模型路径配置
MODEL_DIR = '/root/Train-RL/models/gemma-2b-it'
BASELINE_MODEL_DIR = MODEL_DIR
OUTPUT_BASE = '/root/Train-RL/outputs'

# 算法模型路径
MODELS = {
    'Baseline': BASELINE_MODEL_DIR,
    'DPO': os.path.join(OUTPUT_BASE, 'dpo/dpo_model'),
}

# 测试问题
TEST_QUESTIONS = [
    "What is 2+2?",
    "If John has 5 apples and Mary has 3 apples, how many do they have in total?",
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and uses eight in her bakery each day. How many eggs are left over at the end of the day?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does the robe take?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many meters does James run per week?",
]

class ModelManager:
    """管理多个模型的加载和推理"""
    
    def __init__(self):
        self.tokenizer = None
        self.models = {}
        self.load_tokenizer()
    
    def load_tokenizer(self):
        """加载分词器"""
        print("加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✓ 分词器加载完成\n")
    
    def load_model(self, name: str, model_dir: str):
        """加载单个模型"""
        print(f"加载 {name} 模型...")
        
        if not os.path.exists(model_dir):
            print(f"  ⚠️  模型路径不存在: {model_dir}")
            return False
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                trust_remote_code=True,
                device_map='cuda:0' if num_gpus > 0 else None,
            )
            
            if num_gpus > 1:
                model = nn.DataParallel(model)
            
            model.eval()
            self.models[name] = model
            print(f"✓ {name} 加载完成")
            return True
        except Exception as e:
            print(f"✗ {name} 加载失败: {str(e)}")
            return False
    
    def load_all_models(self):
        """加载所有模型"""
        print(f"{'='*80}")
        print(f"加载所有模型")
        print(f"{'='*80}\n")
        
        loaded_count = 0
        for name, model_dir in MODELS.items():
            if self.load_model(name, model_dir):
                loaded_count += 1
        
        print(f"\n✓ 成功加载 {loaded_count}/{len(MODELS)} 个模型\n")
        return loaded_count > 0
    
    def generate(self, model_name: str, prompt: str, max_length: int = 200) -> tuple:
        """生成答案"""
        if model_name not in self.models:
            return None, None, 0
        
        model = self.models[model_name]
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        
        start_time = time.time()
        
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
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = answer.replace(prompt, '').strip()
        
        return answer, inference_time, output_ids[0].shape[0]

def run_comparison(manager: ModelManager):
    """运行对比评估"""
    print(f"{'='*80}")
    print(f"对比测试 ({len(TEST_QUESTIONS)} 个问题)")
    print(f"{'='*80}\n")
    
    results = {name: {'times': [], 'tokens': [], 'answers': []} for name in manager.models.keys()}
    
    for q_idx, question in enumerate(TEST_QUESTIONS, 1):
        prompt = f"Question: {question}\nAnswer: "
        
        print(f"\n[问题 {q_idx}/{len(TEST_QUESTIONS)}] {question[:70]}...")
        print("-" * 80)
        
        for model_name in manager.models.keys():
            answer, inference_time, token_count = manager.generate(model_name, prompt)
            
            if answer is not None:
                results[model_name]['times'].append(inference_time)
                results[model_name]['tokens'].append(token_count)
                results[model_name]['answers'].append(answer)
                
                print(f"\n{model_name:12} ({inference_time:.3f}s, {token_count:3d}t):")
                print(f"  {answer[:100]}...")
            else:
                print(f"\n{model_name:12} - 生成失败")
    
    return results

def print_comparison_summary(results: dict):
    """打印对比总结"""
    print(f"\n\n{'='*80}")
    print(f"性能对比总结")
    print(f"{'='*80}\n")
    
    # 推理时间对比
    print("📊 推理时间对比 (秒):")
    print("-" * 80)
    
    model_names = list(results.keys())
    times_stats = {}
    
    for model_name, data in results.items():
        if data['times']:
            mean_time = np.mean(data['times'])
            std_time = np.std(data['times'])
            times_stats[model_name] = mean_time
            print(f"{model_name:12}: {mean_time:.4f}±{std_time:.4f}s")
    
    # 计算加速比
    if len(times_stats) > 1:
        baseline_time = times_stats[model_names[0]]
        print("\n加速比 (相对于基准):")
        for model_name in model_names[1:]:
            speedup = baseline_time / times_stats[model_name]
            print(f"{model_name:12}: {speedup:.2f}x")
    
    # Token 统计
    print(f"\n\n📈 生成Token数对比:")
    print("-" * 80)
    
    for model_name, data in results.items():
        if data['tokens']:
            mean_tokens = np.mean(data['tokens'])
            std_tokens = np.std(data['tokens'])
            print(f"{model_name:12}: {mean_tokens:.1f}±{std_tokens:.1f}")
    
    # 答案示例对比
    print(f"\n\n📝 答案示例对比 (第1个问题):")
    print("-" * 80)
    
    question = TEST_QUESTIONS[0]
    print(f"问题: {question}\n")
    
    for model_name, data in results.items():
        if data['answers']:
            print(f"{model_name}:")
            print(f"  {data['answers'][0][:120]}...")
            print()
    
    print(f"{'='*80}\n")
    
    # 保存详细结果
    output_file = os.path.join(OUTPUT_BASE, 'algorithm_comparison.json')
    
    # 转换为可序列化的格式
    results_to_save = {}
    for model_name, data in results.items():
        results_to_save[model_name] = {
            'mean_inference_time': float(np.mean(data['times'])) if data['times'] else 0,
            'std_inference_time': float(np.std(data['times'])) if data['times'] else 0,
            'mean_token_count': float(np.mean(data['tokens'])) if data['tokens'] else 0,
            'num_tests': len(data['times']),
            'sample_answers': data['answers'][:2]  # 保存前2个答案示例
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 详细结果已保存: {output_file}\n")

def main():
    manager = ModelManager()
    
    if not manager.load_all_models():
        print("✗ 至少需要加载一个模型")
        return
    
    results = run_comparison(manager)
    print_comparison_summary(results)

if __name__ == '__main__':
    main()
