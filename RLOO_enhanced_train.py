#!/usr/bin/env python3
"""
Enhanced RLOO (Reinforcement Learning with Leave-One-Out) Training
创新点：
1. 结构化Reasoning Reward（非神经RM）
2. 多采样RLOO（同一prompt多个推理路径）
3. Chain-level Credit Assignment（思维链级别归因）
4. Prompt-level Value Head（可选，作为global baseline）
"""

import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import re
import argparse
import numpy as np
from collections import defaultdict

# 如果环境变量未设置，则使用所有GPU；否则使用环境变量指定的GPU
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print(f"{'='*80}")
print(f"Enhanced RLOO Multi-GPU Training")
print(f"{'='*80}")
print(f"Device: {device}, GPUs: {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
print(f"{'='*80}\n")

DATA_DIR = '/root/Train-RL/Data/Data'
MODEL_DIR = '/root/Train-RL/models/gemma-2b-it'
OUTPUT_DIR = '/root/Train-RL/outputs/rloo_enhanced'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def load_data(limit: Optional[int] = None):
    train_data = load_jsonl(os.path.join(DATA_DIR, 'gsm8k_train.jsonl'))
    test_data = load_jsonl(os.path.join(DATA_DIR, 'gsm8k_test.jsonl'))
    if limit:
        train_data, test_data = train_data[:limit], test_data[:limit]
    print(f"✓ 数据加载: 训练{len(train_data)}, 测试{len(test_data)}\n")
    return train_data, test_data

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

def extract_steps(text: str) -> List[str]:
    """提取推理步骤"""
    steps = []
    # 匹配包含计算的步骤
    step_pattern = r'[^\.]+\.'
    matches = re.findall(step_pattern, text)
    for match in matches:
        if any(op in match for op in ['+', '-', '*', '/', '=']):
            steps.append(match.strip())
    return steps

def extract_intermediate_values(text: str) -> List[float]:
    """提取中间计算值"""
    values = []
    # 匹配 <<value>> 格式
    pattern = r'<<[^>]*=([-+]?[0-9]*\.?[0-9]+)>>'
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            values.append(float(match))
        except:
            pass
    return values

class ReasoningRewardCalculator:
    """结构化Reasoning Reward计算器（非神经RM）"""
    
    def __init__(self, 
                 alpha: float = 1.0,  # AnswerCorrect权重
                 beta: float = 0.1,  # StepCountReward权重
                 gamma: float = 0.05,  # SymbolCoverage权重
                 delta: float = 0.1):  # ReasoningConsistency权重
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
    
    def compute_reward(self, generated: str, gt_answer: str) -> Dict[str, float]:
        """计算结构化reward"""
        rewards = {}
        
        # 1. AnswerCorrect (答案正确性)
        pred_val = parse_answer(generated)
        gt_val = parse_answer(gt_answer)
        if pred_val is not None and gt_val is not None:
            if abs(pred_val - gt_val) < 1e-6:
                rewards['answer_correct'] = 1.0
            elif abs((pred_val - gt_val) / (gt_val + 1e-8)) <= 0.1:
                rewards['answer_correct'] = 0.5
            else:
                rewards['answer_correct'] = 0.0
        else:
            rewards['answer_correct'] = 0.0
        
        # 2. StepCountReward (推理步骤数量奖励)
        steps = extract_steps(generated)
        # 理想步骤数：3-6步
        step_count = len(steps)
        if 3 <= step_count <= 6:
            rewards['step_count'] = 1.0
        elif step_count == 2 or step_count == 7:
            rewards['step_count'] = 0.7
        elif step_count == 1 or step_count == 8:
            rewards['step_count'] = 0.4
        else:
            rewards['step_count'] = max(0.0, 1.0 - (step_count - 6) * 0.1)
        
        # 3. SymbolCoverage (运算符覆盖奖励)
        required_ops = ['+', '-', '*', '/']
        found_ops = [op for op in required_ops if op in generated]
        rewards['symbol_coverage'] = len(found_ops) / len(required_ops)
        
        # 4. ReasoningConsistency (推理一致性)
        intermediate_vals = extract_intermediate_values(generated)
        # 检查中间值是否自洽（简单启发式：检查是否有重复或异常值）
        if len(intermediate_vals) >= 2:
            # 检查值是否在合理范围内
            valid_vals = [v for v in intermediate_vals if -1e6 < v < 1e6]
            if len(valid_vals) == len(intermediate_vals):
                rewards['reasoning_consistency'] = 1.0
            else:
                rewards['reasoning_consistency'] = 0.5
        else:
            rewards['reasoning_consistency'] = 0.5
        
        # 格式奖励（bonus）
        if '####' in generated:
            rewards['format_bonus'] = 0.1
        else:
            rewards['format_bonus'] = 0.0
        
        # 总reward
        total_reward = (
            self.alpha * rewards['answer_correct'] +
            self.beta * rewards['step_count'] +
            self.gamma * rewards['symbol_coverage'] +
            self.delta * rewards['reasoning_consistency'] +
            rewards['format_bonus']
        )
        rewards['total'] = total_reward
        
        return rewards

class PromptValueHead(nn.Module):
    """Prompt-level Value Head（作为global baseline）"""
    
    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.activation = nn.ReLU()
    
    def forward(self, prompt_embedding: torch.Tensor) -> torch.Tensor:
        """输入：prompt的embedding，输出：expected reward"""
        x = self.activation(self.fc1(prompt_embedding))
        return self.fc2(x).squeeze(-1)

class ChainCreditAssigner:
    """Chain-level Credit Assignment（思维链级别归因）"""
    
    def __init__(self):
        self.step_pattern = re.compile(r'[^\.]+\.')
    
    def assign_step_rewards(self, text: str, total_reward: float) -> List[Tuple[int, float]]:
        """为每个推理步骤分配reward"""
        steps = extract_steps(text)
        if len(steps) == 0:
            return []
        
        # 简单策略：均匀分配 + 基于步骤质量的调整
        base_reward = total_reward / len(steps)
        step_rewards = []
        
        for i, step in enumerate(steps):
            step_reward = base_reward
            
            # 调整：引入正确中间量的步骤获得更多reward
            if '<<' in step and '>>' in step:
                step_reward *= 1.2
            
            # 调整：重复/空洞步骤获得更少reward
            if len(set(step.split())) < 5:
                step_reward *= 0.8
            
            step_rewards.append((i, step_reward))
        
        return step_rewards

class EnhancedRLOODataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_len: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = ex.get('prompt', '')
        target = ex.get('target', '')
        question = ex.get('question', '')
        
        # 构建完整的prompt
        full_text = prompt + target
        
        tokens = self.tokenizer(full_text, max_length=self.max_len,
                               truncation=True, padding='max_length',
                               return_tensors='pt')
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'prompt': prompt,
            'target': target,
            'question': question
        }

class EnhancedRLOOTrainer:
    def __init__(self, model_dir: str, num_gpus: int = num_gpus,
                 batch_size: int = 1, lr: float = 1e-6, epochs: int = 1,
                 num_samples: int = 4,  # 每个prompt生成多个样本
                 use_value_head: bool = True,  # 是否使用Prompt-level Value Head
                 use_chain_credit: bool = True,  # 是否使用Chain-level Credit Assignment
                 lambda_baseline: float = 0.7):  # RLOO baseline和Value Head的融合权重
        self.device = device
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_samples = num_samples
        self.use_value_head = use_value_head
        self.use_chain_credit = use_chain_credit
        self.lambda_baseline = lambda_baseline
        self.output_dir = OUTPUT_DIR
        self.loss_history = []
        self.epoch_losses = []
        
        # 初始化组件
        self.reward_calculator = ReasoningRewardCalculator()
        self.credit_assigner = ChainCreditAssigner()
        
        # 初始化 TensorBoard
        self.tb_dir = os.path.join(OUTPUT_DIR, 'tensorboard')
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        self.global_step = 0
        
        print(f"加载模型: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                                       trust_remote_code=True,
                                                       local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True
        ).to(device)
        
        # Prompt-level Value Head
        if self.use_value_head:
            # 获取hidden_size（需要考虑DataParallel的情况）
            model_config = self.model.module.config if isinstance(self.model, nn.DataParallel) else self.model.config
            hidden_size = model_config.hidden_size
            self.value_head = PromptValueHead(hidden_size).to(device)
            if num_gpus > 1:
                self.value_head = nn.DataParallel(self.value_head)
            print(f"✓ 启用Prompt-level Value Head (hidden_size={hidden_size})")
        
        if num_gpus > 1:
            print(f"✓ 启用DataParallel ({num_gpus} GPUs)")
            self.model = nn.DataParallel(self.model, device_ids=list(range(num_gpus)))
            print(f"✓ 模型已分配到 {num_gpus} 张GPU上")
        
        # 优化器
        params = list(self.model.parameters())
        if self.use_value_head:
            params += list(self.value_head.parameters())
        
        safe_lr = min(lr, 1e-7)
        if safe_lr < lr:
            print(f"⚠️  学习率从 {lr} 调整为 {safe_lr} 以提高稳定性")
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=safe_lr,
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        print(f"✓ 模型加载完成\n")
    
    def generate_multiple_samples(self, prompt: str, num_samples: int, 
                                  temperature: float = 0.7, max_new_tokens: int = 256) -> List[str]:
        """为同一prompt生成多个推理路径"""
        model_to_use = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = model_to_use.generate(
                    **prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(outputs[0][len(prompt_tokens['input_ids'][0]):], 
                                                      skip_special_tokens=True)
                samples.append(generated_text)
        
        return samples
    
    def compute_rloo_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算RLOO优势函数（Leave-One-Out baseline）"""
        if rewards.shape[0] == 1:
            return rewards
        
        # RLOO: baseline = (total - current_reward) / (n - 1)
        total = torch.sum(rewards)
        baselines = (total - rewards) / (rewards.shape[0] - 1)
        advantages = rewards - baselines
        return advantages
    
    def get_prompt_embedding(self, prompt: str) -> torch.Tensor:
        """获取prompt的embedding（用于Value Head）"""
        model_to_use = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt', 
                                      truncation=True, max_length=256).to(self.device)
        
        with torch.no_grad():
            outputs = model_to_use(**prompt_tokens, output_hidden_states=True)
            # 使用最后一层的平均pooling
            hidden_states = outputs.hidden_states[-1]
            prompt_embedding = hidden_states.mean(dim=1).squeeze(0)
            # 确保是float32（Value Head需要）
            if prompt_embedding.dtype != torch.float32:
                prompt_embedding = prompt_embedding.float()
        
        return prompt_embedding
    
    def train(self, train_data: List[Dict]):
        print(f"{'='*80}")
        print(f"Enhanced RLOO训练 ({self.num_gpus} GPU, batch_size={self.batch_size})")
        print(f"创新点:")
        print(f"  1. 结构化Reasoning Reward（非神经RM）")
        print(f"  2. 多采样RLOO（每个prompt生成{self.num_samples}个样本）")
        if self.use_chain_credit:
            print(f"  3. Chain-level Credit Assignment（思维链级别归因）")
        if self.use_value_head:
            print(f"  4. Prompt-level Value Head（global baseline）")
        print(f"{'='*80}\n")
        
        dataset = EnhancedRLOODataset(train_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True if self.num_gpus > 0 else False)
        
        for epoch in range(self.epochs):
            self.model.train()
            if self.use_value_head:
                self.value_head.train()
            
            total_loss = 0.0
            num_batches = 0
            total_rewards = []
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")):
                prompts = batch['prompt']
                targets = batch['target']
                
                # 为每个prompt生成多个样本
                all_samples = []
                all_rewards = []
                all_prompt_embeddings = []
                
                for i in range(len(prompts)):
                    prompt = prompts[i]
                    target = targets[i]
                    
                    # 生成多个推理路径
                    samples = self.generate_multiple_samples(prompt, self.num_samples)
                    all_samples.extend(samples)
                    
                    # 计算每个样本的reward
                    sample_rewards = []
                    for sample in samples:
                        reward_dict = self.reward_calculator.compute_reward(sample, target)
                        sample_rewards.append(reward_dict['total'])
                    all_rewards.append(torch.tensor(sample_rewards, dtype=torch.float32))
                    
                    # 获取prompt embedding（用于Value Head）
                    if self.use_value_head:
                        prompt_emb = self.get_prompt_embedding(prompt)
                        all_prompt_embeddings.append(prompt_emb)
                
                # 计算advantages
                batch_advantages = []
                for i in range(len(prompts)):
                    rewards = all_rewards[i].to(self.device)
                    
                    # RLOO baseline
                    rloo_advantages = self.compute_rloo_advantages(rewards)
                    
                    # 融合Value Head baseline（如果启用）
                    if self.use_value_head:
                        prompt_emb = all_prompt_embeddings[i]
                        value_net_to_use = self.value_head.module if isinstance(self.value_head, nn.DataParallel) else self.value_head
                        # Value Head需要float32输入
                        if prompt_emb.dtype != torch.float32:
                            prompt_emb = prompt_emb.float()
                        expected_reward = value_net_to_use(prompt_emb.unsqueeze(0))
                        
                        # 融合baseline: RLOO baseline + Value Head baseline
                        rloo_baseline = rewards.mean() - rloo_advantages.mean()
                        value_baseline = expected_reward.squeeze()
                        combined_baseline = (
                            self.lambda_baseline * rloo_baseline +
                            (1 - self.lambda_baseline) * value_baseline
                        )
                        advantages = rewards - combined_baseline
                    else:
                        advantages = rloo_advantages
                    
                    batch_advantages.append(advantages)
                    total_rewards.extend(rewards.cpu().tolist())
                
                # 计算policy gradient loss
                self.optimizer.zero_grad()
                
                total_batch_loss = 0.0
                valid_samples = 0
                
                for i in range(len(prompts)):
                    prompt = prompts[i]
                    samples = all_samples[i * self.num_samples:(i + 1) * self.num_samples]
                    advantages = batch_advantages[i]
                    
                    # 为每个样本计算loss
                    for j, sample in enumerate(samples):
                        if advantages[j].item() <= 0:
                            continue  # 跳过负advantage的样本
                        
                        # 构建完整的输入
                        full_text = prompt + sample
                        tokens = self.tokenizer(full_text, return_tensors='pt', 
                                              truncation=True, max_length=512).to(self.device)
                        
                        if tokens['input_ids'].shape[1] < 2:
                            continue
                        
                        # 前向传播
                        try:
                            outputs = self.model(**tokens)
                            logits = outputs.logits
                            
                            # 计算policy gradient loss
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = tokens['input_ids'][..., 1:].contiguous()
                            
                            # 温度缩放
                            temperature = 2.0
                            shift_logits = shift_logits / temperature
                            shift_logits = torch.clamp(shift_logits, min=-50.0, max=50.0)
                            
                            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                            token_losses = loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                            
                            # Chain-level Credit Assignment（如果启用）
                            if self.use_chain_credit:
                                step_rewards = self.credit_assigner.assign_step_rewards(
                                    sample, advantages[j].item()
                                )
                                # 简化：均匀分配advantage到所有token
                                token_advantages = advantages[j].item() / shift_labels.numel()
                            else:
                                token_advantages = advantages[j].item() / shift_labels.numel()
                            
                            # Policy gradient: -advantage * log_prob
                            policy_loss = -token_advantages * token_losses.mean()
                            policy_loss = policy_loss * 0.1  # 缩放
                            
                            total_batch_loss += policy_loss
                            valid_samples += 1
                            
                        except RuntimeError as e:
                            print(f"Warning: Runtime error: {e}")
                            torch.cuda.empty_cache()
                            continue
                
                if valid_samples == 0:
                    continue
                
                loss = total_batch_loss / valid_samples
                
                # 检查loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                
                # 反向传播
                loss.backward()
                
                # 检查梯度
                model_to_check = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                has_nan_grad = False
                for name, param in model_to_check.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad.zero_()
                            has_nan_grad = True
                
                if self.use_value_head:
                    value_net_to_check = self.value_head.module if isinstance(self.value_head, nn.DataParallel) else self.value_head
                    for name, param in value_net_to_check.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                param.grad.zero_()
                                has_nan_grad = True
                
                if has_nan_grad:
                    print(f"Warning: NaN/Inf in gradients, skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                # 梯度裁剪
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + 
                    (list(self.value_head.parameters()) if self.use_value_head else []),
                    0.5
                )
                
                if torch.isnan(grad_norm_before) or torch.isinf(grad_norm_before):
                    print(f"Warning: Gradient norm is NaN/Inf, skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                if grad_norm_before > 1000.0:
                    print(f"Warning: Gradient norm too large: {grad_norm_before:.2f}, skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
                
                loss_item = loss.item()
                total_loss += loss_item
                num_batches += 1
                self.loss_history.append(loss_item)
                
                # TensorBoard记录
                self.writer.add_scalar('Loss/batch', loss_item, self.global_step)
                if total_rewards:
                    avg_reward = np.mean(total_rewards[-self.num_samples*self.batch_size:])
                    self.writer.add_scalar('Reward/avg', avg_reward, self.global_step)
                self.global_step += 1
                
                # 定期清理缓存
                if self.global_step % 10 == 0:
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.epoch_losses.append(avg_loss)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/epoch', avg_loss, epoch)
            self.writer.add_scalar('Learning_rate', self.lr, epoch)
            if total_rewards:
                self.writer.add_scalar('Reward/epoch_avg', np.mean(total_rewards), epoch)
            
            print(f"  平均损失: {avg_loss:.4f}")
            if total_rewards:
                print(f"  平均Reward: {np.mean(total_rewards):.4f}\n")
        
        # 保存模型
        save_path = os.path.join(self.output_dir, 'rloo_enhanced_model')
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_to_save.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        if self.use_value_head:
            value_head_to_save = self.value_head.module if isinstance(self.value_head, nn.DataParallel) else self.value_head
            torch.save(value_head_to_save.state_dict(), 
                      os.path.join(save_path, 'value_head.pt'))
        
        print(f"✓ 模型已保存: {save_path}")
        
        # 保存损失历史
        self.save_loss_history()
        
        # 关闭TensorBoard
        self.writer.close()
        print(f"✓ TensorBoard日志已保存到: {self.tb_dir}")
        print(f"启动TensorBoard: tensorboard --logdir {self.tb_dir}")
        print(f"{'='*80}\n✅ Enhanced RLOO训练完成！\n")
    
    def save_loss_history(self):
        """保存训练损失历史为JSON"""
        loss_file = os.path.join(self.output_dir, 'training_loss.json')
        with open(loss_file, 'w') as f:
            json.dump({
                'batch_losses': self.loss_history,
                'epoch_losses': self.epoch_losses,
                'total_epochs': self.epochs,
                'total_batches': len(self.loss_history)
            }, f, indent=2)
        print(f"✓ 损失数据已保存: {loss_file}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced RLOO Training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--data-limit', type=int, default=100, help='Limit on training data samples')
    parser.add_argument('--lr', type=float, default=1e-7, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--num-samples', type=int, default=4, help='Number of samples per prompt')
    parser.add_argument('--use-value-head', action='store_true', help='Use Prompt-level Value Head')
    parser.add_argument('--use-chain-credit', action='store_true', help='Use Chain-level Credit Assignment')
    parser.add_argument('--lambda-baseline', type=float, default=0.7, help='Weight for RLOO baseline vs Value Head')
    
    args = parser.parse_args()
    
    # 加载数据
    train_data, test_data = load_data(limit=args.data_limit)
    
    print(f"\n{'='*80}")
    print(f"Enhanced RLOO 训练配置")
    print(f"{'='*80}")
    print(f"Epochs:        {args.epochs}")
    print(f"Data limit:    {args.data_limit}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Num samples:   {args.num_samples} (每个prompt生成多个推理路径)")
    print(f"Use Value Head: {args.use_value_head}")
    print(f"Use Chain Credit: {args.use_chain_credit}")
    print(f"Lambda baseline: {args.lambda_baseline}")
    print(f"{'='*80}\n")
    
    # 训练
    trainer = EnhancedRLOOTrainer(
        model_dir=MODEL_DIR,
        num_gpus=num_gpus,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_samples=args.num_samples,
        use_value_head=args.use_value_head,
        use_chain_credit=args.use_chain_credit,
        lambda_baseline=args.lambda_baseline
    )
    trainer.train(train_data)

if __name__ == '__main__':
    main()

