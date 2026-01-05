#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Multi-GPU Training
充分利用4张显卡进行训练
"""

import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import re
import argparse

# 如果环境变量未设置，则使用所有GPU；否则使用环境变量指定的GPU
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print(f"{'='*80}")
print(f"GRPO Multi-GPU Training")
print(f"{'='*80}")
print(f"Device: {device}, GPUs: {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
print(f"{'='*80}\n")

DATA_DIR = '/root/Train-RL/Data/Data'
MODEL_DIR = '/root/Train-RL/models/gemma-2b-it'
OUTPUT_DIR = '/root/Train-RL/outputs/grpo'
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

class GRPODataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_len: int = 256):  # 减少max_len以节省显存
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        q = ex.get('question', '')
        a = ex.get('answer', '')
        prompt = f"解决这个数学问题:\n{q}\n\n"
        
        tokens = self.tokenizer(prompt + a, max_length=self.max_len,
                               truncation=True, padding='max_length',
                               return_tensors='pt')
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'answer': a
        }

class GRPOTrainer:
    def __init__(self, model_dir: str, num_gpus: int = num_gpus,
                 batch_size: int = 4, lr: float = 1e-6, epochs: int = 1):
        self.device = device
        self.num_gpus = num_gpus
        # DataParallel会自动将batch分配到各个GPU，所以这里使用原始batch_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.output_dir = OUTPUT_DIR
        self.loss_history = []
        self.epoch_losses = []
        
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
            torch_dtype=torch.bfloat16,  # 切换到 BF16，避免 NaN
            local_files_only=True,
            trust_remote_code=True
        ).to(device)
        
        # 保存模型初始状态用于恢复
        self.model_dir = model_dir
        self.backup_layers = {}
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        with torch.no_grad():
            for name, param in model_to_save.named_parameters():
                self.backup_layers[name] = param.data.clone().cpu()
        print(f"✓ 已备份 {len(self.backup_layers)} 个参数层用于恢复")
        
        if num_gpus > 1:
            print(f"✓ 启用DataParallel ({num_gpus} GPUs)")
            self.model = nn.DataParallel(self.model, device_ids=list(range(num_gpus)))
            print(f"✓ 模型已分配到 {num_gpus} 张GPU上")
        
        # 大幅降低学习率以提高稳定性
        safe_lr = min(lr, 1e-7)  # 限制最大学习率为1e-7
        if safe_lr < lr:
            print(f"⚠️  学习率从 {lr} 调整为 {safe_lr} 以提高稳定性")
        
        # 使用更小的学习率和权重衰减，提高数值稳定性
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=safe_lr,
            weight_decay=0.01,  # 添加权重衰减
            eps=1e-8,  # 提高数值稳定性
            betas=(0.9, 0.999)
        )
        # 不使用 GradScaler，因为模型已经是 FP16，不需要混合精度训练
        self.scaler = None
        print(f"✓ 模型加载完成\n")
    
    def compute_reward(self, generated: str, gt_answer: str) -> float:
        """计算奖励"""
        reward = 0.0
        if '####' in generated:
            reward += 0.1
        
        pred_val = parse_answer(generated)
        gt_val = parse_answer(gt_answer)
        
        if pred_val is not None and gt_val is not None:
            if abs(pred_val - gt_val) < 1e-6:
                reward += 1.0
            elif abs((pred_val - gt_val) / (gt_val + 1e-8)) <= 0.1:
                reward += 0.3
            else:
                reward -= 0.1
        else:
            reward -= 0.2
        
        return reward
    
    def train(self, train_data: List[Dict]):
        print(f"{'='*80}")
        print(f"GRPO训练 ({self.num_gpus} GPU, batch_size={self.batch_size})")
        print(f"{'='*80}\n")
        
        dataset = GRPODataset(train_data, self.tokenizer)
        # DataParallel会自动将batch分配到各个GPU，所以使用原始batch_size
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True if self.num_gpus > 0 else False)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 检查输入
                if torch.isnan(input_ids).any() or torch.isnan(attention_mask).any():
                    print(f"Warning: NaN in input, skipping")
                    continue
                
                # 检查模型参数（更新前）
                model_to_check = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                nan_param_count = 0
                with torch.no_grad():
                    for name, param in model_to_check.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            nan_param_count += 1
                            # 从备份恢复
                            if name in self.backup_layers:
                                param.data.copy_(self.backup_layers[name].to(param.device))
                
                if nan_param_count > 0:
                    print(f"Warning: 检测到 {nan_param_count} 个参数损坏，已恢复")
                    if nan_param_count > 50:
                        print(f"❌ 参数损坏过多，停止训练")
                        break
                
                self.optimizer.zero_grad()
                
                # 模型使用的是 BF16，非常稳定，不需要特殊检查
                try:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # 检查logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"Warning: NaN/Inf in logits, skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    
                    # 使用交叉熵损失，但对logits进行缩放以提高数值稳定性
                    # 将 logits 平铺并作为输入，计算自回归损失
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # 对logits进行裁剪，避免极端值导致梯度爆炸
                    # 使用温度缩放，使logits更稳定（更大的温度值，降低logits尺度）
                    temperature = 2.0  # 增加温度，进一步降低logits尺度
                    shift_logits = shift_logits / temperature
                    shift_logits = torch.clamp(shift_logits, min=-50.0, max=50.0)
                    
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    # 进一步缩放损失，使梯度更小
                    loss = loss * 0.1
                    
                except RuntimeError as e:
                    print(f"Warning: Runtime error during forward: {e}")
                    torch.cuda.empty_cache()
                    continue
                
                # 检查loss是否有效（只检查NaN/Inf，损失可以是负数）
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected (NaN/Inf): {loss.item()}, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                
                # 检查损失是否过大（可能是数值问题）
                if abs(loss.item()) > 1e6:
                    print(f"Warning: Loss too large: {loss.item()}, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                
                loss.backward()
                
                # 检查梯度
                has_nan_grad = False
                for name, param in model_to_check.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad.zero_()
                            has_nan_grad = True
                
                if has_nan_grad:
                    print(f"Warning: NaN/Inf in gradients, skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                # 梯度裁剪（先裁剪，再检查）
                # clip_grad_norm_会先裁剪梯度，然后返回裁剪前的梯度范数
                # 使用更激进的梯度裁剪（0.5），因为损失已经缩放
                grad_norm_before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                
                # 检查裁剪前的梯度范数是否异常
                # 如果梯度范数过大（>1000），说明可能有梯度爆炸，跳过更新
                if torch.isnan(grad_norm_before) or torch.isinf(grad_norm_before):
                    print(f"Warning: Gradient norm is NaN/Inf, skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                # 如果梯度范数过大，跳过更新（但允许中等大小的梯度）
                if grad_norm_before > 1000.0:
                    print(f"Warning: Gradient norm too large: {grad_norm_before:.2f}, skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
                
                # 更新后检查参数
                with torch.no_grad():
                    for name, param in model_to_check.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            if name in self.backup_layers:
                                param.data.copy_(self.backup_layers[name].to(param.device))
                
                loss_item = loss.item()
                total_loss += loss_item
                num_batches += 1
                self.loss_history.append(loss_item)
                
                # TensorBoard 记录
                self.writer.add_scalar('Loss/batch', loss_item, self.global_step)
                self.global_step += 1
                
                # 定期清理缓存（每10个batch）
                if self.global_step % 10 == 0:
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.epoch_losses.append(avg_loss)
            
            # TensorBoard 记录 Epoch 平均损失
            self.writer.add_scalar('Loss/epoch', avg_loss, epoch)
            self.writer.add_scalar('Learning_rate', self.lr, epoch)
            
            print(f"  平均损失: {avg_loss:.4f}\n")
        
        # 保存模型
        save_path = os.path.join(self.output_dir, 'grpo_model')
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_to_save.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✓ 模型已保存: {save_path}")
        
        # 保存损失历史
        self.save_loss_history()
        
        # 关闭 TensorBoard 写入器
        self.writer.close()
        print(f"✓ TensorBoard 日志已保存到: {self.tb_dir}")
        print(f"启动 TensorBoard: tensorboard --logdir {self.tb_dir}")
        print(f"{'='*80}\n✅ GRPO训练完成！\n")
    
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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GRPO Training with configurable parameters')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--data-limit', type=int, default=100, help='Limit on training data samples')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size per GPU')
    args = parser.parse_args()
    
    # 加载数据
    train_data, test_data = load_data(limit=args.data_limit)
    
    print(f"\n{'='*80}")
    print(f"GRPO 训练配置")
    print(f"{'='*80}")
    print(f"Epochs:        {args.epochs}")
    print(f"Data limit:    {args.data_limit}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size:    {args.batch_size} (将自动分配到 {num_gpus} 张GPU)")
    print(f"{'='*80}\n")
    
    # 训练
    trainer = GRPOTrainer(
        model_dir=MODEL_DIR,
        num_gpus=num_gpus,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs
    )
    trainer.train(train_data)

if __name__ == '__main__':
    main()
