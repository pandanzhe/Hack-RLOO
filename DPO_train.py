#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Multi-GPU Training
充分利用4张显卡进行训练 - 修复版本（支持可配置参数）
python DPO_train.py --epochs 4 --data-limit 800 --lr 1e-5 --beta 1.0
"""

import os
# 如果环境变量未设置，则使用所有GPU；否则使用环境变量指定的GPU
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import json
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print(f"{'='*80}")
print(f"DPO Multi-GPU Training (演示版本)")
print(f"{'='*80}")
print(f"Device: {device}, GPUs: {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
print(f"{'='*80}\n")

DATA_DIR = '/root/Train-RL/Data/Data'
MODEL_DIR = '/root/Train-RL/models/gemma-2b-it'
OUTPUT_DIR = '/root/Train-RL/outputs/dpo'
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

class DPODataset(Dataset):
    def __init__(self, pairs: List[Dict], tokenizer, max_len: int = 256):  # 减少max_len以节省显存
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        chosen = pair['prompt'] + pair['chosen']
        rejected = pair['prompt'] + pair['rejected']
        
        chosen_tokens = self.tokenizer(chosen, max_length=self.max_len,
                                      truncation=True, padding='max_length',
                                      return_tensors='pt')
        rejected_tokens = self.tokenizer(rejected, max_length=self.max_len,
                                        truncation=True, padding='max_length',
                                        return_tensors='pt')
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_mask': rejected_tokens['attention_mask'].squeeze(),
        }

class DPOTrainer:
    def __init__(self, model_dir: str, num_gpus: int = num_gpus,
                 batch_size: int = 1, lr: float = 5e-6, epochs: int = 3, beta: float = 0.5):
        self.device = device
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.beta = beta
        self.output_dir = OUTPUT_DIR        
        self.loss_history = []  # 记录训练损失
        self.epoch_losses = []  # 记录每个epoch的损失
        
        # 初始化 TensorBoard
        self.tb_dir = os.path.join(OUTPUT_DIR, 'tensorboard')
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        self.global_step = 0
        print(f"加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                                       trust_remote_code=True,
                                                       local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型到主GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device)
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device)
        self.ref_model.eval()
        
        # 多卡包装 - 使用DataParallel充分利用4张GPU
        if num_gpus > 1:
            print(f"✓ 启用DataParallel ({num_gpus} GPUs)")
            self.model = nn.DataParallel(self.model, device_ids=list(range(num_gpus)))
            self.ref_model = nn.DataParallel(self.ref_model, device_ids=list(range(num_gpus)))
            print(f"✓ 模型已分配到 {num_gpus} 张GPU上")
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        print(f"✓ 模型加载完成\n")
    
    def compute_log_probs(self, model, input_ids, attention_mask, require_grad=False):
        """计算log概率 - 改进数值稳定性"""
        if require_grad:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits.float()  # 转换为float32计算
        log_probs = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1)
        shift_labels = input_ids[..., 1:].unsqueeze(-1)
        log_probs = torch.gather(log_probs, -1, shift_labels).squeeze(-1)
        mask = attention_mask[..., 1:].float()
        masked_log_probs = log_probs * mask
        sum_lp = masked_log_probs.sum(dim=-1)
        seq_len = mask.sum(dim=-1).clamp(min=1e-8)
        return sum_lp / seq_len
    
    def train(self, train_data: List[Dict]):
        print(f"{'='*80}")
        print(f"{'='*80}")
        print(f"DPO训练 ({self.num_gpus} GPU)")
        print(f"{'='*80}\n")
        
        # 生成偏好对 (使用所有训练数据)
        preference_pairs = []
        for ex in train_data:
            q = ex.get('question', '')
            a = ex.get('answer', '')
            prompt = f"数学问题: {q}\n答案: "
            preference_pairs.append({
                'prompt': prompt,
                'chosen': a,
                'rejected': a[:max(1, len(a)//2)]
            })
        
        print(f"✓ 生成{len(preference_pairs)}个偏好对 (使用全部 {len(train_data)} 个样本)\n")
        
        dataset = DPODataset(preference_pairs, self.tokenizer)
        # 多GPU时，DataParallel会自动将batch分配到各个GPU
        # 所以这里使用原始batch_size，DataParallel会自动处理
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                chosen_ids = batch['chosen_input_ids'].to(self.device)
                chosen_mask = batch['chosen_mask'].to(self.device)
                rejected_ids = batch['rejected_input_ids'].to(self.device)
                rejected_mask = batch['rejected_mask'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 参考模型的log概率（无梯度）
                with torch.no_grad():
                    ref_chosen_lp = self.compute_log_probs(self.ref_model, chosen_ids, chosen_mask, require_grad=False)
                    ref_rejected_lp = self.compute_log_probs(self.ref_model, rejected_ids, rejected_mask, require_grad=False)
                
                # 策略模型的log概率（需要梯度）
                policy_chosen_lp = self.compute_log_probs(self.model, chosen_ids, chosen_mask, require_grad=True)
                policy_rejected_lp = self.compute_log_probs(self.model, rejected_ids, rejected_mask, require_grad=True)
                
                # 标准DPO损失：-log(sigmoid(beta*(log(pi/ref_chosen) - log(pi/ref_rejected))))
                policy_log_ratio = policy_chosen_lp - policy_rejected_lp
                ref_log_ratio = ref_chosen_lp - ref_rejected_lp
                log_ratio_diff = policy_log_ratio - ref_log_ratio
                
                # 使用logsigmoid提高数值稳定性
                loss = -torch.nn.functional.logsigmoid(self.beta * log_ratio_diff).mean()
                
                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf detected in loss, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                loss_item = loss.item()
                total_loss += loss_item
                num_batches += 1
                self.loss_history.append(loss_item)  # 记录每个batch的损失
                
                # TensorBoard 记录
                self.writer.add_scalar('Loss/batch', loss_item, self.global_step)
                self.global_step += 1
                
                # 定期清理缓存（每10个batch）
                if self.global_step % 10 == 0:
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.epoch_losses.append(avg_loss)  # 记录每个epoch的平均损失
            
            # TensorBoard 记录 Epoch 平均损失
            self.writer.add_scalar('Loss/epoch', avg_loss, epoch)
            self.writer.add_scalar('Learning_rate', self.lr, epoch)
            
            print(f"  平均损失: {avg_loss:.4f}\n")
        
        # 保存模型
        save_path = os.path.join(self.output_dir, 'dpo_model')
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
        print(f"{'='*80}\n✅ DPO训练完成！\n")
    
    def save_loss_history(self):
        """保存训练损失历史为JSON和CSV"""
        import json
        
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
    parser = argparse.ArgumentParser(description='DPO Training with configurable parameters')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--data-limit', type=int, default=30, help='Limit on training data samples')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for DPO loss')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    args = parser.parse_args()
    
    # 加载数据
    train_data, test_data = load_data(limit=args.data_limit)
    
    print(f"\n{'='*80}")
    print(f"DPO 训练配置")
    print(f"{'='*80}")
    print(f"Epochs:        {args.epochs}")
    print(f"Data limit:    {args.data_limit} (将生成约{args.data_limit * 20 // 30}个偏好对)")
    print(f"Learning rate: {args.lr}")
    print(f"Beta:          {args.beta}")
    print(f"Batch size:    {args.batch_size}")
    print(f"{'='*80}\n")
    
    # 训练
    trainer = DPOTrainer(
        model_dir=MODEL_DIR,
        num_gpus=num_gpus,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        beta=args.beta
    )
    trainer.train(train_data)

if __name__ == '__main__':
    main()
