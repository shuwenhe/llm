"""训练脚本"""
import os
import time
import math
import torch
from torch.nn import functional as F
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from model import GPT
from data import prepare_data, create_dataloader


def get_lr(it, config):
    """学习率调度：warmup + cosine decay"""
    # 1) 线性warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) 最小学习率
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) cosine衰减
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, val_loader, config, device):
    """评估损失"""
    model.eval()
    losses = []
    
    for i, (x, y) in enumerate(val_loader):
        if i >= config.eval_iters:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
            logits, loss = model(x, y)
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)


def train():
    """主训练函数"""
    # 配置
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(train_config.out_dir, exist_ok=True)
    
    # 准备数据
    print("准备数据...")
    clean_data = getattr(train_config, 'clean_data', True)  # 默认清洗数据
    train_dataset, val_dataset, tokenizer = prepare_data(
        dataset_name=train_config.dataset_name,
        dataset_config=train_config.dataset_config,
        block_size=model_config.block_size,
        clean_data=clean_data
    )
    
    train_loader = create_dataloader(train_dataset, train_config.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, train_config.batch_size, shuffle=False)
    
    # 创建模型
    print("创建模型...")
    model = GPT(model_config)
    model.to(device)
    print(f"模型参数量: {model.get_num_params()/1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay
    )
    
    # 可选：编译模型（PyTorch 2.0+）
    if train_config.compile:
        print("编译模型...")
        model = torch.compile(model)
    
    # 训练循环
    print(f"\n开始训练，共{train_config.max_iters}步...")
    model.train()
    iter_num = 0
    best_val_loss = float('inf')
    t0 = time.time()
    
    while iter_num < train_config.max_iters:
        for x, y in train_loader:
            # 学习率调度
            lr = get_lr(iter_num, train_config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # 前向传播
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', 
                                   dtype=torch.float16, enabled=device.type=='cuda'):
                logits, loss = model(x, y)
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # 梯度裁剪
            if train_config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            
            optimizer.step()
            
            # 日志
            if iter_num % train_config.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(f"iter {iter_num}: loss {loss.item():.4f}, lr {lr:.2e}, time {dt*1000:.2f}ms")
            
            # 评估
            if iter_num % train_config.eval_interval == 0 and iter_num > 0:
                val_loss = estimate_loss(model, val_loader, train_config, device)
                print(f"iter {iter_num}: val_loss {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_config': model_config.__dict__,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"保存最佳模型 (val_loss={val_loss:.4f})")
                    torch.save(checkpoint, os.path.join(train_config.out_dir, 'best_model.pt'))
            
            # 定期保存checkpoint
            if iter_num % train_config.save_interval == 0 and iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_config': model_config.__dict__,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(train_config.out_dir, f'ckpt_{iter_num}.pt'))
                print(f"保存checkpoint: iter {iter_num}")
            
            iter_num += 1
            if iter_num >= train_config.max_iters:
                break
    
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
