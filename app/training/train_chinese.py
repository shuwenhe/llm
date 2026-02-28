"""简化版中文文本训练 - 使用示例数据"""
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from app.modeling.config import ModelConfig
from app.modeling.model import GPT
from app.modeling.data import TextDataset, load_tokenizer


def create_sample_chinese_corpus():
    """创建示例中文语料库用于演示"""
    texts = [
        "北京是中国的首都，位于华北平原的中部。",
        "长城是中国古代最伟大的建筑之一。",
        "中国有着悠久的历史文化传统。",
        "技术创新推动了社会的发展和进步。",
        "语言是人类最重要的交流工具。",
        "教育是发展国家的基础和关键。",
        "科学研究对人类文明进步至关重要。",
        "文学作品反映了不同时代的社会现实。",
        "音乐艺术是人类文化遗产的重要组成部分。",
        "体育运动促进了身心的健康发展。",
    ] * 100  # 重复以增加数据量
    
    return texts


def train_chinese_text(
    learning_rate=1e-4,
    batch_size=4,
    num_epochs=1,
    checkpoint_path="checkpoints/model.pt",
    output_path="checkpoints/model.pt",
    data_file=None,
    save_every_epoch=True,
    keep_last_n=3,
):
    """训练中文文本能力
    
    Args:
        data_file: 中文文本数据文件路径，如果为None则使用示例数据
        save_every_epoch: 是否保存每个epoch的检查点
        keep_last_n: 保留最近N个检查点（0表示保留所有）
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 设备: {device}")
    
    # 创建checkpoints目录
    checkpoint_dir = Path(output_path).parent
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 加载或创建模型
    print("📋 加载模型...")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = ModelConfig(**checkpoint["model_config"])
        model = GPT(model_config)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✓ 从 {checkpoint_path} 恢复训练 (从 Epoch {start_epoch} 开始)")
    else:
        model_config = ModelConfig()
        model = GPT(model_config)
        print(f"✓ 创建新模型")
    
    model = model.to(device)
    
    # 加载数据
    if data_file and os.path.exists(data_file):
        print(f"\n📚 从文件加载数据: {data_file}")
        with open(data_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"✓ 加载了 {len(texts)} 条文本")
    else:
        if data_file:
            print(f"\n⚠️  文件不存在: {data_file}")
        print(f"📚 使用示例中文数据集...")
        texts = create_sample_chinese_corpus()
        print(f"✓ 创建了 {len(texts)} 条示例文本")
    
    # Tokenize
    tokenizer = load_tokenizer()
    print(f"🔤 Tokenizing 数据...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.uint32)
    print(f"✓ 总共 {len(all_tokens):,} 个 tokens")
    
    # 创建数据集
    print(f"\n📦 创建数据集 (block_size={model_config.block_size})...")
    train_size = int(len(all_tokens) * 0.8)
    
    train_tokens = all_tokens[:train_size]
    val_tokens = all_tokens[train_size:]
    
    train_dataset = TextDataset(train_tokens, model_config.block_size)
    val_dataset = TextDataset(val_tokens, model_config.block_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"✓ 训练集: {len(train_dataset):,} 样本")
    print(f"✓ 验证集: {len(val_dataset):,} 样本")
    
    # 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\n{'='*70}")
    print(f"🎓 训练配置")
    print(f"{'='*70}")
    print(f"  设备: {device}")
    print(f"  学习率: {learning_rate}")
    print(f"  批大小: {batch_size}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  训练样本: {len(train_dataset):,} 个")
    print(f"  验证样本: {len(val_dataset):,} 个")
    print(f"  每轮步数: {len(train_loader):,} 步")
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  保存路径: {output_path}")
    print(f"{'='*70}\n")
    
    best_loss = float('inf')
    
    import time
    training_start = time.time()
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"📚 Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
        model.train()
        epoch_loss = 0
        epoch_batches = 0
        
        # 训练进度条添加实时loss显示
        progress_bar = tqdm(train_loader, desc=f"训练中")
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)
            
            # 前向传播
            logits, loss = model(x, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
            
            # 实时显示当前loss
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/epoch_batches:.4f}'
            })
        
        # 计算损失
        avg_epoch_loss = epoch_loss / max(1, epoch_batches)
        
        # 验证
        model.eval()
        val_loss = 0
        val_batches = 0
        
        print(f"\n🔍 验证中...")
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="验证"):
                x = x.to(device)
                y = y.to(device)
                logits, loss = model(x, y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(1, val_batches)
        epoch_time = time.time() - epoch_start
        
        # 记录历史
        history['train_loss'].append(avg_epoch_loss)
        history['val_loss'].append(avg_val_loss)
        history['epochs'].append(epoch + 1)
        
        # 显示结果
        print(f"\n{'='*70}")
        print(f"📊 Epoch {epoch+1}/{start_epoch + num_epochs} 结果")
        print(f"{'='*70}")
        print(f"  ⏱️  用时: {epoch_time:.1f}s ({epoch_time/60:.1f}min)")
        print(f"  📉 训练损失: {avg_epoch_loss:.4f}")
        print(f"  📊 验证损失: {avg_val_loss:.4f}")
        
        # 保存检查点的通用函数
        def save_checkpoint(path, is_best=False):
            torch.save({
                'model': model.state_dict(),
                'model_config': model_config.__dict__,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'train_loss': avg_epoch_loss,
                'val_loss': avg_val_loss,
                'history': history,
            }, path)
        
        print(f"\n💾 保存检查点...")
        
        # 1. 保存每个epoch的检查点
        if save_every_epoch:
            epoch_checkpoint = checkpoint_dir / f"model_epoch_{epoch+1}.pt"
            save_checkpoint(epoch_checkpoint)
            size_mb = epoch_checkpoint.stat().st_size / (1024*1024)
            print(f"  ✓ Epoch检查点: {epoch_checkpoint.name} ({size_mb:.1f}MB)")
            
            # 清理旧检查点（保留最近N个）
            if keep_last_n > 0:
                epoch_files = sorted(checkpoint_dir.glob("model_epoch_*.pt"))
                if len(epoch_files) > keep_last_n:
                    for old_file in epoch_files[:-keep_last_n]:
                        old_file.unlink()
                        print(f"  🗑️  删除旧检查点: {old_file.name}")
        
        # 2. 保存最佳模型
        if avg_val_loss < best_loss:
            improvement = best_loss - avg_val_loss
            best_loss = avg_val_loss
            
            best_model_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(best_model_path, is_best=True)
            size_mb = best_model_path.stat().st_size / (1024*1024)
            print(f"  🏆 最佳模型: {best_model_path.name} ({size_mb:.1f}MB) [改进: {improvement:.4f}]")
        else:
            print(f"  ℹ️  验证损失未改进 (最佳: {best_loss:.4f})")
        
        # 3. 始终保存最新模型（用于断点续训）
        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(latest_path)
        size_mb = latest_path.stat().st_size / (1024*1024)
        print(f"  📌 最新模型: {latest_path.name} ({size_mb:.1f}MB) [用于恢复训练]")
        
        # 4. 更新主检查点（保持向后兼容）
        save_checkpoint(output_path)
        size_mb = Path(output_path).stat().st_size / (1024*1024)
        print(f"  📍 主检查点: {Path(output_path).name} ({size_mb:.1f}MB)")
        
        print(f"{'='*70}")
    
    training_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"✅ 训练完成!")
    print(f"{'='*70}")
    print(f"  ⏱️  总用时: {training_time:.1f}s ({training_time/60:.1f}min)")
    print(f"  📊 最佳验证损失: {best_loss:.4f}")
    print(f"\n📁 保存的模型:")
    print(f"  🏆 最佳模型: checkpoints/best_model.pt")
    print(f"  📌 最新模型: checkpoints/latest.pt")
    if save_every_epoch:
        print(f"  📦 Epoch检查点: checkpoints/model_epoch_*.pt (最近{keep_last_n}个)")
    print(f"\n🚀 使用训练后的模型:")
    print(f"  # 使用最佳模型")
    print(f"  LLM_CHECKPOINT=checkpoints/best_model.pt make serve")
    print(f"  # 或继续训练")
    print(f"  python train_chinese.py --checkpoint checkpoints/latest.pt --epochs 3")
    print(f"{'='*70}\n")
    
    # 保存训练历史
    history_file = checkpoint_dir / "training_history.json"
    import json
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📈 训练历史已保存: {history_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="中文文本训练")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint", default="checkpoints/model.pt", help="加载的检查点路径")
    parser.add_argument("--output", default="checkpoints/model.pt", help="输出检查点路径")
    parser.add_argument("--data-file", help="中文文本数据文件路径 (如: data/zh_wiki.txt)")
    parser.add_argument("--save-every-epoch", action="store_true", default=True, help="保存每个epoch的检查点")
    parser.add_argument("--keep-last-n", type=int, default=3, help="保留最近N个epoch检查点")
    parser.add_argument("--resume", action="store_true", help="从最新检查点恢复训练")
    
    args = parser.parse_args()
    
    # 如果指定resume，使用latest.pt
    checkpoint_path = args.checkpoint
    if args.resume:
        latest = Path(args.output).parent / "latest.pt"
        if latest.exists():
            checkpoint_path = str(latest)
            print(f"🔄 从最新检查点恢复: {checkpoint_path}")
    
    train_chinese_text(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        output_path=args.output,
        data_file=args.data_file,
        save_every_epoch=args.save_every_epoch,
        keep_last_n=args.keep_last_n,
    )
