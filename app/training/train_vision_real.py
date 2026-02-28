"""使用真实图像数据集训练视觉编码器

支持的数据集:
1. 本地图像文件夹 (images/*.jpg + captions.txt)
2. Hugging Face 数据集 (例如: nlphuji/flickr30k)
"""
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from app.modeling.config import ModelConfig
from app.modeling.model import GPT
from app.modeling.data import load_tokenizer


class LocalImageCaptionDataset(Dataset):
    """本地图像-文本数据集
    
    目录结构:
        data/vision_train/
            images/
                img001.jpg
                img002.jpg
            captions.json  # {"img001.jpg": "描述文本", ...}
    """
    def __init__(self, data_dir, image_size=224, block_size=128, transform=None):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.image_size = image_size
        self.block_size = block_size
        self.tokenizer = load_tokenizer()
        
        # 加载标注
        captions_file = self.data_dir / "captions.json"
        if not captions_file.exists():
            raise FileNotFoundError(f"找不到 {captions_file}")
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)
        
        self.image_files = list(self.captions.keys())
        print(f"加载了 {len(self.image_files)} 个图像-文本对")
        
        # 图像预处理
        if transform is None:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 编码文本
        caption = self.captions[img_name]
        text = f"图片描述：{caption}"
        tokens = self.tokenizer.encode(text)
        
        # 截断或填充
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.block_size - len(tokens))
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y, image, torch.zeros(1)


class HuggingFaceImageCaptionDataset(Dataset):
    """Hugging Face 图像-文本数据集"""
    def __init__(self, dataset_name="nlphuji/flickr30k", split="train", 
                 image_size=224, block_size=128, max_samples=None):
        from datasets import load_dataset
        from torchvision import transforms
        
        self.image_size = image_size
        self.block_size = block_size
        self.tokenizer = load_tokenizer()
        
        print(f"加载数据集 {dataset_name} ({split})...")
        self.dataset = load_dataset(dataset_name, split=split)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"数据集大小: {len(self.dataset)}")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # 加载图像 - 处理不同的字段名
        image = None
        for img_key in ['image', 'img', 'picture']:
            if img_key in sample:
                image = sample[img_key]
                break
        
        if image is None:
            raise ValueError(f"找不到图像字段，可用字段: {list(sample.keys())}")
            
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        image = self.transform(image)
        
        # 获取文本 - 处理不同的字段名和格式
        caption = None
        for cap_key in ['caption', 'text', 'captions', 'sentences']:
            if cap_key in sample:
                caption = sample[cap_key]
                break
        
        if caption is None:
            raise ValueError(f"找不到文本字段，可用字段: {list(sample.keys())}")
        
        # 处理列表形式的caption
        if isinstance(caption, list):
            import random
            caption = random.choice(caption) if caption else "图片"
        
        # 编码文本
        text = f"图片：{caption}"
        tokens = self.tokenizer.encode(text)
        
        # 截断或填充
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.block_size - len(tokens))
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y, image, torch.zeros(1)


class CIFAR10CaptionDataset(Dataset):
    """CIFAR-10 图像分类数据转文本描述数据集（可直接下载）"""
    def __init__(self, root="data/cifar10", train=True, image_size=224, block_size=128):
        from torchvision import datasets, transforms

        self.block_size = block_size
        self.tokenizer = load_tokenizer()
        self.class_names = [
            "飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"
        ]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )
        print(f"CIFAR-10 数据集大小: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        caption = f"这张图片的主要物体是{self.class_names[int(label)]}。"
        text = f"图片描述：{caption}"
        tokens = self.tokenizer.encode(text)

        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.block_size - len(tokens))

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y, image, torch.zeros(1)


def train_vision_encoder_real(
    data_source="cifar10",  # "local" / "huggingface" / "cifar10"
    data_path="data/vision_train",  # 本地数据路径
    dataset_name="nlphuji/flickr30k",  # HF数据集名称
    batch_size=8,
    num_epochs=10,
    max_steps=None,
    learning_rate=1e-4,
    checkpoint_path="checkpoints/best_model.pt",
    output_path="checkpoints/vision_trained_model.pt"
):
    """使用真实数据训练视觉编码器"""
    
    print("=" * 60)
    print("视觉编码器训练脚本 (真实数据)")
    print("=" * 60)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型从 {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到检查点 {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ModelConfig(**checkpoint["model_config"])
    model_config.multimodal_enabled = True
    
    model = GPT(model_config)
    state_dict = checkpoint["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    
    # 冻结语言模型，只训练视觉编码器
    print("\n配置训练参数...")
    for name, param in model.named_parameters():
        if 'visual_encoder' not in name:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 创建数据集
    print(f"\n准备数据集 (source={data_source})...")
    try:
        if data_source == "local":
            train_dataset = LocalImageCaptionDataset(data_path)
        elif data_source == "huggingface":
            train_dataset = HuggingFaceImageCaptionDataset(
                dataset_name=dataset_name,
                max_samples=5000  # 限制样本数以加快训练
            )
        elif data_source == "cifar10":
            train_dataset = CIFAR10CaptionDataset()
        else:
            raise ValueError(f"未知的数据源: {data_source}")
    except Exception as e:
        print(f"错误: 无法加载数据集 - {e}")
        print("\n提示: 如果使用 HuggingFace 数据集，请先安装: pip install datasets")
        print("      如果使用本地数据，请确保数据格式正确")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 优化器
    vision_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(vision_params, lr=learning_rate, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader)
    )
    
    # 训练循环
    print(f"\n开始训练 {num_epochs} 个 epoch...")
    print(f"批次大小: {batch_size}, 每个epoch {len(train_loader)} 个批次")
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    stop_early = False
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (x, y, image, audio) in enumerate(pbar):
            try:
                x = x.to(device)
                y = y.to(device)
                image = image.to(device)
                
                # 前向传播
                optimizer.zero_grad()
                logits, loss = model(x, y, image=image, audio=None)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vision_params, 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
                global_step += 1

                if max_steps is not None and global_step >= max_steps:
                    stop_early = True
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{epoch_loss/(batch_idx+1):.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
            except Exception as e:
                print(f"\n警告: 批次 {batch_idx} 处理失败: {e}")
                continue

            if stop_early:
                break
        
        avg_epoch_loss = epoch_loss / max(1, epoch_batches)
        print(f"\nEpoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"→ 新的最佳损失! 保存模型...")
            
            save_state_dict = {}
            for k, v in model.state_dict().items():
                save_state_dict[f"_orig_mod.{k}"] = v
            
            torch.save({
                'model': save_state_dict,
                'model_config': model_config.__dict__,
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'best_loss': best_loss,
            }, output_path)

        if stop_early:
            print(f"\n达到 max_steps={max_steps}，提前结束训练。")
            break
    
    print("\n" + "=" * 60)
    print(f"✓ 训练完成! 最佳损失: {best_loss:.4f}")
    print(f"✓ 模型已保存到: {output_path}")
    print("\n使用微调后的模型:")
    print(f"  export LLM_CHECKPOINT={output_path}")
    print("  make serve-dev")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练视觉编码器")
    parser.add_argument("--data-source", choices=["local", "huggingface", "cifar10"], 
                       default="cifar10", help="数据源类型")
    parser.add_argument("--data-path", default="data/vision_train", 
                       help="本地数据路径")
    parser.add_argument("--dataset-name", default="nlphuji/flickr30k",
                       help="HuggingFace数据集名称")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0,
                       help="最大训练步数，>0 时提前结束（用于快速验证）")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--output", default="checkpoints/model.pt")
    
    args = parser.parse_args()
    
    train_vision_encoder_real(
        data_source=args.data_source,
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_steps=(args.max_steps if args.max_steps > 0 else None),
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint,
        output_path=args.output
    )
