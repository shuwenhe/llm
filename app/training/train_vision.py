"""视觉编码器微调脚本 - 使用COCO Captions数据集"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO

from app.modeling.config import ModelConfig
from app.modeling.model import GPT
from app.modeling.data import load_tokenizer


class SimpleCaptionDataset(Dataset):
    """简单的图像-文本配对数据集"""
    def __init__(self, image_size=224, block_size=128, num_samples=1000):
        self.image_size = image_size
        self.block_size = block_size
        self.tokenizer = load_tokenizer()
        
        # 使用一些示例数据（实际应用中应该使用真实的图像-文本对）
        self.samples = []
        print(f"生成 {num_samples} 个训练样本...")
        
        # 示例：创建一些合成数据用于演示
        # 实际应该使用COCO、Flickr30k等真实数据集
        descriptions = [
            "一只猫坐在窗台上",
            "蓝天白云下的高山",
            "城市街道上的汽车",
            "桌子上的食物",
            "一朵红色的花",
            "海边的日落景色",
            "森林中的小路",
            "雪地里的房子",
            "一个微笑的人",
            "书架上的书籍"
        ]
        
        for i in range(num_samples):
            desc = descriptions[i % len(descriptions)]
            self.samples.append({
                'description': desc,
                'image_id': i  # 用于生成合成图像
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 生成合成图像（实际应该加载真实图像）
        # 这里用随机噪声模拟，实际训练应该加载真实图片
        image = torch.randn(3, self.image_size, self.image_size)
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        # 编码文本
        text = f"描述：{sample['description']}"
        tokens = self.tokenizer.encode(text)
        
        # 截断或填充到block_size
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            tokens = tokens + [self.tokenizer.eos_token_id] * (self.block_size - len(tokens))
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        # image 已是 (C,H,W)，DataLoader 拼接后得到 (B,C,H,W)
        return x, y, image, torch.zeros(1)


def train_vision_encoder():
    """训练/微调视觉编码器"""
    print("=" * 60)
    print("视觉编码器训练脚本")
    print("=" * 60)
    
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 使用脚本所在目录的相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoints/best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到检查点文件 {checkpoint_path}")
        return
    
    # 加载模型
    print(f"加载模型从 {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ModelConfig(**checkpoint["model_config"])
    model_config.multimodal_enabled = True  # 确保启用多模态
    
    model = GPT(model_config)
    
    # 加载权重
    state_dict = checkpoint["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    
    # 冻结除视觉编码器外的所有参数
    print("冻结语言模型参数，只训练视觉编码器...")
    for name, param in model.named_parameters():
        if 'visual_encoder' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"  可训练: {name}")
    
    # 创建数据集
    print("\n准备数据集...")
    train_dataset = SimpleCaptionDataset(
        image_size=224,
        block_size=128,
        num_samples=500  # 小数据集用于演示
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 优化器（只优化视觉编码器参数）
    vision_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(vision_params, lr=1e-4, weight_decay=0.01)
    
    # 训练循环
    num_epochs = 5
    print(f"\n开始训练 {num_epochs} 个 epoch...")
    
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (x, y, image, audio) in enumerate(pbar):
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
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
    
    # 保存微调后的模型
    output_path = os.path.join(script_dir, "checkpoints/vision_finetuned_model.pt")
    print(f"\n保存微调后的模型到 {output_path}...")
    
    # 获取当前state dict（需要添加_orig_mod前缀以匹配原始格式）
    save_state_dict = {}
    for k, v in model.state_dict().items():
        save_state_dict[f"_orig_mod.{k}"] = v
    
    torch.save({
        'model': save_state_dict,
        'model_config': model_config.__dict__,
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
    }, output_path)
    
    print("✓ 训练完成！")
    print(f"\n要使用微调后的模型，请设置环境变量:")
    print(f"  export LLM_CHECKPOINT={output_path}")
    print("  然后重启服务: make serve-dev")


if __name__ == "__main__":
    train_vision_encoder()
