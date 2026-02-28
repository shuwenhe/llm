"""创建演示模型的脚本"""
import os
import torch
from pathlib import Path
from config import ModelConfig
from model import GPT


def create_demo_model(output_path="checkpoints/best_model.pt"):
    """创建一个小的演示模型用于测试生成脚本"""
    
    # 创建 checkpoints 目录
    Path(output_path).parent.mkdir(exist_ok=True)
    
    # 创建小配置（便于快速演示）
    config = ModelConfig(
        n_layer=2,      # 仅 2 层
        n_head=2,       # 2 个注意力头
        n_embd=64,      # 小嵌入维度
        block_size=128  # 短序列
    )
    
    # 创建模型
    model = GPT(config)
    
    print(f"✓ 创建演示模型: {output_path}")
    print(f"  参数量: {model.get_num_params()/1e6:.2f}M")
    print(f"  层数: {config.n_layer}")
    print(f"  嵌入维度: {config.n_embd}")
    
    # 保存模型
    checkpoint = {
        'model': model.state_dict(),
        'model_config': {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'block_size': config.block_size,
            'dropout': config.dropout,
            'bias': config.bias,
            'vocab_size': config.vocab_size,
        },
        'epoch': 0,
    }
    
    torch.save(checkpoint, output_path)
    print(f"✓ 模型已保存到: {output_path}\n")
    print("现在你可以运行:")
    print("  make generate        # 测试文本生成")
    print("  make quick-generate  # 快速测试多个参数组合")


if __name__ == "__main__":
    create_demo_model()
