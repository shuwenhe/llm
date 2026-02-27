"""快速测试模型是否正常工作"""
import torch
from config import ModelConfig
from model import GPT


def test_model():
    """测试模型的基本功能"""
    print("="*50)
    print("测试LLM模型")
    print("="*50)
    
    # 创建小模型配置（用于测试）
    config = ModelConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=128,
        block_size=64
    )
    
    print(f"\n模型配置:")
    print(f"  - 词表大小: {config.vocab_size}")
    print(f"  - 层数: {config.n_layer}")
    print(f"  - 注意力头数: {config.n_head}")
    print(f"  - 嵌入维度: {config.n_embd}")
    print(f"  - 最大序列长度: {config.block_size}")
    
    # 创建模型
    print(f"\n创建模型...")
    model = GPT(config)
    print(f"✓ 模型创建成功")
    print(f"  参数量: {model.get_num_params()/1e6:.2f}M")
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    batch_size = 2
    seq_length = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    logits, loss = model(x, y)
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {x.shape}")
    print(f"  输出logits形状: {logits.shape}")
    print(f"  损失值: {loss.item():.4f}")
    
    # 测试生成
    print(f"\n测试文本生成...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(x[:1], max_new_tokens=10, temperature=1.0)
    print(f"✓ 生成成功")
    print(f"  输入长度: {x.shape[1]}")
    print(f"  生成后长度: {generated.shape[1]}")
    print(f"  生成的tokens: {generated[0].tolist()[:20]}...")
    
    # 测试模型保存和加载
    print(f"\n测试模型保存和加载...")
    checkpoint_path = "test_checkpoint.pt"
    torch.save({
        'model': model.state_dict(),
        'config': config.__dict__
    }, checkpoint_path)
    print(f"✓ 模型保存成功: {checkpoint_path}")
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path)
    new_model = GPT(config)
    new_model.load_state_dict(checkpoint['model'])
    print(f"✓ 模型加载成功")
    
    # 清理测试文件
    import os
    os.remove(checkpoint_path)
    print(f"✓ 清理测试文件")
    
    print(f"\n" + "="*50)
    print(f"✅ 所有测试通过！模型工作正常。")
    print(f"="*50)
    print(f"\n下一步:")
    print(f"  1. 运行 'python train.py' 开始训练")
    print(f"  2. 训练完成后运行 'python generate.py' 生成文本")


if __name__ == "__main__":
    test_model()
