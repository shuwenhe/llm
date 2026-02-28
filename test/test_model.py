"""快速测试自研后端模型是否正常工作"""
import os
import pickle

import numpy as np

from app.core.models import TinyLM


def test_model():
    """测试模型的基本功能"""
    print("="*50)
    print("测试自研后端 TinyLM 模型")
    print("="*50)

    vocab_size = 1000
    n_embd = 128
    seq_length = 32
    np.random.seed(42)
    
    print(f"\n模型配置:")
    print(f"  - 词表大小: {vocab_size}")
    print(f"  - 嵌入维度: {n_embd}")
    print(f"  - 最大序列长度: {seq_length}")
    
    # 创建模型
    print(f"\n创建模型...")
    model = TinyLM(vocab_size=vocab_size, n_embd=n_embd)
    print(f"✓ 模型创建成功")
    total_params = sum(p.data.size for p in model.parameters())
    print(f"  参数量: {total_params/1e6:.2f}M")
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    batch_size = 2
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_length), dtype=np.int64)
    y = np.random.randint(0, vocab_size, size=(batch_size, seq_length), dtype=np.int64)
    
    logits, loss = model(x, y)
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {x.shape}")
    print(f"  输出logits形状: {logits.shape}")
    print(f"  损失值: {loss.item():.4f}")

    # 自研最小后端不含完整 generate，这里做 argmax 伪生成检查
    print(f"\n测试简化生成(argmax)...")
    next_token = np.argmax(logits.data[0, -1])
    pseudo_generated = np.concatenate([x[:1], np.array([[next_token]], dtype=np.int64)], axis=1)
    print(f"✓ 简化生成成功")
    print(f"  输入长度: {x.shape[1]}")
    print(f"  生成后长度: {pseudo_generated.shape[1]}")
    print(f"  新token: {int(next_token)}")
    
    # 测试模型保存和加载
    print(f"\n测试模型保存和加载...")
    checkpoint_path = "test_checkpoint.pkl"
    state_dict = {f"param_{i}": p.data.copy() for i, p in enumerate(model.parameters())}
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"state_dict": state_dict, "vocab_size": vocab_size, "n_embd": n_embd}, f)
    print(f"✓ 模型保存成功: {checkpoint_path}")

    # 加载模型
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    new_model = TinyLM(vocab_size=checkpoint["vocab_size"], n_embd=checkpoint["n_embd"])
    for i, p in enumerate(new_model.parameters()):
        p.data[...] = checkpoint["state_dict"][f"param_{i}"]

    logits2, loss2 = new_model(x, y)
    assert np.allclose(logits.data, logits2.data), "加载后 logits 不一致"
    assert abs(loss.item() - loss2.item()) < 1e-10, "加载后 loss 不一致"
    print(f"✓ 模型加载成功")

    # 清理测试文件
    os.remove(checkpoint_path)
    print(f"✓ 清理测试文件")
    
    print(f"\n" + "="*50)
    print(f"✅ 所有测试通过！模型工作正常。")
    print(f"="*50)
    print(f"\n下一步:")
    print(f"  1. 运行 'make train'（或 'make train-core'）开始自研后端训练")
    print(f"  2. 运行 'make serve-core' 启动自研后端 API 服务")
    print(f"  3. 运行 'make generate' 或 'make quick-generate' 验证推理链路")


if __name__ == "__main__":
    test_model()
