"""Step 1: 最小前向传播验证（模型结构与输出形状）"""

import torch

from app.modeling.config import ModelConfig
from app.modeling.model import GPT


def run_step1_forward_check() -> None:
    config = ModelConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=128,
        block_size=64,
        dropout=0.0,
    )

    model = GPT(config)
    model.eval()

    batch_size = 2
    seq_len = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        logits, loss = model(x, y)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), (
        f"logits 形状错误: {logits.shape}, 期望 {(batch_size, seq_len, config.vocab_size)}"
    )
    assert loss is not None, "loss 不应为 None"
    assert torch.isfinite(loss), f"loss 非有限值: {loss}"

    print("✅ Step 1 完成：模型前向传播打通")
    print(f"   输入形状: {tuple(x.shape)}")
    print(f"   logits形状: {tuple(logits.shape)}")
    print(f"   loss: {loss.item():.6f}")


if __name__ == "__main__":
    run_step1_forward_check()
