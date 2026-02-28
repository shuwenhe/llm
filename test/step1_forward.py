"""Step 1: 最小前向传播验证（自研后端）"""

import numpy as np

from app.core.models import TinyLM


def run_step1_forward_check() -> None:
    np.random.seed(42)
    vocab_size = 1000
    n_embd = 128
    model = TinyLM(vocab_size=vocab_size, n_embd=n_embd)

    batch_size = 2
    seq_len = 32
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    y = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)

    logits, loss = model(x, y)

    assert logits.shape == (batch_size, seq_len, vocab_size), (
        f"logits 形状错误: {logits.shape}, 期望 {(batch_size, seq_len, vocab_size)}"
    )
    assert loss is not None, "loss 不应为 None"
    assert np.isfinite(loss.item()), f"loss 非有限值: {loss.item()}"

    print("✅ Step 1 完成：模型前向传播打通")
    print(f"   输入形状: {tuple(x.shape)}")
    print(f"   logits形状: {tuple(logits.shape)}")
    print(f"   loss: {loss.item():.6f}")


if __name__ == "__main__":
    run_step1_forward_check()
