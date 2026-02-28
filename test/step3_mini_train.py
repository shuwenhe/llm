"""Step 3: 迷你训练 10 step 验证（检查 loss 下降趋势）"""

import torch

from app.modeling.config import ModelConfig
from app.modeling.model import GPT


def run_step3_mini_train_check() -> None:
    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=128,
        block_size=64,
        dropout=0.0,
    )

    model = GPT(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batch_size = 4
    seq_len = 32

    # 固定小批次，验证是否能在短步数内过拟合
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)

    losses: list[float] = []

    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)

        assert logits.shape == (batch_size, seq_len, config.vocab_size), (
            f"logits 形状错误: {logits.shape}, 期望 {(batch_size, seq_len, config.vocab_size)}"
        )
        assert loss is not None and torch.isfinite(loss), f"step={step} loss 非法: {loss}"

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    start_loss = losses[0]
    end_loss = losses[-1]

    # 采用宽松但明确的检查：末尾 loss 需要低于起始 loss
    assert end_loss < start_loss, (
        f"10 step 后 loss 未下降: start={start_loss:.6f}, end={end_loss:.6f}, losses={losses}"
    )

    print("✅ Step 3 完成：迷你训练 10 step 打通")
    print(f"   初始 loss: {start_loss:.6f}")
    print(f"   结束 loss: {end_loss:.6f}")
    print("   loss 序列:")
    print("   " + " -> ".join(f"{v:.4f}" for v in losses))


if __name__ == "__main__":
    run_step3_mini_train_check()
