"""Step 2: 单步反向传播验证（forward + backward + optimizer.step）"""

import copy

import torch

from app.modeling.config import ModelConfig
from app.modeling.model import GPT


def _pick_param_tensor(model: GPT) -> torch.Tensor:
    for param in model.parameters():
        if param.requires_grad:
            return param
    raise RuntimeError("模型中没有可训练参数")


def run_step2_backward_check() -> None:
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 2
    seq_len = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)

    tracked_param = _pick_param_tensor(model)
    param_before = tracked_param.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), (
        f"logits 形状错误: {logits.shape}, 期望 {(batch_size, seq_len, config.vocab_size)}"
    )
    assert loss is not None, "loss 不应为 None"
    assert torch.isfinite(loss), f"loss 非有限值: {loss}"

    loss.backward()

    grad_norm_sq = 0.0
    grad_found = False
    for param in model.parameters():
        if param.grad is not None:
            grad_found = True
            grad_norm_sq += float(param.grad.detach().pow(2).sum().item())
    assert grad_found, "未找到任何梯度，backward 可能未生效"
    grad_norm = grad_norm_sq ** 0.5
    assert grad_norm > 0, "梯度范数为 0，训练步无效"

    optimizer.step()

    param_after = tracked_param.detach().clone()
    param_changed = not torch.equal(param_before, param_after)
    assert param_changed, "参数未发生变化，optimizer.step 可能未生效"

    print("✅ Step 2 完成：单步反向传播打通")
    print(f"   输入形状: {tuple(x.shape)}")
    print(f"   logits形状: {tuple(logits.shape)}")
    print(f"   loss: {loss.item():.6f}")
    print(f"   grad_norm: {grad_norm:.6f}")
    print("   参数更新: 已发生")


if __name__ == "__main__":
    run_step2_backward_check()
