"""自研后端训练主链路（纯 numpy）"""

import argparse
import os
import pickle

import numpy as np

from app.core.models import TinyLM
from app.core.optim import AdamW
from app.core.tokenizer import CharTokenizer


def _sample_corpus():
    return [
        "北京是中国的首都。",
        "人工智能正在改变世界。",
        "语言模型可以生成文本。",
        "机器学习需要数据和算力。",
        "模型训练要关注损失函数下降。",
    ] * 200


def _make_batches(token_ids, batch_size, seq_len):
    token_ids = np.asarray(token_ids, dtype=np.int64)
    max_start = len(token_ids) - seq_len - 1
    while True:
        starts = np.random.randint(0, max_start, size=batch_size)
        x = np.stack([token_ids[s:s + seq_len] for s in starts], axis=0)
        y = np.stack([token_ids[s + 1:s + seq_len + 1] for s in starts], axis=0)
        yield x, y


def train_core(batch_size=8, epochs=1, learning_rate=3e-3, output="checkpoints/model_core.pkl"):
    np.random.seed(42)

    texts = _sample_corpus()
    tokenizer = CharTokenizer.from_texts(texts)
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenizer.encode(t))

    seq_len = 32
    model = TinyLM(vocab_size=tokenizer.vocab_size, n_embd=128)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    steps_per_epoch = 10
    total_steps = max(1, epochs * steps_per_epoch)
    batches = _make_batches(all_tokens, batch_size=batch_size, seq_len=seq_len)

    losses = []
    for step in range(total_steps):
        x, y = next(batches)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == total_steps - 1:
            print(f"step {step+1}/{total_steps}: loss={loss.item():.4f}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    state_dict = {f"param_{i}": p.data.copy() for i, p in enumerate(model.parameters())}
    payload = {
        "backend": "core",
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "n_embd": 128,
            "state_dict": state_dict,
        },
        "tokenizer": tokenizer.to_dict(),
        "metrics": {
            "start_loss": losses[0],
            "end_loss": losses[-1],
        },
    }
    with open(output, "wb") as f:
        pickle.dump(payload, f)

    print("✅ core 训练完成")
    print(f"   start_loss={losses[0]:.4f}, end_loss={losses[-1]:.4f}")
    print(f"   checkpoint={output}")


def main():
    parser = argparse.ArgumentParser(description="core backend training")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--output", type=str, default="checkpoints/model_core.pkl")
    parser.add_argument("--checkpoint", type=str, default="", help="保留参数兼容，当前未使用")
    args = parser.parse_args()

    train_core(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output=args.output,
    )


if __name__ == "__main__":
    main()
