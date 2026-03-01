"""core 快速文本生成测试实现"""

import os
import pickle

import numpy as np

from app.core.models import TinyLM, TransformerLM
from app.core.tokenizer import CharTokenizer


def quick_test():
    checkpoint_path = os.getenv("LLM_CHECKPOINT", "checkpoints/model_core.pkl")

    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型检查点文件不存在: {checkpoint_path}\n")
        print("请先训练模型:")
        print("  make train-core")
        print("  make train-chinese")
        return

    print("🚀 快速生成测试")
    print("后端: core")
    print("=" * 60)

    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    model_cfg = payload["model"]
    if all(k in model_cfg for k in ("n_layers", "n_heads", "max_seq_len")):
        model = TransformerLM(
            vocab_size=model_cfg["vocab_size"],
            n_embd=model_cfg["n_embd"],
            n_layers=model_cfg["n_layers"],
            n_heads=model_cfg["n_heads"],
            max_seq_len=model_cfg["max_seq_len"],
            dropout=model_cfg.get("dropout", 0.1),
        )
    else:
        model = TinyLM(vocab_size=model_cfg["vocab_size"], n_embd=model_cfg["n_embd"])
    state_dict = model_cfg["state_dict"]
    for i, p in enumerate(model.parameters()):
        key = f"param_{i}"
        if key not in state_dict:
            raise ValueError(f"checkpoint 缺少参数: {key}")
        src = state_dict[key]
        if p.data.shape != src.shape:
            raise ValueError(
                f"checkpoint 参数形状不匹配: {key}, src={src.shape}, dst={p.data.shape}"
            )
        p.data[...] = src

    test_prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In a world where",
    ]

    test_configs = [
        {"name": "保守模式", "temp": 0.7, "tokens": 80},
        {"name": "平衡模式", "temp": 0.8, "tokens": 120},
        {"name": "创意模式", "temp": 1.0, "tokens": 150},
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"📝 提示词: \"{prompt}\"")
        print(f"{'='*60}")

        for cfg in test_configs:
            print(f"\n🔧 {cfg['name']} (temp={cfg['temp']}, tokens={cfg['tokens']})")
            print("-" * 60)

            ids = tokenizer.encode(prompt)
            if not ids:
                ids = [0]

            max_ctx = getattr(model, "max_seq_len", None)
            for _ in range(cfg["tokens"]):
                ctx = ids[-max_ctx:] if isinstance(max_ctx, int) and max_ctx > 0 else ids
                x = np.array([ctx], dtype=np.int64)
                logits, _ = model(x, None)
                next_logits = logits.data[0, -1] / max(cfg["temp"], 1e-6)
                next_logits = next_logits - np.max(next_logits)
                probs = np.exp(next_logits)
                probs = probs / (probs.sum() + 1e-12)
                next_id = int(np.random.choice(len(probs), p=probs))
                ids.append(next_id)

            generated_text = tokenizer.decode(ids)
            print(generated_text)

    print("\n" + "=" * 60)
    print("✅ 测试完成！选择效果最好的参数在 generate.py 中使用")


if __name__ == "__main__":
    quick_test()
