"""core 快速文本生成测试实现"""

import os
import pickle

import numpy as np

from app.core.models import TinyLM
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
    model = TinyLM(vocab_size=payload["model"]["vocab_size"], n_embd=payload["model"]["n_embd"])
    for i, p in enumerate(model.parameters()):
        p.data[...] = payload["model"]["state_dict"][f"param_{i}"]

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

            for _ in range(cfg["tokens"]):
                x = np.array([ids], dtype=np.int64)
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
