"""core 文本生成实现"""

import os
import pickle

import numpy as np

from app.core.models import TinyLM
from app.core.tokenizer import CharTokenizer


def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ 模型检查点文件不存在: {checkpoint_path}\n\n"
            f"请先训练模型:\n"
            f"  make train-core\n"
            f"  make train-chinese\n"
        )

    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)

    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    model = TinyLM(vocab_size=payload["model"]["vocab_size"], n_embd=payload["model"]["n_embd"])
    for i, p in enumerate(model.parameters()):
        p.data[...] = payload["model"]["state_dict"][f"param_{i}"]
    return model, tokenizer


def generate_text(prompt, model, tokenizer, max_new_tokens=120, temperature=0.8):
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]

    for _ in range(max_new_tokens):
        x = np.array([ids], dtype=np.int64)
        logits, _ = model(x, None)
        next_logits = logits.data[0, -1] / max(temperature, 1e-6)
        next_logits = next_logits - np.max(next_logits)
        probs = np.exp(next_logits)
        probs = probs / (probs.sum() + 1e-12)
        next_id = int(np.random.choice(len(probs), p=probs))
        ids.append(next_id)

    generated_text = tokenizer.decode(ids)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    return generated_text.strip()


def main():
    checkpoint_path = os.getenv("LLM_CHECKPOINT", "checkpoints/model_core.pkl")
    print(f"加载 core 模型: {checkpoint_path}")

    model, tokenizer = load_model(checkpoint_path)
    print(f"模型参数量: {sum(p.data.size for p in model.parameters())/1e6:.2f}M")

    presets = {
        "1": {"name": "保守模式", "temp": 0.65, "tokens": 80},
        "2": {"name": "平衡模式", "temp": 0.80, "tokens": 120},
        "3": {"name": "创意模式", "temp": 1.00, "tokens": 160},
    }

    print("\n" + "=" * 50)
    print("文本生成器(core) (输入 'quit' 退出)")
    print("=" * 50)
    current_preset = presets["2"]

    while True:
        prompt = input("\n请输入提示词 (或输入1/2/3切换模式): ")
        if prompt.lower() == "quit":
            break
        if prompt in presets:
            current_preset = presets[prompt]
            print(f"✓ 已切换到: {current_preset['name']}")
            continue
        if not prompt.strip():
            continue

        generated = generate_text(
            prompt,
            model,
            tokenizer,
            max_new_tokens=current_preset["tokens"],
            temperature=current_preset["temp"],
        )
        print(f"\n生成结果 [{current_preset['name']}]:\n{generated}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
