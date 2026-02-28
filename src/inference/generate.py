"""文本生成脚本"""
import re
import torch
from model import GPT
from config import ModelConfig
from data import load_tokenizer


def load_model(checkpoint_path):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建模型配置
    model_config = ModelConfig(**checkpoint['model_config'])
    
    # 创建模型并加载权重
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    
    return model, model_config


def post_process_text(text):
    """轻量后处理：清理维基标记与多余符号"""
    text = text.replace(' @-@ ', '-')
    text = text.replace('@-@', '-')
    text = text.replace(' @,@ ', ',')
    text = text.replace('@,@', ',')
    text = text.replace(' @.@ ', '.')
    text = text.replace('@.@', '.')
    text = re.sub(r'(\s*=\s*){2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_text(prompt, model, tokenizer, device, max_new_tokens=100, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1):
    """生成文本"""
    model.eval()
    model.to(device)
    
    # 编码输入
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\n提示词: {prompt}")
    print(f"生成中...\n")
    
    # 生成
    with torch.no_grad():
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    
    # 解码
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    generated_text = post_process_text(generated_text)
    return generated_text


def main():
    """主函数"""
    # 配置
    checkpoint_path = "checkpoints/best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"加载模型: {checkpoint_path}")
    
    # 加载模型和分词器
    model, config = load_model(checkpoint_path)
    tokenizer = load_tokenizer()
    
    print(f"模型参数量: {model.get_num_params()/1e6:.2f}M")
    
    # 预设参数
    presets = {
        '1': {'name': '保守模式', 'temp': 0.65, 'top_k': 40, 'top_p': 0.85, 'rp': 1.15, 'tokens': 120},
        '2': {'name': '平衡模式', 'temp': 0.75, 'top_k': 120, 'top_p': 0.9, 'rp': 1.1, 'tokens': 180},
        '3': {'name': '创意模式', 'temp': 0.9, 'top_k': 200, 'top_p': 0.95, 'rp': 1.05, 'tokens': 240},
    }
    
    # 交互式生成
    print("\n" + "="*50)
    print("文本生成器 (输入 'quit' 退出)")
    print("="*50)
    print("\n📝 生成模式:")
    for k, v in presets.items():
        print(f"  {k}. {v['name']} (temperature={v['temp']}, top_k={v['top_k']}, top_p={v['top_p']}, repetition_penalty={v['rp']}, tokens={v['tokens']})")
    print("\n💡 提示词示例:")
    print("  - Once upon a time")
    print("  - The meaning of life is")
    print("  - In a world where")
    
    # 默认使用平衡模式
    current_preset = presets['2']
    
    while True:
        prompt = input("\n请输入提示词 (或输入1/2/3切换模式): ")
        
        if prompt.lower() == 'quit':
            break
        
        # 切换模式
        if prompt in presets:
            current_preset = presets[prompt]
            print(f"✓ 已切换到: {current_preset['name']}")
            continue
        
        if not prompt.strip():
            continue
        
        # 生成文本
        generated = generate_text(
            prompt,
            model,
            tokenizer,
            device,
            max_new_tokens=current_preset['tokens'],
            temperature=current_preset['temp'],
            top_k=current_preset['top_k'],
            top_p=current_preset['top_p'],
            repetition_penalty=current_preset['rp']
        )
        
        print(f"\n生成结果 [{current_preset['name']}]:\n{generated}\n")
        print("-"*50)


if __name__ == "__main__":
    main()
