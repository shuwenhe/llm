"""快速文本生成测试脚本"""
import torch
from model import GPT
from config import ModelConfig
from data import load_tokenizer


def quick_test():
    """快速测试不同参数的生成效果"""
    # 配置
    checkpoint_path = "checkpoints/best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"🚀 快速生成测试")
    print(f"设备: {device}")
    print("="*60)
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint['model_config'])
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    
    tokenizer = load_tokenizer()
    
    # 测试提示词
    test_prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In a world where",
    ]
    
    # 测试参数组合
    test_configs = [
        {'name': '保守模式', 'temp': 0.7, 'top_k': 50},
        {'name': '平衡模式', 'temp': 0.8, 'top_k': 200},
        {'name': '创意模式', 'temp': 1.0, 'top_k': 300},
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"📝 提示词: \"{prompt}\"")
        print(f"{'='*60}")
        
        for cfg in test_configs:
            print(f"\n🔧 {cfg['name']} (temp={cfg['temp']}, top_k={cfg['top_k']})")
            print("-"*60)
            
            # 编码
            tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # 生成
            with torch.no_grad():
                generated_tokens = model.generate(
                    tokens,
                    max_new_tokens=150,
                    temperature=cfg['temp'],
                    top_k=cfg['top_k']
                )
            
            # 解码
            generated_text = tokenizer.decode(generated_tokens[0].tolist())
            print(generated_text)
    
    print("\n" + "="*60)
    print("✅ 测试完成！选择效果最好的参数在 generate.py 中使用")


if __name__ == "__main__":
    quick_test()
