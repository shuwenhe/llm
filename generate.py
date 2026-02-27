"""文本生成脚本"""
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


def generate_text(prompt, model, tokenizer, device, max_new_tokens=100, temperature=0.8, top_k=40):
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
            top_k=top_k
        )
    
    # 解码
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
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
    
    # 交互式生成
    print("\n" + "="*50)
    print("文本生成器 (输入 'quit' 退出)")
    print("="*50)
    
    while True:
        prompt = input("\n请输入提示词: ")
        if prompt.lower() == 'quit':
            break
        
        if not prompt.strip():
            continue
        
        # 生成文本
        generated = generate_text(
            prompt,
            model,
            tokenizer,
            device,
            max_new_tokens=128,
            temperature=0.8,
            top_k=40
        )
        
        print(f"\n生成结果:\n{generated}\n")
        print("-"*50)


if __name__ == "__main__":
    main()
