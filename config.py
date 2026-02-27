"""模型和训练配置"""

class ModelConfig:
    """GPT模型配置"""
    # 模型架构
    vocab_size = 50257  # GPT-2 tokenizer的词表大小
    n_layer = 6  # Transformer层数（从小模型开始）
    n_head = 6   # 注意力头数
    n_embd = 384  # 嵌入维度
    
    # 序列和训练
    block_size = 512  # 最大序列长度
    dropout = 0.1
    bias = True  # 是否在Linear和LayerNorm中使用bias
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TrainConfig:
    """训练配置"""
    # 数据
    dataset_name = "wikitext"  # 或使用自己的数据
    dataset_config = "wikitext-2-raw-v1"
    clean_data = False  # 是否清洗WikiText格式标记（暂时禁用以快速收敛）
    
    # 训练参数
    batch_size = 16
    max_iters = 10000
    eval_interval = 500
    eval_iters = 100
    log_interval = 10
    
    # 优化器
    learning_rate = 1e-5      # 1e-4 → 1e-5（再降10倍）
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 0.1           # 0.5 → 0.1（更严格）
    grad_norm_warn = 5.0      # 梯度范数告警阈值
    
    # 学习率调度
    warmup_iters = 100
    lr_decay_iters = 10000
    min_lr = 1e-6
    
    # 系统
    device = "cuda"  # cuda, mps, cpu
    compile = False  # 是否使用torch.compile（需要PyTorch 2.0+）
    
    # 日志和保存
    out_dir = "checkpoints"
    save_interval = 1000
    log_to_wandb = False
    wandb_project = "my-llm"
    wandb_run_name = "gpt-small"
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
