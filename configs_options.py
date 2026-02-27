"""
优化版配置文件示例
=================

根据不同优化目标，提供多个配置选项。
使用方法：复制对应的配置到 config.py
"""

# ============================================================
# 选项1：继续当前训练（推荐开始）⭐⭐⭐⭐⭐
# ============================================================
"""
目标：从当前checkpoint继续，降低损失
操作：不需要修改config.py，直接 make train
预期：损失从8.38降到7.0左右
时间：1-2小时（10000-20000步）
"""


# ============================================================
# 选项2：优化训练效率
# ============================================================
class ModelConfig_Option2:
    """模型配置 - 保持不变"""
    vocab_size = 50257
    n_layer = 6
    n_head = 6
    n_embd = 384
    block_size = 512
    dropout = 0.1
    bias = True

class TrainConfig_Option2:
    """训练配置 - 优化版"""
    # 数据（启用清洗）
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    clean_data = True  # ⭐ 新增：清洗数据
    
    # 训练参数（延长训练）
    batch_size = 16
    max_iters = 20000      # ⬆️ 10000 → 20000
    eval_interval = 500
    eval_iters = 100
    log_interval = 10
    
    # 优化器
    learning_rate = 3e-4
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    
    # 学习率调度
    warmup_iters = 100
    lr_decay_iters = 20000  # ⬆️ 10000 → 20000
    min_lr = 3e-5
    
    # 系统
    device = "cuda"
    compile = True  # ⭐ 启用编译加速
    
    # 日志和保存
    out_dir = "checkpoints"
    save_interval = 1000


# ============================================================
# 选项3：使用更大数据集
# ============================================================
class TrainConfig_Option3:
    """训练配置 - 大数据集"""
    # 数据
    dataset_name = "wikitext"
    dataset_config = "wikitext-103-raw-v1"  # ⭐ 100倍数据量
    clean_data = True
    
    # 训练参数
    batch_size = 16
    max_iters = 50000      # ⬆️ 需要更多步数
    eval_interval = 1000
    eval_iters = 200
    log_interval = 10
    
    # 优化器
    learning_rate = 3e-4
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    
    # 学习率调度
    warmup_iters = 500     # ⬆️ 更长warmup
    lr_decay_iters = 50000
    min_lr = 3e-5
    
    # 系统
    device = "cuda"
    compile = True
    
    # 日志和保存
    out_dir = "checkpoints"
    save_interval = 2000


# ============================================================
# 选项4：中等模型（117M参数）
# ============================================================
class ModelConfig_Option4:
    """模型配置 - 中等规模"""
    vocab_size = 50257
    n_layer = 12       # ⬆️ 6 → 12
    n_head = 12        # ⬆️ 6 → 12
    n_embd = 768       # ⬆️ 384 → 768
    block_size = 1024  # ⬆️ 512 → 1024
    dropout = 0.1
    bias = True

class TrainConfig_Option4:
    """训练配置 - 适配中等模型"""
    # 数据
    dataset_name = "wikitext"
    dataset_config = "wikitext-103-raw-v1"  # 大数据集
    clean_data = True
    
    # 训练参数
    batch_size = 8         # ⬇️ 16 → 8（内存限制）
    max_iters = 100000     # ⬆️ 更大模型需要更多步数
    eval_interval = 1000
    eval_iters = 200
    log_interval = 10
    
    # 优化器
    learning_rate = 3e-4
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    
    # 学习率调度
    warmup_iters = 2000    # ⬆️ 更长warmup
    lr_decay_iters = 100000
    min_lr = 3e-5
    
    # 系统
    device = "cuda"
    compile = True
    
    # 日志和保存
    out_dir = "checkpoints"
    save_interval = 5000


# ============================================================
# 选项5：大模型（345M参数）- 需要强力GPU
# ============================================================
class ModelConfig_Option5:
    """模型配置 - 大规模"""
    vocab_size = 50257
    n_layer = 24       # ⬆️ 6 → 24
    n_head = 16        # ⬆️ 6 → 16
    n_embd = 1024      # ⬆️ 384 → 1024
    block_size = 1024
    dropout = 0.1
    bias = True

class TrainConfig_Option5:
    """训练配置 - 适配大模型"""
    # 数据
    dataset_name = "wikitext"
    dataset_config = "wikitext-103-raw-v1"
    clean_data = True
    
    # 训练参数
    batch_size = 4         # ⬇️ 需要大显存
    max_iters = 200000     # ⬆️ 大量训练步数
    eval_interval = 2000
    eval_iters = 200
    log_interval = 10
    
    # 优化器
    learning_rate = 2.5e-4  # ⬇️ 稍微降低
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    
    # 学习率调度
    warmup_iters = 5000    # ⬆️ 长warmup
    lr_decay_iters = 200000
    min_lr = 2.5e-5
    
    # 系统
    device = "cuda"
    compile = True
    
    # 日志和保存
    out_dir = "checkpoints"
    save_interval = 10000


# ============================================================
# 选项6：快速实验（小模型，快速迭代）
# ============================================================
class ModelConfig_Option6:
    """模型配置 - 超小模型"""
    vocab_size = 50257
    n_layer = 4        # ⬇️ 6 → 4
    n_head = 4         # ⬇️ 6 → 4
    n_embd = 256       # ⬇️ 384 → 256
    block_size = 256   # ⬇️ 512 → 256
    dropout = 0.1
    bias = True

class TrainConfig_Option6:
    """训练配置 - 快速实验"""
    # 数据
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    clean_data = True
    
    # 训练参数
    batch_size = 32        # ⬆️ 可以用更大batch
    max_iters = 5000       # ⬇️ 快速完成
    eval_interval = 250
    eval_iters = 50
    log_interval = 10
    
    # 优化器
    learning_rate = 6e-4   # ⬆️ 更高学习率
    weight_decay = 0.05    # ⬇️ 更少正则化
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    
    # 学习率调度
    warmup_iters = 100
    lr_decay_iters = 5000
    min_lr = 6e-5
    
    # 系统
    device = "cuda"
    compile = True
    
    # 日志和保存
    out_dir = "checkpoints"
    save_interval = 500


# ============================================================
# 使用说明
# ============================================================
"""
如何应用这些配置：

1. 复制你选择的配置类到 config.py
2. 重命名为 ModelConfig 和 TrainConfig
3. 如果要从头训练，删除旧checkpoint：
   rm checkpoints/best_model.pt
4. 开始训练：
   make train

推荐路线：
- 新手/快速测试：选项6 (15分钟看到结果)
- 当前状态继续：选项2 (清洗数据+延长训练)
- 想要更好效果：选项3 (大数据集)
- 追求高质量：选项4 (中等模型，需要几小时)
- 研究级别：选项5 (大模型，需要GPU+数天)

资源要求对照：
选项1: CPU可运行，1-2小时
选项2: CPU可运行，2-4小时
选项3: 建议GPU，5-10小时
选项4: 需要GPU(8GB+)，1-2天
选项5: 需要强力GPU(16GB+)，3-7天
选项6: CPU可运行，15-30分钟
"""


# ============================================================
# 参数数量对照
# ============================================================
"""
当前（30M）:  n_layer=6,  n_embd=384,  params≈30M
选项6（10M）: n_layer=4,  n_embd=256,  params≈10M
选项4（117M）:n_layer=12, n_embd=768,  params≈117M
选项5（345M）:n_layer=24, n_embd=1024, params≈345M

对比参考：
GPT-2 Small: 117M
GPT-2 Medium: 345M
GPT-2 Large: 774M
GPT-2 XL: 1.5B
"""
