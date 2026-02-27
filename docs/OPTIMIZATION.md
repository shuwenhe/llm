"""
LLM 优化指南
==========

根据当前训练情况（验证损失 8.38，10000步），这里提供系统的优化方案。
"""

# ============================================================
# 🎯 优先级1: 继续训练（最快见效）
# ============================================================

## 方案 1A: 直接继续训练（推荐⭐⭐⭐⭐⭐）
"""
问题：验证损失 8.38 太高，模型还未收敛
目标：降低到 6-7（WikiText-2 合理范围）
"""

# 操作步骤：
"""
1. 直接运行：
   make train
   
2. 训练会自动从 checkpoints/best_model.pt 继续
   
3. 再训练 5000-10000 步，观察损失变化

4. 预期结果：
   - 10000-15000步: 损失降到 7.5-8.0
   - 15000-20000步: 损失降到 6.5-7.5
   - 20000+步: 损失稳定在 6.0-7.0
"""


## 方案 1B: 延长训练（如果有时间）
"""
修改 config.py:
"""
class TrainConfig:
    max_iters = 20000        # 10000 → 20000
    lr_decay_iters = 20000   # 10000 → 20000
    
"""
然后重新训练（会覆盖之前的checkpoint）
"""


# ============================================================
# 🔧 优先级2: 调整训练配置（提升效果）
# ============================================================

## 方案 2A: 增大批量大小
"""
修改 config.py:
"""
class TrainConfig:
    batch_size = 32  # 16 → 32（如果显存够）
    # 或使用梯度累积
    gradient_accumulation_steps = 2  # 模拟batch_size=32

"""
优点：
- 更稳定的梯度估计
- 可能更快收敛

缺点：
- 需要更多显存/内存
"""


## 方案 2B: 调整学习率
"""
修改 config.py:
"""
class TrainConfig:
    learning_rate = 6e-4     # 3e-4 → 6e-4（加速早期训练）
    warmup_iters = 200       # 100 → 200（更平滑）
    min_lr = 6e-5            # 3e-5 → 6e-5（保持学习能力）

"""
适用场景：训练进展太慢
"""


## 方案 2C: 减少正则化
"""
修改 config.py:
"""
class ModelConfig:
    dropout = 0.05  # 0.1 → 0.05（减少dropout）

class TrainConfig:
    weight_decay = 0.05  # 0.1 → 0.05（减少权重衰减）

"""
适用场景：损失下降太慢，可能是正则化过强
"""


# ============================================================
# 📊 优先级3: 数据优化（提升质量）
# ============================================================

## 方案 3A: 清洗 WikiText 数据
"""
问题：WikiText-2 包含维基百科格式标记（@-@, = = =）

解决方案：在 data.py 中添加清洗函数
"""

def clean_wikitext(text):
    """清洗维基百科格式"""
    import re
    
    # 移除维基标记
    text = text.replace('@-@', '-')
    text = text.replace('@,@', ',')
    text = text.replace('@.@', '.')
    
    # 移除标题分隔符
    text = re.sub(r'\s*=+\s*', '\n', text)
    
    # 移除多余空白
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

# 在 prepare_data() 中使用
def prepare_data(config):
    dataset = load_dataset(...)
    
    # 添加清洗步骤
    train_text = clean_wikitext(dataset['train']['text'])
    val_text = clean_wikitext(dataset['validation']['text'])
    
    ...


## 方案 3B: 使用更好的数据集
"""
WikiText-2 较小（~2MB），考虑：
"""

# 选项1：更大的 WikiText（推荐）
class TrainConfig:
    dataset_name = "wikitext"
    dataset_config = "wikitext-103-raw-v1"  # 100倍数据量

# 选项2：图书语料（更自然的文本）
class TrainConfig:
    dataset_name = "bookcorpus"  # 需要申请访问

# 选项3：通用爬虫数据
class TrainConfig:
    dataset_name = "c4"
    dataset_config = "en"


## 方案 3C: 使用自定义数据
"""
准备 .txt 文件，修改 data.py:
"""

def prepare_data(config):
    # 读取本地文件
    with open('my_training_data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 分割训练/验证集
    n = len(text)
    train_text = text[:int(n*0.9)]
    val_text = text[int(n*0.9):]
    
    # 其余逻辑不变
    ...


# ============================================================
# 🚀 优先级4: 扩大模型规模（大幅提升）
# ============================================================

## 方案 4A: 中等模型（117M参数）
"""
修改 config.py:
"""
class ModelConfig:
    n_layer = 12   # 6 → 12
    n_head = 12    # 6 → 12
    n_embd = 768   # 384 → 768
    block_size = 1024  # 512 → 1024

class TrainConfig:
    batch_size = 8       # 16 → 8（减小以适应内存）
    max_iters = 50000    # 更大模型需要更多训练
    
"""
预期效果：生成质量显著提升
所需时间：3-5倍
所需内存：4-6倍
"""


## 方案 4B: 大模型（345M参数）
"""
修改 config.py:
"""
class ModelConfig:
    n_layer = 24    # 6 → 24
    n_head = 16     # 6 → 16
    n_embd = 1024   # 384 → 1024
    block_size = 1024

class TrainConfig:
    batch_size = 4       # 需要很大显存
    max_iters = 100000
    
"""
注意：需要 GPU 和大量时间
"""


# ============================================================
# ⚡ 优先级5: 生成优化（快速改善输出）
# ============================================================

## 方案 5A: 调整采样策略
"""
已实现在 generate.py，可以尝试：
"""

# 保守策略（更连贯）
temperature = 0.5  # 降低随机性
top_k = 20         # 只从最可能的20个词中选

# 平衡策略
temperature = 0.8
top_k = 200

# 添加 top-p (nucleus sampling)
def generate_with_top_p(logits, top_p=0.9):
    """
    只从累积概率达到 top_p 的词中采样
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 移除累积概率超过阈值的token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    
    return logits


## 方案 5B: 提示词工程
"""
使用更好的提示词：
"""

# ❌ 不好的提示词
"Once upon a time"  # 太通用，模型不知道往哪个方向

# ✅ 更好的提示词
"Once upon a time, in a small village near the mountains, there lived"
"The scientist discovered that the mysterious signal was coming from"
"Chapter 1: The Beginning\n\nJohn woke up and realized"

# 技巧：
# 1. 提供更多上下文
# 2. 指定风格/领域
# 3. 使用完整句子开头


## 方案 5C: 后处理
"""
在生成后清理文本：
"""

def post_process_text(text):
    """后处理生成的文本"""
    import re
    
    # 移除重复符号
    text = re.sub(r'(=\s*){3,}', '', text)
    text = re.sub(r'(@-@\s*)+', '-', text)
    
    # 移除不完整的句子（以句号结尾）
    sentences = text.split('.')
    if len(sentences) > 1:
        text = '.'.join(sentences[:-1]) + '.'
    
    # 清理空白
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


# ============================================================
# 📈 推荐的优化路线（按顺序执行）
# ============================================================

"""
第1步：立即继续训练（优先级1）⭐⭐⭐⭐⭐
--------------------------------------
操作：make train
时间：1-2小时
预期：损失降到 7.0 左右


第2步：清洗数据（优先级3A）⭐⭐⭐⭐
----------------------------------
操作：实现 clean_wikitext 函数
时间：10分钟
效果：显著减少格式标记


第3步：调整生成参数（优先级5A）⭐⭐⭐
------------------------------------
操作：尝试不同 temperature 和 top_k
时间：5分钟
效果：立即看到输出改善


第4步（可选）：使用更大数据集（优先级3B）⭐⭐
-----------------------------------------------
操作：切换到 wikitext-103
时间：重新训练 3-5小时
效果：更好的泛化能力


第5步（可选）：扩大模型（优先级4A）⭐
------------------------------------
操作：升级到 117M 参数
时间：重新训练 5-10小时
效果：质量大幅提升
"""


# ============================================================
# 🛠️ 快速实操指令
# ============================================================

"""
# 方案1：最简单（5分钟见效）
make train  # 继续训练5000步

# 方案2：添加数据清洗（15分钟）
# 1. 编辑 data.py，添加 clean_wikitext()
# 2. 在 prepare_data() 中调用
# 3. 重新训练

# 方案3：使用更大数据集（3小时）
# 1. 修改 config.py: dataset_config = "wikitext-103-raw-v1"
# 2. 删除旧 checkpoint: rm checkpoints/best_model.pt
# 3. 重新训练: make train

# 方案4：扩大模型（5-10小时）
# 1. 修改 config.py 的 ModelConfig（参数见上面）
# 2. 删除旧 checkpoint
# 3. 重新训练
"""


# ============================================================
# 📊 预期改善对照表
# ============================================================

"""
当前状态：
- 损失: 8.38
- 质量: 碎片化，格式标记多，不连贯

继续训练到20000步（方案1）：
- 损失: 6.5-7.5
- 质量: 基本连贯，仍有较多格式标记

+数据清洗（方案2）：
- 损失: 6.5-7.5
- 质量: 连贯，格式标记大幅减少 ✅

+调整采样（方案3）：
- 损失: 6.5-7.5
- 质量: 更流畅，减少重复 ✅✅

+更大数据集（方案4）：
- 损失: 5.5-6.5
- 质量: 更自然，词汇更丰富 ✅✅✅

+扩大模型到117M（方案5）：
- 损失: 4.5-5.5
- 质量: 接近可用水平 ✅✅✅✅
"""


# ============================================================
# ��� 常见问题
# ============================================================

"""
Q: 训练多久才能看到明显改善？
A: 每5000步检查一次，损失每降低0.5-1.0，质量就会有明显提升

Q: 显存/内存不够怎么办？
A: 1) 减小 batch_size
   2) 减小 block_size
   3) 使用梯度累积

Q: 训练太慢怎么办？
A: 1) 使用 GPU
   2) 启用 compile=True（PyTorch 2.0+）
   3) 考虑使用云GPU（Colab, AWS等）

Q: 什么时候该停止训练？
A: 1) 验证损失不再下降（过拟合）
   2) 生成质量满意
   3) 达到时间/资源限制

Q: 先优化什么？
A: 按顺序：继续训练 → 清洗数据 → 调整采样 → 大数据集 → 大模型
"""
