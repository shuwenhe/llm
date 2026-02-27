# 从零开发大语言模型 (LLM)

一个从零实现的GPT风格大语言模型训练项目，包含完整的训练、评估和生成功能。

## 🌟 特性

- 🔧 **完整实现**: 从数据处理到模型训练的完整流程
- 🎯 **GPT架构**: 基于Transformer的自回归语言模型
- 📊 **可配置**: 灵活的模型和训练配置
- 🚀 **易用**: 简洁的API和清晰的代码结构
- 💡 **教育性**: 注释详尽，适合学习

## 📁 项目结构

```
llm/
├── config.py          # 模型和训练配置
├── model.py           # GPT模型实现
├── data.py            # 数据加载和处理
├── train.py           # 训练脚本
├── generate.py        # 文本生成脚本
├── requirements.txt   # 依赖包
└── checkpoints/       # 模型保存目录（自动创建）
```

## 🚀 快速开始
> 💡 **遇到安装问题？** 查看详细的 [安装指南 (INSTALL.md)](INSTALL.md)


### 方式一：使用 Makefile（推荐）

```bash
# 一键设置（创建虚拟环境并安装所有依赖）
make setup-all

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 测试模型
make test

# 训练模型
make train

# 生成文本
make generate
```

**或者分步执行：**

```bash
# 1. 创建虚拟环境
make setup

# 2. 激活虚拟环境
source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
make install

# 4. 查看所有可用命令
make help
```

### 方式二：直接运行Python脚本

#### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 2. 训练模型

```bash
python train.py
```

训练将使用WikiText-2数据集，模型checkpoint会保存在 `checkpoints/` 目录。

#### 3. 生成文本

训练完成后，使用以下命令进行文本生成：

```bash
python generate.py
```

## 🔧 Makefile 命令参考

| 命令 | 说明 |
|------|------|
| `make help` | 显示所有可用命令 |
| **环境设置** | |
| `make setup-all` | ⭐ 一键设置（创建虚拟环境+安装依赖） |
| `make setup` | 仅创建虚拟环境 |
| `make install` | 安装依赖（需要先激活虚拟环境） |
| `make install-force` | 强制安装（跳过虚拟环境检查，不推荐） |
| **开发与训练** | |
| `make test` | 运行模型测试 |
| `make train` | 开始训练模型 |
| `make generate` | 运行文本生成 |
| `make quick-test` | 快速测试（验证模型可用） |
| **工具** | |
| `make info` | 查看模型配置信息 |
| `make check-deps` | 检查依赖安装情况 |
| `make init` | 创建必要的项目目录 |
| **清理** | |
| `make clean` | 清理Python缓存文件 |
| `make clean-checkpoints` | 删除所有checkpoint |
| `make clean-all` | 清理所有生成文件 |

## ⚙️ 配置说明

### 模型配置 (`ModelConfig`)

```python
vocab_size = 50257    # 词表大小
n_layer = 6           # Transformer层数
n_head = 6            # 注意力头数
n_embd = 384          # 嵌入维度
block_size = 512      # 最大序列长度
```

### 训练配置 (`TrainConfig`)

```python
batch_size = 16       # 批次大小
learning_rate = 3e-4  # 学习率
max_iters = 10000     # 最大训练步数
eval_interval = 500   # 评估间隔
```

## 📊 模型架构

本项目实现了基于GPT的自回归语言模型：

1. **Token Embedding + Position Embedding**
2. **多层Transformer Block**
   - 多头自注意力机制 (Multi-Head Self-Attention)
   - 前馈神经网络 (Feed-Forward Network)
   - 层归一化 (Layer Normalization)
   - 残差连接 (Residual Connections)
3. **输出层** (Language Modeling Head)

## 🎓 学习路径

1. **模型理解**: 从 [`model.py`](model.py) 开始，理解Transformer架构
2. **数据处理**: 查看 [`data.py`](data.py) 了解数据准备流程
3. **训练过程**: 阅读 [`train.py`](train.py) 学习训练循环
4. **文本生成**: 探索 [`generate.py`](generate.py) 了解推理过程

## 📈 扩展建议

### 增加模型规模

修改 `config.py` 中的参数：

```python
# 小模型（当前）
n_layer = 6
n_embd = 384
# 约 30M 参数

# 中等模型
n_layer = 12
n_embd = 768
# 约 117M 参数

# 大模型
n_layer = 24
n_embd = 1024
# 约 345M 参数
```

### 使用自定义数据

修改 [`data.py`](data.py) 中的数据加载函数，或准备自己的文本数据：

```python
# 使用本地文本文件
with open('my_data.txt', 'r') as f:
    text = f.read()
# 然后进行分词和处理
```

### 添加高级功能

- **混合精度训练**: 已支持，加速训练
- **梯度累积**: 模拟更大的batch size
- **分布式训练**: 使用DDP进行多GPU训练
- **更好的采样策略**: Top-p (nucleus) sampling
- **Checkpoint平均**: 提升模型稳定性

## 🔧 常见问题

### Q: 安装依赖时出现 "externally-managed-environment" 错误？
A: 
这是新版 Linux 系统的安全特性，需要使用虚拟环境：
```bash
# 一键解决
make setup-all

# 然后激活虚拟环境
source venv/bin/activate

# 或者分步执行
make setup              # 创建虚拟环境
source venv/bin/activate  # 激活
make install            # 安装依赖
```

### Q: 训练很慢怎么办？
A: 
- 减小模型规模或batch_size
- 使用GPU（CUDA）而不是CPU
- 启用 `torch.compile`（PyTorch 2.0+）

### Q: 内存不足怎么办？
A:
- 减小 `batch_size`
- 减小 `block_size`
- 减小模型参数（n_layer, n_embd）

### Q: 如何使用更大的数据集？
A:
- 修改 `data.py` 中的 `dataset_name` 和 `dataset_config`
- 或实现自定义数据加载器

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2论文
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy的GPT实现
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Transformer可视化讲解

## 📝 License

MIT License

## 🙏 致谢

本项目受 [nanoGPT](https://github.com/karpathy/nanoGPT) 启发，旨在提供一个清晰易懂的LLM实现。