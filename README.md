# LLM 工业级训练系统

一个完整的、OpenAI风格的、工业级的大语言模型训练系统。

## 🚀 快速开始

**👋 首次使用？从这里开始：** [docs/START_HERE.md](docs/START_HERE.md)

```bash
python train_cli.py --preset quick
```

## 🌟 核心特性

- 🔧 **完整实现**: 从数据处理到模型训练的完整流程
- 🎯 **GPT架构**: 基于Transformer的自回归语言模型
- 📊 **可配置**: 灵活的模型和训练配置
- 🚀 **易用**: 简洁的API和清晰的代码结构
- 💡 **教育性**: 注释详尽，适合学习
- ⭐ **工业级**: OpenAI风格的命令行 + 完整文档

## 📁 项目结构

```
llm/
├─ 核心文件
├─ config.py          # 模型和训练配置
├─ model.py           # GPT模型实现
├─ data.py            # 数据加载和处理
├─ train.py           # 训练脚本
├─ generate.py        # 文本生成脚本
├─ test_model.py      # 模型测试
├─ requirements.txt   # 依赖程序
├─ Makefile           # Make加速命令
├─ setup.sh           # 自动设置脚本
├─ .gitignore         # Git忽略配置
├─ README.md          # 项目轻转
├─
├─ 文档
├─ docs/              # 详细文档
│  ├─ INSTALL.md       # 安装指南（横跨 Linux/macOS/Windows）
│  ├─ MATHEMATICS.md    # 数学知识分析
│  └─ VENV_ISSUE.md     # 虚拟环境问题解决
├─
├─ 数据与模型
├─ checkpoints/      # 模型下载位置（自动创建）
├─ data/              # 数据下载位置（自动创建）
├─ logs/              # 训练日妋（自动创建）
```

## 🚀 快速开始
> 💡 **遇到安装问题？** 查看详细的 [安装指南 (INSTALL.md)](docs/INSTALL.md)


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

# 完整多模态训练
make train-multimodal

# 启动工业化推理API
make serve

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
| `make train-multimodal` | 开始完整多模态训练（文本+图像+语音） |
| `make serve` | 启动推理API服务（生产模式） |
| `make serve-dev` | 启动推理API服务（开发模式） |
| `make generate` | 运行文本生成 |
| `make quick-test` | 快速测试（验证模型可用） |

## 🏭 工业化接口

服务启动后提供标准健康检查与推理端点：

- `GET /healthz`：进程健康检查
- `GET /readyz`：模型就绪检查
- `GET /metrics`：Prometheus监控指标
- `POST /v1/generate`：文本生成接口

### 安全与限流（生产建议）

- `LLM_API_KEYS`：逗号分隔的API Key列表，设置后 `/v1/generate` 必须携带请求头 `X-API-Key`
- `LLM_USERS`：OAuth2账号密码，格式 `user1:pass1,user2:pass2`
- `LLM_JWT_SECRET`：JWT签名密钥（生产必须修改）
- `LLM_JWT_EXPIRE_MINUTES`：JWT有效期（分钟）
- `LLM_RATE_LIMIT_RPM`：每分钟每个调用方限流（0表示关闭）
- `LLM_LOG_LEVEL`：日志级别（默认 `INFO`）
- `LLM_SESSION_DB`：会话SQLite文件路径（默认 `sessions.db`）

示例：

```bash
export LLM_API_KEYS="prod-key-1,prod-key-2"
export LLM_USERS="admin:admin123"
export LLM_JWT_SECRET="replace-with-strong-secret"
export LLM_RATE_LIMIT_RPM=60
make serve
```

OAuth2 获取 token：

```bash
curl -X POST http://127.0.0.1:8000/oauth/token \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=admin&password=admin123"
```

使用 Bearer Token 请求：

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <access_token>" \
    -d '{"prompt":"Hello","max_new_tokens":64,"session_id":"demo-session"}'
```

会话接口：

- `GET /v1/sessions/{session_id}`：读取历史会话
- `DELETE /v1/sessions/{session_id}`：删除历史会话

请求示例（带鉴权）：

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
    -H "Content-Type: application/json" \
    -H "X-API-Key: prod-key-1" \
    -d '{"prompt":"Hello","max_new_tokens":64}'
```

示例请求：

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Hello","max_new_tokens":64}'
```

### 容器化部署

```bash
docker build -t my-llm:latest .
docker run --rm -p 8000:8000 -e LLM_CHECKPOINT=checkpoints/best_model.pt my-llm:latest
```

### Prometheus + Grafana

```bash
make obs-up
```

- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000 （默认 admin/admin）

停止：

```bash
make obs-down
```

### CI/CD（镜像构建与自动发布）

已提供 GitHub Actions 工作流：[.github/workflows/cicd.yml](.github/workflows/cicd.yml)

- PR/Push 自动执行测试
- Push 到主分支或 tag 自动构建并推送镜像到 GHCR
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

### 📖 完整文档系统

**👋 首次使用？请从这里开始：** [**docs/START_HERE.md**](docs/START_HERE.md)

所有项目文档都在 **docs/** 目录。以下是主要文档：

**快速参考**:
- 🚀 [START_HERE.md](docs/START_HERE.md) - 新用户入门指南
- ⚡ [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - 1页速查卡
- 📋 [CHEATSHEET.md](docs/CHEATSHEET.md) - 核心命令速记
- 📖 [commands_reference.md](docs/commands_reference.md) - 完整命令参考

**深度学习**:
- 🔍 [checkpoint_system.md](docs/checkpoint_system.md) - 检查点系统详解
- 📊 [training_visualization.md](docs/training_visualization.md) - 实时训练监控
- 🏭 [openai_training_guide.md](docs/openai_training_guide.md) - 工业级训练指南
- ⚖️ [openai_vs_local_comparison.md](docs/openai_vs_local_comparison.md) - OpenAI对比分析

**更新文档**:
- 📝 [UPDATE_SUMMARY.md](docs/UPDATE_SUMMARY.md) - 完整更新说明
- 🎯 [CHECKPOINT_UPDATE.md](docs/CHECKPOINT_UPDATE.md) - 检查点改进说明
- 📊 [BEFORE_AND_AFTER.md](docs/BEFORE_AND_AFTER.md) - 改进前后对比

**文档导航**:
- 🗺️ [README_DOCS.md](docs/README_DOCS.md) - 文档导航和索引
- 📚 [TRAINING_README.md](docs/TRAINING_README.md) - 训练系统完整说明

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