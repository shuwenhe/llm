# 🎓 OpenAI风格工业级训练系统

## 📌 最新功能

✨ **现在支持类似OpenAI的工业级训练命令！**

```bash
# 快速验证（1轮）
python train_cli.py --preset quick

# 标准训练（3轮）
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 高精度训练（20轮）
python train_cli.py --preset precision --data-file data/zh_wiki.txt
```

## 🚀 5分钟快速开始

### 1️⃣ 列出预设配置
```bash
python train_cli.py --list-presets
```

### 2️⃣ 快速测试管道
```bash
python train_cli.py --preset quick
```

### 3️⃣ 查看训练结果
```bash
python train_manager.py history
```

### 4️⃣ 开始标准训练
```bash
bash quick_start.sh standard data/zh_sample.txt
```

## 📚 完整命令指南

### 预设训练（推荐）

| 预设 | 命令 | 用途 | 时间 |
|-----|------|------|------|
| **QUICK** | `--preset quick` | 验证管道 | ~1分钟 |
| **STANDARD** | `--preset standard` | 日常使用 | ~30分钟 |
| **EXTENDED** | `--preset extended` | 长期训练 | ~2小时 |
| **PRECISION** | `--preset precision` | 高精度 | ~5小时 |

### 基础命令

```bash
# 列出所有预设
python train_cli.py --list-presets

# 快速验证
python train_cli.py --preset quick

# 标准训练
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 从latest.pt恢复
python train_cli.py --preset standard --resume

# 干运行模式（不执行）
python train_cli.py --preset precision --dry-run

# 完整参数
python train_cli.py \
  --batch-size 8 \
  --epochs 5 \
  --learning-rate 5e-5 \
  --data-file data/zh_sample.txt \
  --keep-last-n 5
```

### 配置管理

```bash
# 从配置文件加载
python train_cli.py --config config/training_standard.json

# 保存当前配置
python train_cli.py --preset precision --save-config my_config.json

# 查看配置
cat config/training_standard.json
```

### 检查点管理

```bash
# 列出所有检查点
python train_manager.py list

# 查看训练历史和损失曲线
python train_manager.py history

# 对比两个模型
python train_manager.py compare model1.pt model2.pt

# 清理旧检查点
python train_manager.py clean
```

## 🧪 实战场景

### 场景1：我想快速验证是否能训练

```bash
python train_cli.py --preset quick
```

### 场景2：我想做标准训练

```bash
bash quick_start.sh standard data/zh_sample.txt
```

### 场景3：我想中断后恢复

```bash
# 查看最后的检查点
python train_manager.py list

# 从latest恢复
python train_cli.py --preset extended --resume

# 或继续标准训练10轮
python train_cli.py --resume --epochs 10
```

### 场景4：我想对比不同配置

```bash
# 先试快速
python train_cli.py --preset quick

# 再试标准
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 对比结果
python train_manager.py compare checkpoints/best_model.pt checkpoints/model_epoch_1.pt
```

### 场景5：我想部署最优模型

```bash
# 查看训练历史
python train_manager.py history

# 使用最佳模型部署
LLM_CHECKPOINT=checkpoints/best_model.pt make serve
```

## 📊 系统架构

```
训练系统层次结构:

┌─────────────────────────────────────────┐
│   train_cli.py (OpenAI风格命令行)      │  ← 用户界面
│   - 预设管理 (quick/standard/...)      │
│   - 配置文件支持                        │
│   - 干运行模式                          │
│   - 日志记录                            │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│   train_chinese.py (核心训练脚本)       │  ← 训练执行
│   - DataLoader加载数据                  │
│   - 模型前向/反向传播                  │
│   - 多级检查点保存                      │
│   - 断点续训                            │
│   - 训练历史JSON                        │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│   train_manager.py (检查点管理)         │  ← 结果分析
│   - 列出检查点                          │
│   - 查看训练历史                        │
│   - 对比模型                            │
│   - 清理旧文件                          │
└─────────────────────────────────────────┘
```

## 📁 文件结构

```
llm/
├── train_cli.py                    # OpenAI风格命令行接口
├── train_chinese.py                # 核心训练脚本
├── train_manager.py                # 检查点管理工具
├── download_chinese_data.py         # 数据下载工具
├── quick_start.sh                  # 快速启动脚本
│
├── config/
│   ├── training_quick.json         # 快速预设
│   ├── training_standard.json      # 标准预设
│   ├── training_extended.json      # 长期预设
│   └── training_precision.json     # 高精度预设
│
├── data/
│   └── zh_sample.txt               # 示例数据 (131KB, 1050文本)
│
├── checkpoints/
│   ├── best_model.pt               # 最优模型（自动保存）
│   ├── latest.pt                   # 最新检查点（用于恢复）
│   ├── model_epoch_1.pt            # 轮次检查点
│   ├── model_epoch_2.pt
│   ├── model_epoch_3.pt
│   ├── model.pt                    # 兼容性链接
│   └── training_history.json       # 训练历史
│
├── logs/
│   └── training_20260228_*.log     # 训练日志
│
└── docs/
    ├── openai_training_guide.md    # 完整指南
    └── commands_reference.md       # 命令参考
```

## 🎓 核心特性

### 1. 预设系统（类OpenAI）
4个预定义配置，适应不同场景：
- `quick`: 1轮, bs=2（快速验证）
- `standard`: 3轮, bs=4（日常使用）
- `extended`: 10轮, bs=8（深度训练）
- `precision`: 20轮, bs=16（生产部署）

### 2. 配置管理
- JSON配置文件支持
- 命令行参数覆盖
- 配置保存和加载
- 每次训练记录配置快照

### 3. 多级检查点
- `best_model.pt`: 最低验证损失时保存
- `latest.pt`: 每轮自动保存（用于恢复）
- `model_epoch_*.pt`: 完整轮次历史
- `training_history.json`: 完整训练指标

### 4. 干运行模式
执行前验证命令：
```bash
python train_cli.py --preset precision --dry-run
```

### 5. 日志系统
- 实时终端输出
- 时间戳日志文件
- 配置JSON快照
- 完整训练历史

### 6. 断点续训
```bash
python train_cli.py --preset standard --resume
```

## ⚡ 性能指标

在15.6GB VRAM GPU上的测试结果（zh_sample.txt）：

| 预设 | 批次 | 轮数 | 时间 | 显存占用 | 状态 |
|-----|------|------|-----|---------|------|
| QUICK | 2 | 1 | ~1分钟 | ~3GB | ✅ |
| STANDARD | 4 | 3 | ~30分钟 | ~5GB | ✅ |
| EXTENDED | 8 | 10 | ~2小时 | ~8GB | ✅ |
| PRECISION | 16 | 20 | ~5小时 | ~12GB | ✅ |

## 🔧 常见命令

```bash
# 1. 查看预设
python train_cli.py --list-presets

# 2. 验证命令（不执行）
python train_cli.py --preset quick --dry-run

# 3. 快速训练
python train_cli.py --preset quick

# 4. 标准训练
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 5. 查看结果
python train_manager.py list
python train_manager.py history

# 6. 恢复训练
python train_cli.py --resume --epochs 5

# 7. 清理旧检查点
python train_manager.py clean --keep 5

# 8. 部署最优模型
LLM_CHECKPOINT=checkpoints/best_model.pt make serve
```

## 📖 文档

- **快速指南**: [openai_training_guide.md](docs/openai_training_guide.md)
- **命令参考**: [commands_reference.md](docs/commands_reference.md)
- **完整说明**: [README.md](README.md)

## 💡 最佳实践

1. **快速验证** → `python train_cli.py --preset quick`
2. **查看预设** → `python train_cli.py --list-presets`
3. **干运行验证** → `python train_cli.py --preset standard --dry-run`
4. **标准训练** → `bash quick_start.sh standard`
5. **监控日志** → `tail -f logs/training_*.log`
6. **查看结果** → `python train_manager.py history`
7. **部署模型** → `LLM_CHECKPOINT=checkpoints/best_model.pt make serve`

## 🎯 下一步

- ✅ 执行快速训练验证系统
- ✅ 尝试标准训练获得更好的模型
- ✅ 使用train_manager查看训练历史
- 📥 下载更大的中文数据集
- 🚀 部署最优模型到生产环境
- 📊 尝试超参数调优实验

---

**快速开始**: `bash quick_start.sh standard data/zh_sample.txt`
