# 📚 文档索引和导航

> **新用户？从 [CHEATSHEET.md](CHEATSHEET.md) 开始！** ⭐

## 🎯 按需求查找文档

### "我想立即开始训练"
- ⏱️ 阅读时间: 5分钟
- 📄 文档: **[CHEATSHEET.md](CHEATSHEET.md)**
- 内容: 5个核心命令 + 4个预设 + 常见场景
- 快速开始: `python train_cli.py --preset quick`

### "我想了解完整系统"
- ⏱️ 阅读时间: 10分钟
- 📄 文档: **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**
- 内容: 快速开始 + 4个工作流 + 自定义训练
- 下一步: 执行第一次训练

### "为什么檢查点没有实时显示"
- ⏱️ 阅读时间: 5分钟
- 📄 文档: **[CHECKPOINT_UPDATE.md](CHECKPOINT_UPDATE.md)**
- 内容: 本次更新详情 + 改进说明 + 验证方法
- 关键: 检查点正在生成，只是显示改进了

### "我想理解检查点系统"
- ⏱️ 阅读时间: 15分钟
- 📄 文档: **[docs/checkpoint_system.md](docs/checkpoint_system.md)**
- 内容: 3级检查点详解 + 4个实战例子 + 管理工具
- 学习: 什么时候用什么文件

### "我想实时监控训练"
- ⏱️ 阅读时间: 20分钟
- 📄 文档: **[docs/training_visualization.md](docs/training_visualization.md)**
- 内容: 分屏方案 + 监控脚本 + 故障排查
- 实战: 3种监控方式 + 完整脚本

### "我想查询特定命令"
- ⏱️ 阅读时间: 快速查询
- 📄 文档: **[docs/commands_reference.md](docs/commands_reference.md)**
- 内容: 所有命令速查 + 生产部署 + CI/CD
- 用途: 命令参考手册

### "我想用OpenAI风格"
- ⏱️ 阅读时间: 20分钟
- 📄 文档: **[docs/openai_training_guide.md](docs/openai_training_guide.md)**
- 内容: OpenAI风格使用 + 预设详解 + 高级用法
- 特点: 预设系统 + 配置文件 + 干运行

### "我想与OpenAI对标"
- ⏱️ 阅读时间: 20分钟
- 📄 文档: **[docs/openai_vs_local_comparison.md](docs/openai_vs_local_comparison.md)**
- 内容: 功能对标 + 成本分析 + 用户体验对比
- 优势: 无成本 + 100倍快 + 完全可控

### "我想看完整的功能说明"
- ⏱️ 阅读时间: 30分钟
- 📄 文档: **[TRAINING_README.md](TRAINING_README.md)**
- 内容: 完整系统说明 + 架构图 + 所有命令 + 最佳实践
- 用途: 全面理解整个系统

## 📊 按使用阶段选择

### 🟢 初级用户（第一次使用）

**推荐阅读顺序**:
1. [CHEATSHEET.md](CHEATSHEET.md) (5分钟)
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (10分钟)

**关键命令**:
```bash
# 第1步: 查看预设
python train_cli.py --list-presets

# 第2步: 快速测试
python train_cli.py --preset quick

# 第3步: 查看结果
python train_manager.py history
```

**预期结果**:
```
✓ 理解4个预设的区别
✓ 知道如何执行训练
✓ 理解基本的检查点概念
```

### 🟡 中级用户（日常使用）

**推荐阅读顺序**:
1. [docs/checkpoint_system.md](docs/checkpoint_system.md) (15分钟)
2. [docs/training_visualization.md](docs/training_visualization.md) (20分钟)
3. [docs/commands_reference.md](docs/commands_reference.md) (快速查询)

**关键任务**:
```bash
# 标准训练
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 监控进度
watch -n 1 'ls -lh checkpoints/*.pt'

# 查看结果
python train_manager.py list
python train_manager.py history

# 恢复训练
python train_cli.py --resume --epochs 5
```

**预期结果**:
```
✓ 掌握参数调优
✓ 能实时监控训练
✓ 会使用所有管理工具
✓ 理解检查点的用途
```

### 🔴 高级用户（深度理解和定制）

**推荐阅读顺序**:
1. [TRAINING_README.md](TRAINING_README.md) (30分钟)
2. [docs/openai_vs_local_comparison.md](docs/openai_vs_local_comparison.md) (20分钟)
3. [train_cli.py](train_cli.py) 源代码 (20分钟)
4. [train_chinese.py](train_chinese.py) 源代码 (20分钟)

**关键任务**:
```bash
# 自定义参数组合
python train_cli.py \
  --batch-size 8 \
  --epochs 10 \
  --learning-rate 5e-5 \
  --keep-last-n 5 \
  --save-config my_model.json

# 对比不同配置
python train_manager.py compare model1.pt model2.pt

# 在最优模型基础上微调
python train_cli.py --preset extended \
  --checkpoint checkpoints/best_model.pt \
  --learning-rate 5e-5 \
  --epochs 10
```

**预期结果**:
```
✓ 理解系统架构设计
✓ 能根据需求定制配置
✓ 理解成本和性能权衡
✓ 能进行高级优化
```

## 🔍 按特定任务查找

| 任务 | 相关文档 | 关键命令 |
|------|---------|---------|
| 快速验证系统 | CHEATSHEET.md | `python train_cli.py --preset quick` |
| 标准训练 | QUICK_REFERENCE.md | `bash quick_start.sh standard` |
| 查看检查点 | checkpoint_system.md | `python train_manager.py list` |
| 实时监控 | training_visualization.md | `watch -n 1 'ls checkpoints/*.pt'` |
| 恢复训练 | QUICK_REFERENCE.md | `python train_cli.py --resume` |
| 对比模型 | commands_reference.md | `python train_manager.py compare` |
| 生产部署 | commands_reference.md | `LLM_CHECKPOINT=... make serve` |
| 参数调优 | openai_training_guide.md | `--batch-size`, `--epochs` 等 |
| 理论背景 | openai_vs_local_comparison.md | 为什么这样设计 |

## 📖 完整文档列表

### 根目录文档（项目级）

| 文件 | 用途 | 优先级 | 长度 |
|------|------|--------|------|
| [CHEATSHEET.md](CHEATSHEET.md) | 一页速查卡 | ⭐⭐⭐ | 3页 |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | 快速开始 | ⭐⭐ | 5页 |
| [CHECKPOINT_UPDATE.md](CHECKPOINT_UPDATE.md) | 本次更新 | ⭐⭐ | 4页 |
| [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md) | 完整总结 | ⭐⭐ | 6页 |
| [TRAINING_README.md](TRAINING_README.md) | 功能说明 | ⭐ | 8页 |
| [README.md](README.md) | 项目说明 | - | 10页 |

### 详细文档（docs目录）

| 文件 | 用途 | 优先级 | 长度 |
|------|------|--------|------|
| [checkpoint_system.md](docs/checkpoint_system.md) | 检查点详解 | ⭐⭐⭐ | 8页 |
| [training_visualization.md](docs/training_visualization.md) | 监控可视化 | ⭐⭐ | 10页 |
| [commands_reference.md](docs/commands_reference.md) | 命令参考 | ⭐⭐ | 12页 |
| [openai_training_guide.md](docs/openai_training_guide.md) | OpenAI风格 | ⭐ | 10页 |
| [openai_vs_local_comparison.md](docs/openai_vs_local_comparison.md) | 对标分析 | ⭐ | 8页 |

## 🎯 快速导航卡

### 最常用的3个命令

```bash
# 1️⃣ 查看预设
python train_cli.py --list-presets

# 2️⃣ 执行标准训练
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 3️⃣ 查看结果
python train_manager.py history
```

### 最常查的4个文档

```
需要快速命令 → CHEATSHEET.md
需要快速开始 → QUICK_REFERENCE.md
需要理解原理 → checkpoint_system.md
需要完整说明 → TRAINING_README.md
```

### 最常见的3个场景

```bash
# 场景1: 我想快速验证
python train_cli.py --preset quick

# 场景2: 我想标准训练
bash quick_start.sh standard

# 场景3: 我想恢复训练
python train_cli.py --resume
```

## ✨ 文档特色

### CHEATSHEET.md
- ✅ 最简洁的快速参考
- ✅ 包含所有常用命令
- ✅ 一页纸放得下
- ✅ 适合打印或收藏

### QUICK_REFERENCE.md
- ✅ 详细的快速开始
- ✅ 4个完整的工作流
- ✅ 参数速查表
- ✅ 性能基准数据

### checkpoint_system.md
- ✅ 最详细的原理说明
- ✅ 4个实战案例
- ✅ 管理工具使用
- ✅ 故障排查指南

### training_visualization.md
- ✅ 3种实时监控方案
- ✅ 完整的监控脚本
- ✅ 训练过程演示
- ✅ 关键时刻截图

### commands_reference.md
- ✅ 完整的命令索引
- ✅ 参数详细说明
- ✅ 生产部署checklist
- ✅ CI/CD集成示例

### openai_training_guide.md
- ✅ OpenAI风格使用
- ✅ 预设系统设计
- ✅ 配置文件管理
- ✅ 高级用法展示

### openai_vs_local_comparison.md
- ✅ 功能逐项对标
- ✅ 成本效益分析
- ✅ 用户体验对比
- ✅ 系统设计哲学

## 🚀 推荐学习路径

### Path 1: 速成（30分钟）
```
CHEATSHEET.md (5分钟)
  ↓
执行 python train_cli.py --preset quick (5分钟)
  ↓
看结果 python train_manager.py history (5分钟)
  ↓
理解 QUICK_REFERENCE.md (15分钟)
```

### Path 2: 全面（2小时）
```
QUICK_REFERENCE.md (10分钟)
  ↓
checkpoint_system.md (15分钟)
  ↓
执行标准训练 (30分钟)
  ↓
training_visualization.md (15分钟)
  ↓
commands_reference.md (20分钟)
  ↓
TRAINING_README.md (30分钟)
```

### Path 3: 深度（1天）
```
所有Path 2的内容
  ↓
openai_training_guide.md (20分钟)
  ↓
openai_vs_local_comparison.md (20分钟)
  ↓
阅读源代码 train_cli.py (30分钟)
  ↓
阅读源代码 train_chinese.py (40分钟)
  ↓
自己定制扩展 (自由时间)
```

## 💡 使用建议

### ✅ 推荐做法
1. 新手先读 CHEATSHEET.md
2. 快速试验第一个命令
3. 成功后再读详细文档
4. 需要特定功能时查专项文档

### ❌ 不推荐做法
1. ❌ 一开始就读全部文档（信息过载）
2. ❌ 不实践直接读源代码（难以理解）
3. ❌ 跳过快速开始直接用高级功能

## 🔗 文档间的联系

```
开始学习
  ↓
[CHEATSHEET.md] ← 5个核心命令
  ↓
执行第一次训练
  ↓
[QUICK_REFERENCE.md] ← 理解工作流
  ↓
遇到问题？
  ├─ 想看检查点 → [checkpoint_system.md]
  ├─ 想监控训练 → [training_visualization.md]
  ├─ 想查命令 → [commands_reference.md]
  └─ 想深入理解 → [TRAINING_README.md]
```

## 📱 文档快速访问

### 浏览器书签建议
```
文档导航 (README_DOCS.md) [当前页面]
├─ 速查卡 (CHEATSHEET.md)
├─ 快速开始 (QUICK_REFERENCE.md)
├─ 检查点系统 (checkpoint_system.md)
├─ 实时监控 (training_visualization.md)
└─ 命令参考 (commands_reference.md)
```

### 终端快速访问
```bash
# 查看速查卡
cat CHEATSHEET.md | less

# 查看快速开始
cat QUICK_REFERENCE.md | less

# 搜索特定内容
grep -r "最佳实践" docs/
```

---

## 🎉 总结

- 📚 **9份详细文档** 覆盖所有需求
- 🎯 **按需选择** 不同深度的学习
- ⏱️ **灵活时间** 从5分钟到1天
- 🚀 **循序渐进** 从快速到深度
- 📖 **相互联系** 形成完整体系

**无论你的需求是什么，这里都有对应的文档！** 📖
