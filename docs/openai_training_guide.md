# OpenAI风格的工业级训练命令

## 🎯 概述

这是一套符合OpenAI标准的工业级训练系统，支持：
- ✅ 预设配置（快速、标准、高精度）
- ✅ 配置文件管理
- ✅ 完整日志记录
- ✅ 干运行模式
- ✅ 断点续训
- ✅ 命令行灵活性

## 🚀 快速开始

### 1. 使用预设训练

```bash
# 快速验证（1轮，验证管道）
python train_cli.py --preset quick

# 标准训练（3轮，默认配置）
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 长期训练（10轮）
python train_cli.py --preset extended --data-file data/zh_sample.txt

# 高精度训练（20轮，低学习率）
python train_cli.py --preset precision --data-file data/zh_sample.txt
```

### 2. 列出可用预设

```bash
python train_cli.py --list-presets
```

输出：
```
📋 可用的训练预设:
================================================================================

QUICK
  批次大小: 2
  训练轮数: 1
  学习率: 0.0001

STANDARD
  批次大小: 4
  训练轮数: 3
  学习率: 0.0001

EXTENDED
  批次大小: 8
  训练轮数: 10
  学习率: 5e-05

PRECISION
  批次大小: 16
  训练轮数: 20
  学习率: 1e-05
```

## 📋 配置管理

### 从配置文件加载

```bash
# 使用预设配置文件
python train_cli.py --config config/training_standard.json

# 使用自定义配置
python train_cli.py --config my_config.json --data-file data/zh_wiki.txt
```

### 保存当前配置

```bash
# 保存预设配置
python train_cli.py --preset precision --save-config my_training.json

# 查看保存的配置
cat my_training.json
```

### 配置文件格式

```json
{
  "batch_size": 4,
  "epochs": 3,
  "learning_rate": 1e-4,
  "save_every_epoch": true,
  "keep_last_n": 3
}
```

## ⚙️ 自定义训练

### 覆盖预设参数

```bash
# 从standard预设开始，但改为10轮
python train_cli.py --preset standard --epochs 10

# 组合多个参数
python train_cli.py --preset standard \
  --batch-size 8 \
  --epochs 5 \
  --learning-rate 5e-5 \
  --keep-last-n 5
```

### 完整参数列表

```bash
python train_cli.py \
  --batch-size 4          # 批次大小
  --epochs 3              # 训练轮数
  --learning-rate 1e-4    # 学习率
  --data-file data.txt    # 数据文件
  --keep-last-n 3         # 保留检查点数
  --no-save-every-epoch   # 不保存epoch检查点
  --resume                # 从latest.pt恢复
  --dry-run               # 打印但不执行
  --no-log                # 不记录日志
```

## 🧪 干运行模式

在实际执行前测试命令：

```bash
# 检查快速训练的命令
python train_cli.py --preset quick --dry-run

# 检查自定义参数的命令
python train_cli.py --batch-size 16 --epochs 20 --dry-run
```

输出：
```
✓ 使用预设: quick

================================================================================
🎓 训练配置
================================================================================
  batch_size: 2
  epochs: 1
  learning_rate: 0.0001
  save_every_epoch: True
  keep_last_n: 1
================================================================================

📝 执行命令: ./venv/bin/python train_chinese.py --batch-size 2 --epochs 1 ...
✓ 干运行模式 (不执行)
```

## 📊 日志记录

训练自动生成日志：

```
logs/
├── training_20260228_173000.log    # 训练日志
└── config_20260228_173000.json     # 训练配置
```

查看日志：
```bash
# 查看最新日志
tail -f logs/training_*.log

# 统计训练结果
grep "最佳损失" logs/training_*.log
```

## 🔄 断点续训

从中断处恢复：

```bash
# 继续之前的训练
python train_cli.py --preset standard --resume

# 继续并扩展轮数
python train_cli.py --preset extended --resume --data-file data/zh_sample.txt
```

## 🎓 高级用法

### 比较不同配置

```bash
# 快速vs标准
python train_cli.py --preset quick --dry-run
python train_cli.py --preset standard --dry-run

# 不同学习率
python train_cli.py --learning-rate 1e-4 --dry-run
python train_cli.py --learning-rate 5e-5 --dry-run
```

### 批量训练

```bash
#!/bin/bash
# 顺序运行多个训练

echo "快速验证..."
python train_cli.py --preset quick

echo "标准训练..."
python train_cli.py --preset standard --data-file data/zh_sample.txt

echo "高精度训练..."
python train_cli.py --preset precision --data-file data/zh_sample.txt
```

### 调度训练

```bash
# 在后台运行
nohup python train_cli.py --preset precision --data-file data/zh_wiki.txt > training.log 2>&1 &

# 定时训练（每晚11点）
0 23 * * * cd /home/shuwen/llm && python train_cli.py --preset extended --data-file data/zh_wiki.txt
```

## 📈 预设详解

### QUICK（快速验证）
- 用途：验证管道、快速测试
- 批次：2（节省内存）
- 轮数：1（快速完成）
- 学习率：1e-4（标准）
- 时间：~1分钟

### STANDARD（标准训练）
- 用途：日常训练、模型微调
- 批次：4（平衡显存和质量）
- 轮数：3（基本收敛）
- 学习率：1e-4（标准）
- 时间：~30分钟

### EXTENDED（长期训练）
- 用途：深度微调、大数据集
- 批次：8（更大批次）
- 轮数：10（充分训练）
- 学习率：5e-5（降低学习率）
- 时间：~2小时

### PRECISION（高精度训练）
- 用途：生产部署、最优模型
- 批次：16（充分利用GPU）
- 轮数：20（充分收敛）
- 学习率：1e-5（微调）
- 时间：~5小时

## 💡 最佳实践

### 1. 快速验证
```bash
python train_cli.py --preset quick
```

### 2. 验证成功后标准训练
```bash
python train_cli.py --preset standard --data-file data/zh_sample.txt
```

### 3. 查看结果
```bash
python train_manager.py list
python train_manager.py history
```

### 4. 如需改进，高精度训练
```bash
python train_cli.py --preset precision --resume --data-file data/zh_wiki.txt
```

### 5. 部署最佳模型
```bash
LLM_CHECKPOINT=checkpoints/best_model.pt make serve
```

## 🔧 故障排查

### 问题：GPU内存不足
```bash
# 使用quick预设
python train_cli.py --preset quick

# 或降低batch_size
python train_cli.py --batch-size 2 --epochs 1
```

### 问题：训练中断
```bash
# 查看日志
tail -f logs/training_*.log

# 恢复训练
python train_cli.py --preset standard --resume
```

### 问题：想调试命令
```bash
# 使用干运行模式
python train_cli.py --preset precision --dry-run

# 查看完整命令后修改
```

## 📚 与Makefile集成

在Makefile中添加：

```makefile
train-openai:
	@python train_cli.py $(ARGS)

train-openai-quick:
	@python train_cli.py --preset quick

train-openai-standard:
	@python train_cli.py --preset standard --data-file $(DATA_FILE)

train-openai-precision:
	@python train_cli.py --preset precision --data-file $(DATA_FILE)
```

使用：
```bash
make train-openai --preset quick
make train-openai-standard DATA_FILE=data/zh_sample.txt
```
