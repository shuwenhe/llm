# 🎓 训练和检查点速查卡

## 🚀 最快开始（复制粘贴）

```bash
# 快速验证
python train_cli.py --preset quick

# 标准训练
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 一键启动脚本
bash quick_start.sh standard data/zh_sample.txt
```

## 📊 5个核心命令

| 功能 | 命令 | 说明 |
|------|------|------|
| 📋 查看预设 | `python train_cli.py --list-presets` | 列出4个预设配置 |
| 🧪 干运行 | `python train_cli.py --preset quick --dry-run` | 验证命令不执行 |
| 🎓 开始训练 | `python train_cli.py --preset standard` | 执行标准训练 |
| 📈 查看结果 | `python train_manager.py history` | 训练历史和损失曲线 |
| 🔄 恢复训练 | `python train_cli.py --resume --epochs 5` | 从latest.pt恢复 |

## 🏆 4个训练预设

| 预设 | 命令 | 时间 | 用途 |
|------|------|------|------|
| QUICK | `--preset quick` | 1分钟 | 验证管道 |
| STANDARD | `--preset standard` | 30分钟 | 日常使用 |
| EXTENDED | `--preset extended` | 2小时 | 深度训练 |
| PRECISION | `--preset precision` | 5小时 | 生产部署 |

## 📁 三级检查点速查

| 文件 | 用途 | 何时更新 | 何时使用 |
|------|------|---------|--------|
| `model_epoch_*.pt` | 轮次历史 | 每轮保存 | 对比轮次质量 |
| `best_model.pt` | 最优模型 | 损失改进时 | 生产部署 |
| `latest.pt` | 最新模型 | 每轮保存 | 恢复训练 |

## 🎯 检查点管理命令

```bash
# 列出所有检查点
python train_manager.py list

# 查看训练历史
python train_manager.py history

# 对比两个模型
python train_manager.py compare model1.pt model2.pt

# 清理旧检查点
python train_manager.py clean --keep 3
```

## 🔍 实时监控检查点

### 方式1: watch命令（最简单）
```bash
watch -n 1 'ls -lh checkpoints/*.pt | tail -5'
```

### 方式2: 持续监控
```bash
while true; do
  clear
  echo "📊 检查点:" 
  ls -lh checkpoints/*.pt
  sleep 2
done
```

### 方式3: 监控脚本
```bash
bash monitor_training.sh
```

## 🚀 使用检查点

### 生产部署
```bash
LLM_CHECKPOINT=checkpoints/best_model.pt make serve
```

### 继续训练（快速）
```bash
python train_cli.py --resume --epochs 5
```

### 从特定轮继续
```bash
python train_chinese.py --checkpoint checkpoints/model_epoch_2.pt --epochs 3
```

### 在最优基础上微调
```bash
python train_cli.py --preset extended \
  --checkpoint checkpoints/best_model.pt \
  --learning-rate 5e-5 \
  --epochs 10
```

## 📚 文档导航

| 文档 | 内容 | 何时看 |
|------|------|--------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | 快速指南 | 第1次使用 |
| [TRAINING_README.md](TRAINING_README.md) | 完整功能说明 | 了解全貌 |
| [docs/checkpoint_system.md](docs/checkpoint_system.md) | 检查点详解 | 理解原理 |
| [docs/training_visualization.md](docs/training_visualization.md) | 监控指南 | 实时监控 |
| [docs/commands_reference.md](docs/commands_reference.md) | 命令参考 | 查找命令 |

## 💡 常见场景

### 场景1: 我想快速验证
```bash
python train_cli.py --preset quick
# ✅ 1分钟完成，生成完整的3级检查点
```

### 场景2: 标准训练
```bash
bash quick_start.sh standard data/zh_sample.txt
# ✅ 30分钟，完整UI，最后显示所有检查点
```

### 场景3: 训练中断了，要恢复
```bash
python train_cli.py --resume --epochs 10
# ✅ 从latest.pt恢复，包含优化器状态
```

### 场景4: 用最好的模型部署
```bash
LLM_CHECKPOINT=checkpoints/best_model.pt make serve
# ✅ 自动选择最优模型启动服务
```

### 场景5: 对比不同轮的模型
```bash
python train_manager.py compare \
  checkpoints/model_epoch_1.pt \
  checkpoints/best_model.pt
# ✅ 显示性能对比
```

## 🔧 参数速查

### 训练参数
```bash
python train_cli.py \
  --preset standard           # 使用预设
  --batch-size 8             # 改批次大小
  --epochs 5                 # 改轮数
  --learning-rate 5e-5       # 改学习率
  --data-file data/zh_wiki.txt # 数据文件
  --keep-last-n 5            # 保留5个epoch
```

### 特殊选项
```bash
--dry-run                    # 只显示命令不执行
--resume                     # 从latest恢复
--no-save-every-epoch        # 不保存epoch检查点
--no-log                     # 不记录日志
```

## 📊 性能指标（15.6GB VRAM）

| 预设 | 批次 | 轮数 | 时间 | 显存占用 |
|------|------|------|------|---------|
| QUICK | 2 | 1 | 1分钟 | 3GB |
| STANDARD | 4 | 3 | 30分钟 | 5GB |
| EXTENDED | 8 | 10 | 2小时 | 8GB |
| PRECISION | 16 | 20 | 5小时 | 12GB |

## ✅ 检查清单

### 训练前
- [ ] `python train_cli.py --list-presets` 查看预设
- [ ] `python train_cli.py --preset quick --dry-run` 验证命令
- [ ] 确保有足够的磁盘空间（~3GB）
- [ ] 检查GPU显存足够

### 训练中
- [ ] 用 `watch` 命令监控检查点生成
- [ ] 或在另一个终端看日志 `tail -f logs/*.log`
- [ ] 如需中断，Ctrl+C 即可（latest.pt保存了完整状态）

### 训练后
- [ ] `python train_manager.py list` 查看所有检查点
- [ ] `python train_manager.py history` 查看训练曲线
- [ ] 确认 `best_model.pt` 是最优模型
- [ ] 备份重要的模型 `cp best_model.pt backup/`

### 部署时
- [ ] 使用 `best_model.pt` 而不是 `latest.pt`
- [ ] 命令: `LLM_CHECKPOINT=checkpoints/best_model.pt make serve`
- [ ] 测试模型推理效果

## 🚨 故障排查

### 看不到检查点保存信息
```bash
# 查看日志而不是终端输出
python train_cli.py --preset quick 2>&1 | tee train.log
tail -f train.log | grep -E "(保存|检查点)"
```

### 显存不足
```bash
# 使用quick预设或降低batch_size
python train_cli.py --preset quick
python train_cli.py --batch-size 1 --epochs 1
```

### 检查点文件很大
```bash
# 不保存epoch检查点，只保留best和latest
python train_cli.py --preset standard --no-save-every-epoch
```

### 磁盘空间满
```bash
# 清理旧检查点
python train_manager.py clean --keep 2
```

## 🎉 预期结果

训练3轮后：

```
checkpoints/
├── model_epoch_1.pt (487MB)
├── model_epoch_2.pt (487MB) ⭐ 最优
├── model_epoch_3.pt (487MB)
├── best_model.pt (487MB) 🏆 指向epoch2
├── latest.pt (487MB) 📌 指向epoch3
├── model.pt (487MB) 📍 主检查点
└── training_history.json

$ python train_manager.py history
📊 最佳验证损失: 3.8012 (Epoch 2)

$ LLM_CHECKPOINT=checkpoints/best_model.pt make serve
✓ 使用最优模型启动服务
```

## 🔗 快速链接

- 📖 [完整指南](openai_training_guide.md)
- 📖 [命令参考](docs/commands_reference.md)
- 📖 [检查点系统](docs/checkpoint_system.md)
- 📖 [监控可视化](docs/training_visualization.md)
- 📖 [更新详情](CHECKPOINT_UPDATE.md)

---

**记住**: 
- 🏆 生产部署用 `best_model.pt`
- 📌 恢复训练用 `latest.pt`
- 📦 历史保存在 `model_epoch_*.pt`
- 📈 完整数据在 `training_history.json`

**任何问题**: 查看对应的文档即可找到答案！
