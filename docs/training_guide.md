# 工业级训练解决方案

## 🎯 核心特性

### 1. 多级检查点管理
- **最佳模型** (`best_model.pt`) - 验证损失最低的模型
- **最新模型** (`latest.pt`) - 支持断点续训
- **Epoch检查点** (`model_epoch_*.pt`) - 每轮自动保存
- **智能清理** - 自动保留最近N个检查点

### 2. 断点续训
```bash
# 从最新检查点恢复训练
python train_chinese.py --resume --epochs 5

# 从指定检查点继续
python train_chinese.py --checkpoint checkpoints/model_epoch_3.pt --epochs 5
```

### 3. 训练历史追踪
```bash
# 查看所有检查点
python train_manager.py list

# 查看训练历史曲线
python train_manager.py history

# 比较两个模型
python train_manager.py compare checkpoints/best_model.pt checkpoints/latest.pt
```

## 📋 使用示例

### 基础训练
```bash
# 使用示例数据训练3轮
python train_chinese.py --epochs 3 --batch-size 4

# 使用真实数据集
python train_chinese.py --data-file data/zh_sample.txt --epochs 5 --batch-size 8
```

### 生成的文件结构
```
checkpoints/
├── best_model.pt           # 🏆 最佳模型（验证损失最低）
├── latest.pt               # 📌 最新模型（用于断点续训）
├── model_epoch_1.pt        # 📦 第1轮检查点
├── model_epoch_2.pt        # 📦 第2轮检查点
├── model_epoch_3.pt        # 📦 第3轮检查点
├── model.pt                # 📄 向后兼容的主检查点
└── training_history.json   # 📈 训练历史数据
```

### 高级训练配置
```bash
# 保留最近5个epoch检查点
python train_chinese.py --epochs 10 --keep-last-n 5

# 不保存每个epoch（仅best和latest）
python train_chinese.py --epochs 5 --no-save-every-epoch

# 自定义学习率
python train_chinese.py --epochs 3 --learning-rate 5e-5
```

## 🔧 训练管理工具

### 查看检查点列表
```bash
python train_manager.py list
```
输出示例：
```
文件                           Epoch    验证损失      大小(MB)   修改时间
--------------------------------------------------------------------------------
🏆 best_model.pt              2        1.2345       536.23     2026-02-28 17:30:15
📌 latest.pt                  3        1.3456       536.23     2026-02-28 17:45:20
📦 model_epoch_1.pt           0        1.5678       536.23     2026-02-28 17:10:00
📦 model_epoch_2.pt           1        1.2345       536.23     2026-02-28 17:25:10
📦 model_epoch_3.pt           2        1.3456       536.23     2026-02-28 17:40:15
```

### 查看训练历史
```bash
python train_manager.py history
```
输出示例：
```
Epoch    训练损失         验证损失
--------------------------------------------------------------------------------
1        1.6789          1.5678
2        1.3456          1.2345
3        1.4567          1.3456

📊 统计:
  最佳Epoch: 2
  最佳验证损失: 1.2345
  总训练轮数: 3
```

### 清理旧检查点
```bash
# 保留最近3个
python train_manager.py clean --keep 3

# 保留最近5个
python train_manager.py clean --keep 5
```

### 比较模型
```bash
python train_manager.py compare checkpoints/best_model.pt checkpoints/model_epoch_3.pt
```

## 🚀 部署使用

### 使用最佳模型
```bash
# 启动服务（自动使用best_model.pt）
LLM_CHECKPOINT=checkpoints/best_model.pt make serve

# 或者直接运行
LLM_CHECKPOINT=checkpoints/best_model.pt python serve.py
```

### 测试不同检查点
```bash
# 测试epoch 2的模型
LLM_CHECKPOINT=checkpoints/model_epoch_2.pt python serve.py

# 测试最新模型
LLM_CHECKPOINT=checkpoints/latest.pt python serve.py
```

## 📊 训练日志格式

每轮训练输出：
```
======================================================================
📚 Epoch 1/3
======================================================================
训练中: 100%|██████████| 1000/1000 [05:23<00:00, 3.09it/s, loss=1.234, avg_loss=1.456]

🔍 验证中...
验证: 100%|██████████| 250/250 [00:45<00:00, 5.51it/s]

──────────────────────────────────────────────────────────────────────
📊 Epoch 1/3 结果:
──────────────────────────────────────────────────────────────────────
  ⏱️  用时: 368.5s (6.1min)
  📉 训练损失: 1.4567
  📊 验证损失: 1.2345
  ✨ 新的最佳损失! (提升: 0.3210)
  💾 Epoch检查点: model_epoch_1.pt
  🏆 最佳模型: best_model.pt
  📌 最新模型: latest.pt
──────────────────────────────────────────────────────────────────────
```

## 🎓 最佳实践

### 1. 长时间训练
```bash
# 使用nohup后台训练
nohup python train_chinese.py --epochs 10 --batch-size 4 > training.log 2>&1 &

# 查看训练进度
tail -f training.log
```

### 2. 意外中断后恢复
```bash
# 自动从latest.pt恢复
python train_chinese.py --resume --epochs 5
```

### 3. 定期清理检查点
```bash
# 每次训练后清理，保留最近3个
python train_manager.py clean --keep 3
```

### 4. 性能监控
```bash
# 训练时监控GPU
watch -n 1 nvidia-smi

# 查看训练历史趋势
python train_manager.py history
```

## 🔥 快速开始

```bash
# 1. 下载数据
python download_chinese_data.py --type sample

# 2. 开始训练（3轮，保留最近3个检查点）
python train_chinese.py \
  --data-file data/zh_sample.txt \
  --epochs 3 \
  --batch-size 4 \
  --keep-last-n 3

# 3. 查看结果
python train_manager.py list
python train_manager.py history

# 4. 使用最佳模型
LLM_CHECKPOINT=checkpoints/best_model.pt make serve
```

## 📈 训练历史JSON格式

`checkpoints/training_history.json`:
```json
{
  "train_loss": [1.6789, 1.3456, 1.4567],
  "val_loss": [1.5678, 1.2345, 1.3456],
  "epochs": [1, 2, 3]
}
```

## 🛠️ 故障排查

### 问题1：GPU内存不足
```bash
# 减少batch size
python train_chinese.py --batch-size 2

# 检查GPU状态
nvidia-smi
```

### 问题2：检查点损坏
```bash
# 使用前一个epoch的检查点
python train_chinese.py --checkpoint checkpoints/model_epoch_2.pt --epochs 5
```

### 问题3：训练中断
```bash
# 始终可以从latest.pt恢复
python train_chinese.py --resume --epochs 5
```
