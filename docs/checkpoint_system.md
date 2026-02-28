# 📁 检查点系统详解

## 🎯 三级检查点系统架构

你的训练系统现在自动保存**三级检查点**，提供完整的模型管理：

```
每轮训练自动保存:

checkpoints/
├── model_epoch_1.pt      ← Epoch 1 (第1轮完整检查点)
├── model_epoch_2.pt      ← Epoch 2 (第2轮完整检查点)
├── model_epoch_3.pt      ← Epoch 3 (第3轮完整检查点)
│   [保持最近N个，旧的自动删除]
│
├── best_model.pt         ← 🏆 最优模型 (最低验证损失时自动更新)
│   [只有验证损失改进时才保存]
│
├── latest.pt             ← 📌 最新模型 (每轮自动保存，用于恢复)
│   [每轮都更新，占用空间小]
│
├── model.pt              ← 📍 主检查点 (向后兼容)
│   [每轮都更新]
│
└── training_history.json ← 📈 完整训练历史
    [记录所有轮次的损失值]
```

## 📊 各个检查点的用途

### 1️⃣ `model_epoch_*.pt` - Epoch检查点（轮次历史）

**用途**: 保存每一轮的完整模型状态

```python
# 自动生成：
model_epoch_1.pt  # 第1轮完成后
model_epoch_2.pt  # 第2轮完成后
model_epoch_3.pt  # 第3轮完成后
```

**特点**:
- ✅ 完整保存（包含优化器状态、配置等）
- ✅ 按轮次编号，易于追溯
- ✅ 自动清理（`--keep-last-n` 参数控制保留数量）
- ✅ 便于对比不同轮次的模型质量

**使用场景**:
```bash
# 查看所有epoch检查点
python train_manager.py list

# 对比第1轮和第3轮的模型
python train_manager.py compare checkpoints/model_epoch_1.pt checkpoints/model_epoch_3.pt

# 使用第2轮的模型继续训练
python train_chinese.py --checkpoint checkpoints/model_epoch_2.pt --epochs 5
```

### 2️⃣ `best_model.pt` - 最佳模型（最优结果）

**用途**: 保存整个训练过程中**验证损失最低**的模型

```python
# 自动管理：
# Epoch 1: val_loss=4.50 → 保存为best_model.pt ✓
# Epoch 2: val_loss=3.80 → 更新best_model.pt ✓
# Epoch 3: val_loss=4.20 → 保持不变 (未改进)
```

**特点**:
- 🏆 只保存最优结果
- 📉 验证损失最低的模型
- 🎯 最适合部署生产
- 📦 占用空间（只有1个文件）

**使用场景**:
```bash
# 生产部署使用最优模型
LLM_CHECKPOINT=checkpoints/best_model.pt make serve

# 查看最优模型的详细信息
python train_manager.py list | grep best

# 在最优模型基础上继续微调
python train_chinese.py --checkpoint checkpoints/best_model.pt --epochs 3 --learning-rate 5e-5
```

### 3️⃣ `latest.pt` - 最新模型（断点续训）

**用途**: 保存**最新一轮**的模型，用于**断点续训**和**快速恢复**

```python
# 自动更新：
# Epoch 1 完成 → latest.pt = epoch 1的完整状态
# Epoch 2 完成 → latest.pt = epoch 2的完整状态
# Epoch 3 完成 → latest.pt = epoch 3的完整状态
```

**特点**:
- 📌 每轮自动更新
- 🔄 包含优化器状态（可直接继续训练）
- 💾 完整保存（大小约500MB）
- ⚡ 用于快速恢复

**使用场景**:
```bash
# 查看最新模型的修改时间
ls -lh checkpoints/latest.pt

# 恢复中断的训练
python train_cli.py --resume

# 或在命令行指定
python train_chinese.py --checkpoint checkpoints/latest.pt --epochs 5
```

### 4️⃣ `model.pt` - 主检查点（向后兼容）

**用途**: 保持与旧版本的兼容性

```python
# 每轮都更新，始终是最新的完整模型
# 等同于 latest.pt 的内容
```

### 5️⃣ `training_history.json` - 训练历史

**用途**: 记录所有轮次的**训练损失**和**验证损失**

```json
{
  "train_loss": [5.2341, 3.9843, 3.4521],
  "val_loss": [4.8934, 3.8012, 3.6721],
  "epochs": [1, 2, 3]
}
```

**使用场景**:
```bash
# 查看损失曲线
python train_manager.py history

# 导入到Excel/Python绘图
import json
with open('checkpoints/training_history.json') as f:
    history = json.load(f)
    print(f"最佳验证损失: {min(history['val_loss'])}")
```

## 🎮 实战例子

### 例子1: 标准三轮训练

```bash
python train_cli.py --preset standard --data-file data/zh_sample.txt
```

输出：
```
训练中: 100%|████████████████| 6252/6252 [04:23<00:00, 23.68it/s, loss=3.4521, avg_loss=3.4521]

================================================
📊 Epoch 1/3 结果
================================================
  ⏱️  用时: 264.1s (4.4min)
  📉 训练损失: 5.2341
  📊 验证损失: 4.8934

💾 保存检查点...
  ✓ Epoch检查点: model_epoch_1.pt (487.3MB)
  🏆 最佳模型: best_model.pt (487.3MB) [改进: inf]  ← 第1轮自动成为最佳
  📌 最新模型: latest.pt (487.3MB) [用于恢复训练]
  📍 主检查点: model.pt (487.3MB)
================================================
```

完成后检查点目录：
```bash
$ ls -lh checkpoints/
-rw-r--r-- model_epoch_1.pt     487.3MB
-rw-r--r-- model_epoch_2.pt     487.3MB
-rw-r--r-- model_epoch_3.pt     487.3MB
-rw-r--r-- best_model.pt        487.3MB  ← 最优模型（假设epoch 2最优）
-rw-r--r-- latest.pt            487.3MB  ← 最新的epoch 3
-rw-r--r-- model.pt             487.3MB  ← 主检查点
-rw-r--r-- training_history.json 248B    ← 训练历史
```

### 例子2: 查看和对比检查点

```bash
# 列出所有检查点
$ python train_manager.py list

📋 训练检查点
================================================================================
文件名                   Epoch  验证损失   大小     修改时间
================================================================================
model_epoch_1.pt           1    4.8934    487.3MB  2026-02-28 17:45:23
model_epoch_2.pt           2    3.8012    487.3MB  2026-02-28 17:50:15
model_epoch_3.pt           3    3.6721    487.3MB  2026-02-28 17:55:07
best_model.pt              2    3.8012    487.3MB  2026-02-28 17:50:15  ⭐
latest.pt                  3    3.6721    487.3MB  2026-02-28 17:55:07  📌
================================================================================
```

对比epoch 2（最优）和epoch 3：
```bash
$ python train_manager.py compare checkpoints/model_epoch_2.pt checkpoints/model_epoch_3.pt

📊 模型对比
================================================================================
指标              model_epoch_2.pt      model_epoch_3.pt
================================================================================
验证损失          3.8012                3.6721            (-0.1291)
训练损失          3.9843                3.4521            (-0.5322)
总参数数          124,046,592           124,046,592
检查点大小        487.3MB               487.3MB
================================================================================
```

### 例子3: 断点续训

**场景**: 训练中断了，需要恢复

```bash
# 查看最后的检查点
$ ls -lh checkpoints/latest.pt
-rw-r--r-- latest.pt  487.3MB  2026-02-28 17:50:15

# 从中断处恢复并继续
$ python train_cli.py --resume --epochs 5

# 或指定具体的检查点
$ python train_chinese.py --checkpoint checkpoints/latest.pt --epochs 5
```

**说明**:
- `latest.pt` 包含了**完整的优化器状态**
- 可以直接从该轮次继续，**不需要重新开始**
- 学习率和其他参数可以调整

### 例子4: 在最优模型基础上微调

**场景**: 在最好的模型基础上做进一步优化

```bash
# 方案1: 使用较低的学习率继续微调
python train_cli.py --preset extended \
  --checkpoint checkpoints/best_model.pt \
  --learning-rate 5e-5 \
  --epochs 10

# 方案2: 加载最优模型并切换到高精度训练
python train_chinese.py \
  --checkpoint checkpoints/best_model.pt \
  --batch-size 16 \
  --epochs 20 \
  --learning-rate 1e-5
```

## 🔧 检查点参数控制

### 保存间隔

```bash
# 保存每个epoch的检查点（默认）
python train_cli.py --preset standard

# 不保存epoch检查点（只保存best和latest）
python train_cli.py --preset standard --no-save-every-epoch
```

### 保留检查点数量

```bash
# 保留最近5个epoch检查点
python train_cli.py --preset standard --keep-last-n 5

# 保留最近10个epoch检查点
python train_cli.py --preset extended --keep-last-n 10

# 保留所有epoch检查点（不删除）
python train_cli.py --preset precision --keep-last-n 0
```

## 📈 管理工具命令

```bash
# 列出所有检查点及其元数据
python train_manager.py list

# 查看训练历史和损失曲线
python train_manager.py history

# 对比两个模型的性能
python train_manager.py compare model1.pt model2.pt

# 清理旧检查点（保留最近5个）
python train_manager.py clean --keep 5

# 删除特定检查点
rm checkpoints/model_epoch_1.pt
```

## 🎯 最佳实践

### 原则1: 始终使用 `latest.pt` 恢复训练

```bash
# ✅ 推荐
python train_cli.py --resume

# ✅ 也可以
python train_chinese.py --checkpoint checkpoints/latest.pt
```

### 原则2: 用 `best_model.pt` 进行生产部署

```bash
# ✅ 生产环境
LLM_CHECKPOINT=checkpoints/best_model.pt make serve

# ❌ 不推荐用latest（可能未收敛）
# LLM_CHECKPOINT=checkpoints/latest.pt make serve
```

### 原则3: 定期清理旧检查点节省空间

```bash
# 定期清理（保留最近5个）
python train_manager.py clean --keep 5

# 或手动删除
rm checkpoints/model_epoch_1.pt
```

### 原则4: 备份最优模型

```bash
# 备份最好的模型
cp checkpoints/best_model.pt backups/best_model_v1.pt

# 备份训练历史
cp checkpoints/training_history.json backups/history_v1.json
```

## 🔍 检查点内部结构

每个检查点都包含完整的训练状态：

```python
{
  'model': {...},                    # 模型参数
  'model_config': {...},             # 模型配置
  'optimizer': {...},                # 优化器状态（用于断点续训）
  'epoch': 1,                        # 当前轮数
  'best_loss': 3.8012,               # 最佳验证损失
  'train_loss': 5.2341,              # 当前轮训练损失
  'val_loss': 4.8934,                # 当前轮验证损失
  'history': {                       # 训练历史
    'train_loss': [...],
    'val_loss': [...],
    'epochs': [...]
  }
}
```

## 📊 存储空间指南

每个检查点大小约**487MB**（取决于模型大小）：

```
10轮训练（保留最近3个epoch）:
├── 3个epoch检查点: 487MB × 3 = 1.5GB
├── 1个best_model: 487MB
├── 1个latest: 487MB
└── 日志文件: ~100MB
───────────────────
总计: ~2.8GB
```

**节省空间方案**:
```bash
# 方案1: 不保存epoch检查点
python train_cli.py --preset standard --no-save-every-epoch

# 方案2: 只保留最近2个
python train_cli.py --preset standard --keep-last-n 2

# 方案3: 定期清理
python train_manager.py clean --keep 2
```

---

**总结**: 你的系统现在有完整的检查点管理，每次训练都自动保存多个备份，确保不会丢失任何有价值的模型版本！ 🎉
