# 🎬 训练过程可视化指南

## 📺 实时监控检查点生成

### 分屏监控方案（推荐）

**终端1**: 执行训练
```bash
cd /home/shuwen/llm
python train_cli.py --preset standard --data-file data/zh_sample.txt
```

**终端2**: 实时监控检查点生成
```bash
# 方案A: 每秒监控文件变化
watch -n 1 'ls -lhtr checkpoints/ | tail -10'

# 方案B: 持续监控（更详细）
while true; do
  echo "=== 检查点状态 ===" 
  ls -lh checkpoints/*.pt 2>/dev/null | awk '{print $9, "-", $5}'
  echo "=== 修改时间 ==="
  ls -lt checkpoints/*.pt 2>/dev/null | head -5 | awk '{print $9, "-", $6, $7, $8}'
  sleep 1
done

# 方案C: 针对性监控（推荐）
while true; do
  clear
  echo "📊 检查点监控 (更新时间: $(date '+%H:%M:%S'))"
  echo "=================================================="
  
  echo -e "\n🏆 最佳模型:"
  ls -lh checkpoints/best_model.pt 2>/dev/null | awk '{printf "  %s: %s\n", $9, $5}'
  
  echo -e "\n📌 最新模型:"
  ls -lh checkpoints/latest.pt 2>/dev/null | awk '{printf "  %s: %s\n", $9, $5}'
  
  echo -e "\n📦 Epoch检查点:"
  ls -lh checkpoints/model_epoch_*.pt 2>/dev/null | awk '{printf "  %s: %s\n", $9, $5}'
  
  echo -e "\n📈 训练历史:"
  if [ -f checkpoints/training_history.json ]; then
    python3 -c "
import json
try:
  with open('checkpoints/training_history.json') as f:
    h = json.load(f)
    if h['epochs']:
      latest_ep = h['epochs'][-1]
      latest_val = h['val_loss'][-1]
      print(f'  Epoch {latest_ep}: val_loss={latest_val:.4f}')
except: pass
    "
  fi
  
  sleep 2
done
```

### 日志监控

**终端3**: 查看训练日志
```bash
# 实时查看最新的训练日志
tail -f logs/training_*.log | grep -E "(Epoch|损失|检查点|保存)"

# 或查看特定日志
tail -f /home/shuwen/llm/logs/training_$(date +%Y%m%d)*.log
```

## 📊 完整训练流程演示

### 第1轮 (Epoch 1) - 建立基线

```
训练中:  33%|██████████            | 2084/6252 [01:29<03:05, 22.33it/s]
        loss=5.1234, avg_loss=5.1340

[1分钟后...]

✅ Epoch 1/3 完成

📊 Epoch 1/3 结果
==================================================
  ⏱️  用时: 264.1s (4.4min)
  📉 训练损失: 5.2341
  📊 验证损失: 4.8934

💾 保存检查点...
  ✓ Epoch检查点: model_epoch_1.pt (487.3MB)
  🏆 最佳模型: best_model.pt (487.3MB) [改进: inf] ← 第一次必然是最佳
  📌 最新模型: latest.pt (487.3MB) [用于恢复训练]
  📍 主检查点: model.pt (487.3MB)
==================================================

🎯 此时 checkpoints/ 目录:
  ✓ model_epoch_1.pt (487.3MB) - Epoch 1完整
  ✓ best_model.pt (487.3MB) - 当前最优
  ✓ latest.pt (487.3MB) - 最新保存
  ✓ model.pt (487.3MB) - 主检查点
```

### 第2轮 (Epoch 2) - 模型改进

```
[4分钟后...]

✅ Epoch 2/3 完成

📊 Epoch 2/3 结果
==================================================
  ⏱️  用时: 268.3s (4.5min)
  📉 训练损失: 3.9843
  📊 验证损失: 3.8012

💾 保存检查点...
  ✓ Epoch检查点: model_epoch_2.pt (487.3MB)
  🏆 最佳模型: best_model.pt (487.3MB) [改进: 1.0922] ← 损失从4.89降到3.80
  📌 最新模型: latest.pt (487.3MB) [用于恢复训练]
  📍 主检查点: model.pt (487.3MB)
==================================================

🎯 此时 checkpoints/ 目录:
  ✓ model_epoch_1.pt (487.3MB) - Epoch 1
  ✓ model_epoch_2.pt (487.3MB) - Epoch 2
  ✓ best_model.pt (487.3MB) - ⭐ 更新为Epoch 2
  ✓ latest.pt (487.3MB) - 更新为Epoch 2
  ✓ model.pt (487.3MB) - 更新为Epoch 2
```

### 第3轮 (Epoch 3) - 最后冲刺

```
[4分钟后...]

✅ Epoch 3/3 完成

📊 Epoch 3/3 结果
==================================================
  ⏱️  用时: 265.8s (4.4min)
  📉 训练损失: 3.4521
  📊 验证损失: 3.6721

💾 保存检查点...
  ✓ Epoch检查点: model_epoch_3.pt (487.3MB)
  ℹ️  验证损失未改进 (最佳: 3.8012) ← 这次没有超过Epoch 2
  📌 最新模型: latest.pt (487.3MB) [用于恢复训练]
  📍 主检查点: model.pt (487.3MB)
==================================================

🎯 最终 checkpoints/ 目录:
  ✓ model_epoch_1.pt (487.3MB) - Epoch 1
  ✓ model_epoch_2.pt (487.3MB) - Epoch 2 ⭐ 最优
  ✓ model_epoch_3.pt (487.3MB) - Epoch 3
  ✓ best_model.pt (487.3MB) - 🏆 最优模型 = Epoch 2
  ✓ latest.pt (487.3MB) - 📌 最新模型 = Epoch 3
  ✓ model.pt (487.3MB) - 主检查点 = Epoch 3
  ✓ training_history.json (248B) - 完整历史
```

### 训练完成总结

```
================================================================================
✅ 训练完成!
================================================================================
  ⏱️  总用时: 797.2s (13.3min)
  📊 最佳验证损失: 3.8012 (Epoch 2)

📁 保存的模型:
  🏆 最佳模型: checkpoints/best_model.pt
  📌 最新模型: checkpoints/latest.pt
  📦 Epoch检查点: checkpoints/model_epoch_*.pt (最近3个)

🚀 使用训练后的模型:
  # 使用最佳模型
  LLM_CHECKPOINT=checkpoints/best_model.pt make serve
  # 或继续训练
  python train_chinese.py --checkpoint checkpoints/latest.pt --epochs 3
================================================================================

📈 训练历史已保存: checkpoints/training_history.json
```

## 🔍 查看历史的3种方式

### 方式1: 命令行查看

```bash
python train_manager.py history
```

输出：
```
📊 训练历史分析
================================================================================
总轮数: 3
最佳轮次: 2
最佳验证损失: 3.8012

损失曲线:
  Epoch 1: 📈 val_loss=4.8934
  Epoch 2: 📉 val_loss=3.8012  ⭐ 最佳
  Epoch 3: 📈 val_loss=3.6721
================================================================================
```

### 方式2: 检查列表

```bash
python train_manager.py list
```

输出：
```
📋 训练检查点
================================================================================
文件名              Epoch  验证损失   大小     修改时间
================================================================================
model_epoch_1.pt      1    4.8934    487.3MB  2026-02-28 17:45:23
model_epoch_2.pt      2    3.8012    487.3MB  2026-02-28 17:50:15 ⭐
model_epoch_3.pt      3    3.6721    487.3MB  2026-02-28 17:55:07
best_model.pt         2    3.8012    487.3MB  2026-02-28 17:50:15
latest.pt             3    3.6721    487.3MB  2026-02-28 17:55:07
================================================================================
```

### 方式3: JSON查看（用于分析）

```bash
cat checkpoints/training_history.json | python -m json.tool
```

输出：
```json
{
  "train_loss": [5.2341, 3.9843, 3.4521],
  "val_loss": [4.8934, 3.8012, 3.6721],
  "epochs": [1, 2, 3]
}
```

## 🎯 实时对比示例

### 当前训练进行中

```bash
# 终端1: 执行训练
python train_cli.py --preset extended --data-file data/zh_sample.txt

# 终端2: 监控进度
while true; do
  clear
  echo "📊 训练进度监控 (更新: $(date))"
  
  # 显示当前在运行的进程
  ps aux | grep train_cli | grep -v grep && echo "✓ 训练进行中"
  
  # 显示最新的日志行
  echo -e "\n📝 最新日志:"
  tail -3 logs/training_*.log 2>/dev/null
  
  # 显示当前的检查点
  echo -e "\n📁 检查点状态:"
  ls -lt checkpoints/*.pt 2>/dev/null | head -3 | awk '{print $9, "("$5")", "- 修改于", $6, $7, $8}'
  
  sleep 2
done
```

## 💡 关键时刻截图

### ✅ 第一个检查点出现（Epoch 1完成）

```
时间戳: 17:45:23
新文件出现:
  ✓ checkpoints/model_epoch_1.pt (487.3MB)
  ✓ checkpoints/best_model.pt (487.3MB)
  ✓ checkpoints/latest.pt (487.3MB)
  ✓ checkpoints/model.pt (487.3MB)
```

### 🏆 最佳模型更新（当验证损失改进）

```
时间戳: 17:50:15
最佳模型更新:
  best_model.pt 文件修改时间改变
  从 model_epoch_1 → model_epoch_2
  改进: 4.8934 → 3.8012 (提升 1.0922)
```

### 📌 最新模型更新（每个Epoch）

```
时间戳: 17:55:07
最新模型更新:
  latest.pt 文件修改时间改变
  从 epoch 2 → epoch 3
  这个文件包含完整的优化器状态
```

## 🚨 检查点未生成的排查

如果你看不到检查点生成，可能的原因和解决方案：

### ❌ 问题1: 输出被进度条覆盖

**症状**: 看不到"保存检查点"的信息

**解决**:
```bash
# 1. 重定向到文件查看
python train_cli.py --preset standard 2>&1 | tee training.log

# 2. 在另一个终端查看日志
tail -f training.log | grep -E "(保存|检查点|Epoch)"
```

### ❌ 问题2: 检查点目录不存在

**症状**: 报错 "checkpoints 目录不存在"

**解决**:
```bash
# 手动创建目录
mkdir -p /home/shuwen/llm/checkpoints

# 检查权限
chmod 755 /home/shuwen/llm/checkpoints
```

### ❌ 问题3: 磁盘空间不足

**症状**: 训练开始但未保存检查点

**解决**:
```bash
# 检查磁盘空间
df -h

# 清理旧检查点
python train_manager.py clean --keep 2
```

### ❌ 问题4: GPU显存溢出导致训练中断

**症状**: 训练中断，未完成Epoch 1

**解决**:
```bash
# 使用更小的批次
python train_cli.py --preset quick  # batch_size=2

# 或手动设置
python train_cli.py --batch-size 1 --epochs 1
```

## 📈 绘制损失曲线（可选）

### Python脚本

```python
# plot_history.py
import json
import matplotlib.pyplot as plt

with open('checkpoints/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(history['epochs'], history['train_loss'], 'b-o', label='Train Loss')
plt.plot(history['epochs'], history['val_loss'], 'r-s', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid()
plt.savefig('training_history.png')
print('✓ 图表已保存: training_history.png')
```

运行：
```bash
python plot_history.py
```

## 🎬 完整监控脚本

保存为 `monitor_training.sh`:

```bash
#!/bin/bash
# 实时监控训练进度和检查点生成

CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "🎬 开始监控训练..."
echo "Ctrl+C 退出"

while true; do
  clear
  
  # 标题和时间
  echo "╔════════════════════════════════════════════╗"
  echo "║  🎓 训练监控面板 ($(date '+%H:%M:%S'))  ║"
  echo "╚════════════════════════════════════════════╝"
  
  # 检查训练进程
  if pgrep -f "train_cli\|train_chinese" > /dev/null; then
    echo "✓ 训练进行中..."
  else
    echo "⚠️  训练未运行"
  fi
  
  echo ""
  echo "📊 检查点状态:"
  echo "────────────────────────────────────────────"
  
  # 列出最新的5个文件
  if [ -d "$CHECKPOINT_DIR" ]; then
    ls -lt "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -5 | while read -r line; do
      size=$(echo "$line" | awk '{print $5}')
      time=$(echo "$line" | awk '{print $6, $7, $8}')
      file=$(echo "$line" | awk '{print $NF}' | xargs basename)
      printf "  %-25s %10s  %s\n" "$file" "$size" "$time"
    done
  fi
  
  echo ""
  echo "📈 训练历史:"
  echo "────────────────────────────────────────────"
  
  # 显示训练历史
  if [ -f "$CHECKPOINT_DIR/training_history.json" ]; then
    python3 << 'EOF'
import json
try:
    with open('checkpoints/training_history.json') as f:
        h = json.load(f)
        for i, epoch in enumerate(h['epochs'][-3:]):
            tl = h['train_loss'][i-3] if len(h['train_loss']) >= 3 else h['train_loss'][i]
            vl = h['val_loss'][i-3] if len(h['val_loss']) >= 3 else h['val_loss'][i]
            marker = "⭐" if vl == min(h['val_loss']) else "  "
            print(f"  Epoch {epoch}: train={tl:.4f}, val={vl:.4f} {marker}")
except:
    pass
EOF
  fi
  
  echo ""
  echo "按 Ctrl+C 退出 | 每2秒刷新一次"
  sleep 2
done
```

使用：
```bash
bash monitor_training.sh
```

---

这样你就能完整地看到每个检查点的生成过程和文件大小变化了！
