# 工业级训练命令参考

## 核心命令模板

### 模板1：快速验证管道
```bash
python train_cli.py --preset quick
```
✅ 场景：第一次运行、测试环境是否正常

### 模板2：标准生产训练
```bash
python train_cli.py --preset standard --data-file data/zh_sample.txt
```
✅ 场景：日常使用、中等质量模型

### 模板3：高精度部署
```bash
python train_cli.py --preset precision --data-file data/zh_wiki.txt
```
✅ 场景：生产部署、最优模型质量

### 模板4：自定义训练
```bash
python train_cli.py \
  --batch-size 8 \
  --epochs 5 \
  --learning-rate 5e-5 \
  --data-file data/zh_sample.txt \
  --keep-last-n 5
```
✅ 场景：超参数调优、实验研究

---

## 实战命令集

### 场景1：我要立即开始训练

**第一步：快速验证**
```bash
cd /home/shuwen/llm
python train_cli.py --preset quick --dry-run
```

**第二步：执行快速训练**
```bash
python train_cli.py --preset quick
```

**第三步：查看结果**
```bash
python train_manager.py list
python train_manager.py history
```

---

### 场景2：我要标准生产训练

```bash
# 列出可用数据
ls -lh data/

# 查看预设
python train_cli.py --list-presets

# 执行标准训练
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 监控日志
tail -f logs/training_*.log

# 训练完成后查看结果
python train_manager.py history
```

---

### 场景3：我要高精度模型

```bash
# 假设已有大数据集
python train_cli.py \
  --preset precision \
  --data-file data/zh_wiki.txt \
  --epochs 30

# 实时查看训练日志
tail -f logs/training_*.log

# 训练完成，部署最佳模型
LLM_CHECKPOINT=checkpoints/best_model.pt make serve
```

---

### 场景4：我要恢复中断的训练

```bash
# 查看之前的检查点
python train_manager.py list

# 从latest恢复并继续
python train_cli.py --preset extended --resume

# 或延长训练时间
python train_cli.py --preset precision --resume --epochs 30
```

---

### 场景5：我要对比不同配置

```bash
# 快速vs标准 - 先看命令
python train_cli.py --preset quick --dry-run
python train_cli.py --preset standard --dry-run

# 执行快速
python train_cli.py --preset quick

# 查看结果
python train_manager.py history
python train_manager.py compare checkpoints/best_model.pt checkpoints/model_epoch_1.pt

# 执行标准
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 对比最终结果
python train_manager.py history
```

---

## 参数速查表

| 参数 | 含义 | 示例 | 默认值 |
|-----|------|------|--------|
| `--preset` | 预设配置 | `quick`, `standard`, `extended`, `precision` | - |
| `--batch-size` | 批次大小 | `2`, `4`, `8`, `16` | 来自预设 |
| `--epochs` | 训练轮数 | `1`, `3`, `10`, `20` | 来自预设 |
| `--learning-rate` | 学习率 | `1e-4`, `5e-5`, `1e-5` | 来自预设 |
| `--data-file` | 数据文件 | `data/zh_sample.txt` | - |
| `--keep-last-n` | 保留最后N个检查点 | `3`, `5`, `10` | 来自预设 |
| `--config` | 配置文件路径 | `config/my_config.json` | - |
| `--save-config` | 保存配置到文件 | `my_config.json` | - |
| `--resume` | 从latest.pt恢复 | `--resume` | False |
| `--dry-run` | 不执行只打印 | `--dry-run` | False |
| `--no-log` | 不生成日志 | `--no-log` | False |
| `--list-presets` | 列出所有预设 | `--list-presets` | - |

---

## OpenAI风格对标

### OpenAI API 训练 vs 本地训练

```bash
# OpenAI风格（云端）
openai api fine_tunes.create \
  --training_file file-abc123 \
  --model gpt-3.5-turbo \
  --n_epochs 3

# 本系统（本地）
python train_cli.py \
  --preset standard \
  --data-file data/zh_sample.txt \
  --epochs 3
```

两者对标功能：
- ✅ 预设配置 → 不同的训练规模
- ✅ 配置管理 → 实验重现性
- ✅ 日志记录 → 完整训练历史
- ✅ 检查点保存 → 模型版本管理
- ✅ 断点续训 → 灵活的训练流程

---

## 生产部署checklist

```bash
#!/bin/bash
# 完整的生产训练流程

set -e

PROJECT_DIR="/home/shuwen/llm"
cd $PROJECT_DIR

echo "=== 步骤1：验证环境 ==="
python train_cli.py --list-presets

echo "=== 步骤2：快速测试管道 ==="
python train_cli.py --preset quick

echo "=== 步骤3：标准训练 ==="
python train_cli.py --preset standard --data-file data/zh_sample.txt

echo "=== 步骤4：验证检查点 ==="
python train_manager.py list

echo "=== 步骤5：查看训练历史 ==="
python train_manager.py history

echo "=== 步骤6：清理旧检查点 ==="
python train_manager.py clean

echo "=== 步骤7：准备部署 ==="
echo "最佳模型在: checkpoints/best_model.pt"
echo "使用命令部署: LLM_CHECKPOINT=checkpoints/best_model.pt make serve"

echo "✅ 生产训练流程完成！"
```

运行：
```bash
bash deploy_training.sh
```

---

## 故障诊断

### 命令无法执行
```bash
# 检查Python环境
python --version
./venv/bin/python --version

# 检查依赖
pip list | grep -E "(torch|transformers)"

# 检查文件权限
ls -l train_cli.py
```

### 显存不足
```bash
# 使用quick预设
python train_cli.py --preset quick

# 或手动降低batch_size
python train_cli.py --batch-size 1 --epochs 1
```

### 训练效果不好
```bash
# 查看损失曲线
python train_manager.py history

# 对比不同配置
python train_manager.py compare model1.pt model2.pt

# 尝试不同学习率
python train_cli.py --learning-rate 1e-3 --epochs 3
python train_cli.py --learning-rate 1e-5 --epochs 3
```

---

## 与CI/CD集成示例

### GitHub Actions
```yaml
name: Model Training

on:
  schedule:
    - cron: '0 2 * * *'  # 每天02:00运行

jobs:
  train:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v2
      - name: Run training
        run: |
          cd /home/shuwen/llm
          python train_cli.py --preset precision --data-file data/zh_wiki.txt
      - name: Check results
        run: |
          cd /home/shuwen/llm
          python train_manager.py history
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: training-logs
          path: logs/
```

### Jenkins
```groovy
pipeline {
    agent any
    
    schedule {
        cron('0 2 * * *')  // 每天02:00
    }
    
    stages {
        stage('Train') {
            steps {
                sh 'cd /home/shuwen/llm && python train_cli.py --preset precision'
            }
        }
        stage('Verify') {
            steps {
                sh 'cd /home/shuwen/llm && python train_manager.py history'
            }
        }
    }
}
```

---

## 性能基准

在15.6GB VRAM的GPU上（以zh_sample.txt为例）：

| 预设 | 批次 | 轮数 | 时间 | 显存 | 状态 |
|-----|------|------|-----|------|------|
| QUICK | 2 | 1 | ~1分钟 | ~3GB | ✅ 验证 |
| STANDARD | 4 | 3 | ~30分钟 | ~5GB | ✅ 生产 |
| EXTENDED | 8 | 10 | ~2小时 | ~8GB | ✅ 部署 |
| PRECISION | 16 | 20 | ~5小时 | ~12GB | ✅ 最优 |

---

## 常见组合命令

```bash
# 1. 快速验证 → 标准训练
python train_cli.py --preset quick && \
python train_cli.py --preset standard --data-file data/zh_sample.txt

# 2. 查看所有信息
python train_cli.py --list-presets && \
python train_manager.py list && \
python train_manager.py history

# 3. 完整流程
python train_cli.py --preset quick --dry-run && \
python train_cli.py --preset quick && \
python train_manager.py history

# 4. 高精度训练
python train_cli.py --preset precision \
  --data-file data/zh_wiki.txt \
  --save-config production_model.json
```

---

## 后续步骤

1. ✅ 学习基础命令 → `python train_cli.py --preset quick`
2. ✅ 尝试标准训练 → `python train_cli.py --preset standard`
3. ✅ 检查结果 → `python train_manager.py history`
4. ✅ 优化参数 → 自定义命令
5. ✅ 部署最优模型 → `make serve`
