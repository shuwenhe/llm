#!/bin/bash
# 快速训练测试（小数据集）
echo "🧪 快速训练测试 (10步验证)"
./venv/bin/python -c "
from app.training.train_core import train_core
# 临时创建小数据集
import os
os.makedirs('data', exist_ok=True)
with open('data/test_sample.txt', 'w', encoding='utf-8') as f:
    for _ in range(10):
        f.write('这是一个测试句子，用于快速验证训练流程是否正常工作。\n')

train_core(
    learning_rate=1e-4,
    batch_size=2,
    epochs=1,
    output='checkpoints/model_core.pkl'
)
"
