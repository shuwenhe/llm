#!/bin/bash
# 快速训练测试（小数据集）
echo "🧪 快速训练测试 (10步验证)"
./venv/bin/python -c "
import torch
from train_chinese import train_chinese_text
# 临时创建小数据集
import os
os.makedirs('data', exist_ok=True)
with open('data/test_sample.txt', 'w', encoding='utf-8') as f:
    for _ in range(10):
        f.write('这是一个测试句子，用于快速验证训练流程是否正常工作。\n')

train_chinese_text(
    learning_rate=1e-4,
    batch_size=2,
    num_epochs=1,
    checkpoint_path='checkpoints/model.pt',
    output_path='checkpoints/model.pt',
    data_file='data/test_sample.txt'
)
"
