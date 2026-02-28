"""中文文本训练 - 使用 train_core 作为后端"""
from app.training.train_core import main as train_core_main

if __name__ == "__main__":
    import sys
    print("ℹ️  train_chinese: 使用 train_core 作为后端")
    print("ℹ️  如需中文语料训练，请准备中文文本文件并使用相应参数\n")
    train_core_main()
