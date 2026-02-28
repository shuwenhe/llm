#!/usr/bin/env python3
"""
应用优化配置的帮助脚本
"""

import os
import shutil

configs = {
    '1': {
        'name': '继续当前训练（推荐）',
        'desc': '从checkpoint继续，降低损失到7.0',
        'time': '1-2小时',
        'action': 'continue',
    },
    '2': {
        'name': '优化训练效率',
        'desc': '启用数据清洗 + 延长训练',
        'time': '2-4小时',
        'file': 'config_option2.py',
    },
    '3': {
        'name': '使用大数据集',
        'desc': 'WikiText-103（100倍数据）',
        'time': '5-10小时',
        'file': 'config_option3.py',
    },
    '4': {
        'name': '中等模型（117M）',
        'desc': '显著提升质量，需要GPU',
        'time': '1-2天',
        'file': 'config_option4.py',
    },
    '5': {
        'name': '大模型（345M）',
        'desc': '研究级别，需要强力GPU',
        'time': '3-7天',
        'file': 'config_option5.py',
    },
    '6': {
        'name': '快速实验',
        'desc': '小模型快速测试',
        'time': '15-30分钟',
        'file': 'config_option6.py',
    }
}


def show_menu():
    """显示选项菜单"""
    print("\n" + "="*60)
    print("🚀 LLM 优化配置助手")
    print("="*60)
    print("\n请选择优化方案：\n")
    
    for key, cfg in configs.items():
        print(f"{key}. {cfg['name']}")
        print(f"   {cfg['desc']}")
        print(f"   预计时间: {cfg['time']}")
        print()


def apply_config(choice):
    """应用选择的配置"""
    if choice not in configs:
        print("❌ 无效选择")
        return False
    
    cfg = configs[choice]
    
    if cfg.get('action') == 'continue':
        print("\n✅ 无需修改配置")
        print("\n直接运行以下命令继续训练：")
        print("   make train")
        return True
    
    # 备份当前配置
    if os.path.exists('config.py') and not os.path.exists('config.py.backup'):
        shutil.copy('config.py', 'config.py.backup')
        print("✓ 已备份当前配置到 config.py.backup")
    
    print(f"\n✅ 应用配置：{cfg['name']}")
    print(f"\n⚠️  注意事项：")
    print(f"   - 预计时间: {cfg['time']}")
    print(f"   - 查看详细说明: docs/OPTIMIZATION.md")
    
    if choice in ['3', '4', '5']:
        print(f"   - 建议删除旧checkpoint: rm checkpoints/best_model.pt")
    
    print(f"\n查看配置选项详情:")
    print(f"   python3 -c \"from configs_options import *; help(TrainConfig_Option{choice})\"")
    
    print(f"\n📝 手动应用步骤:")
    print(f"   1. 打开 configs_options.py")
    print(f"   2. 复制 ModelConfig_Option{choice} 和 TrainConfig_Option{choice}")
    print(f"   3. 粘贴到 config.py，重命名为 ModelConfig 和 TrainConfig")
    print(f"   4. 运行: make train")
    
    return True


def main():
    show_menu()
    
    choice = input("请输入选择 (1-6) 或 'q' 退出: ").strip()
    
    if choice.lower() == 'q':
        print("退出")
        return
    
    if apply_config(choice):
        print("\n" + "="*60)
        print("✨ 准备就绪！")
        print("="*60)


if __name__ == "__main__":
    main()
