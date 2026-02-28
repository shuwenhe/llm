#!/usr/bin/env python3
"""OpenAI风格的工业级训练命令 - 支持配置、日志、监控"""
import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class TrainingConfig:
    """训练配置管理"""
    
    PRESETS = {
        # 快速验证
        "quick": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-4,
            "save_every_epoch": True,
            "keep_last_n": 1,
        },
        # 标准训练
        "standard": {
            "batch_size": 4,
            "epochs": 3,
            "learning_rate": 1e-4,
            "save_every_epoch": True,
            "keep_last_n": 3,
        },
        # 长期训练
        "extended": {
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 5e-5,
            "save_every_epoch": True,
            "keep_last_n": 5,
        },
        # 高精度训练
        "precision": {
            "batch_size": 16,
            "epochs": 20,
            "learning_rate": 1e-5,
            "save_every_epoch": True,
            "keep_last_n": 10,
        },
    }
    
    @classmethod
    def from_preset(cls, preset_name: str) -> dict:
        """从预设加载配置"""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"未知的预设: {preset_name}. 可用: {list(cls.PRESETS.keys())}")
        return cls.PRESETS[preset_name]
    
    @classmethod
    def from_file(cls, config_file: str) -> dict:
        """从配置文件加载"""
        with open(config_file) as f:
            return json.load(f)
    
    @classmethod
    def save_config(cls, config: dict, output_file: str):
        """保存配置"""
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ 配置已保存: {output_file}")


class TrainingLogger:
    """训练日志管理"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.log"
        self.config_file = self.log_dir / f"config_{timestamp}.json"
    
    def log(self, message: str):
        """记录日志"""
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")
        print(message)
    
    def save_config(self, config: dict):
        """保存配置"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_summary(self) -> dict:
        """获取日志摘要"""
        return {
            'log_file': str(self.log_file),
            'config_file': str(self.config_file),
            'timestamp': datetime.now().isoformat(),
        }


def build_training_command(config: dict, data_file: Optional[str] = None) -> list:
    """构建训练命令"""
    cmd = [
        "./venv/bin/python",
        "train_chinese.py",
    ]
    
    # 添加参数
    if data_file:
        cmd.extend(["--data-file", data_file])
    
    cmd.extend([
        "--batch-size", str(config.get("batch_size", 4)),
        "--epochs", str(config.get("epochs", 3)),
        "--learning-rate", str(config.get("learning_rate", 1e-4)),
    ])
    
    if not config.get("save_every_epoch", True):
        cmd.append("--no-save-every-epoch")
    
    if "keep_last_n" in config:
        cmd.extend(["--keep-last-n", str(config["keep_last_n"])])
    
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI风格的工业级训练命令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速验证
  python train_cli.py --preset quick

  # 标准训练
  python train_cli.py --preset standard --data-file data/zh_sample.txt

  # 自定义训练
  python train_cli.py --batch-size 8 --epochs 5 --learning-rate 1e-5

  # 从配置文件训练
  python train_cli.py --config config.json

  # 列出可用预设
  python train_cli.py --list-presets
        """
    )
    
    # 预设选项
    preset_group = parser.add_argument_group("预设配置")
    preset_group.add_argument(
        "--preset",
        choices=list(TrainingConfig.PRESETS.keys()),
        help="使用预设配置 (quick, standard, extended, precision)"
    )
    preset_group.add_argument(
        "--list-presets",
        action="store_true",
        help="列出所有可用预设"
    )
    
    # 配置选项
    config_group = parser.add_argument_group("配置管理")
    config_group.add_argument(
        "--config",
        help="从JSON配置文件加载"
    )
    config_group.add_argument(
        "--save-config",
        help="保存当前配置到文件"
    )
    
    # 训练参数
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument(
        "--batch-size",
        type=int,
        help="批次大小"
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        help="训练轮数"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        help="学习率"
    )
    train_group.add_argument(
        "--data-file",
        help="训练数据文件"
    )
    train_group.add_argument(
        "--keep-last-n",
        type=int,
        help="保留最近N个检查点"
    )
    train_group.add_argument(
        "--no-save-every-epoch",
        action="store_true",
        help="不保存每个epoch的检查点"
    )
    
    # 执行选项
    exec_group = parser.add_argument_group("执行选项")
    exec_group.add_argument(
        "--dry-run",
        action="store_true",
        help="打印命令但不执行"
    )
    exec_group.add_argument(
        "--resume",
        action="store_true",
        help="从最新检查点恢复训练"
    )
    exec_group.add_argument(
        "--no-log",
        action="store_true",
        help="不记录日志"
    )
    
    args = parser.parse_args()
    
    # 列出预设
    if args.list_presets:
        print("\n📋 可用的训练预设:")
        print("=" * 80)
        for name, config in TrainingConfig.PRESETS.items():
            print(f"\n{name.upper()}")
            print(f"  批次大小: {config['batch_size']}")
            print(f"  训练轮数: {config['epochs']}")
            print(f"  学习率: {config['learning_rate']}")
        print()
        return
    
    # 加载配置
    config = {}
    
    if args.config:
        config = TrainingConfig.from_file(args.config)
        print(f"✓ 从配置文件加载: {args.config}")
    elif args.preset:
        config = TrainingConfig.from_preset(args.preset)
        print(f"✓ 使用预设: {args.preset}")
    else:
        config = TrainingConfig.from_preset("standard")
        print(f"✓ 使用默认预设: standard")
    
    # 覆盖命令行参数
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.epochs:
        config["epochs"] = args.epochs
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.keep_last_n:
        config["keep_last_n"] = args.keep_last_n
    if args.no_save_every_epoch:
        config["save_every_epoch"] = False
    
    # 添加resume选项
    if args.resume:
        config["resume"] = True
    
    # 初始化日志
    logger = None
    if not args.no_log:
        logger = TrainingLogger()
        logger.save_config(config)
    
    # 打印配置
    print("\n" + "=" * 80)
    print("🎓 训练配置")
    print("=" * 80)
    for key, value in config.items():
        if key not in ["resume"]:
            print(f"  {key}: {value}")
    print("=" * 80 + "\n")
    
    # 构建命令
    cmd = build_training_command(config, args.data_file)
    
    # 添加resume标志
    if args.resume:
        cmd.append("--resume")
    
    if logger:
        logger.log(f"命令: {' '.join(cmd)}")
    
    # 打印命令
    print(f"📝 执行命令: {' '.join(cmd)}\n")
    
    if args.dry_run:
        print("✓ 干运行模式 (不执行)")
        if logger:
            logger.log("干运行模式")
        return
    
    # 执行训练
    try:
        if logger:
            logger.log("=" * 80)
            logger.log("开始训练")
            logger.log("=" * 80)
        
        result = subprocess.run(cmd, check=True)
        
        if logger:
            logger.log("=" * 80)
            logger.log("✓ 训练完成")
            logger.log(f"📊 日志: {logger.get_summary()['log_file']}")
            logger.log("=" * 80)
        
        sys.exit(result.returncode)
        
    except subprocess.CalledProcessError as e:
        if logger:
            logger.log(f"❌ 训练失败: {e}")
        print(f"❌ 训练失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        if logger:
            logger.log("⚠️  训练被中断")
        print("\n⚠️  训练被中断")
        sys.exit(130)


if __name__ == "__main__":
    main()
