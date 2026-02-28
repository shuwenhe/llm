#!/usr/bin/env python3
"""训练管理工具 - 查看检查点、恢复训练、分析历史"""
import json
import pickle
from pathlib import Path
from datetime import datetime
import argparse


def load_checkpoint(path):
    """加载检查点（仅支持 pickle 格式）"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"无法加载检查点 {path}: {e}\\n提示: 新版本仅支持 pickle 格式(.pkl)")


def list_checkpoints(checkpoint_dir="checkpoints"):
    """列出所有检查点"""
    checkpoint_dir = Path(checkpoint_dir)
    
    print("=" * 80)
    print("📦 检查点列表")
    print("=" * 80)
    
    checkpoints = []
    
    # 最佳模型
    best = checkpoint_dir / "best_model.pt"
    if best.exists():
        stat = best.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        try:
            ckpt = load_checkpoint(best)
            val_loss = ckpt.get('val_loss', ckpt.get('best_loss', 'N/A'))
            epoch = ckpt.get('epoch', 'N/A')
            checkpoints.append({
                'name': '🏆 best_model.pt',
                'epoch': epoch,
                'val_loss': val_loss,
                'size': size_mb,
                'time': mtime
            })
        except:
            pass
    
    # 最新模型
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        stat = latest.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        try:
            ckpt = load_checkpoint(latest)
            val_loss = ckpt.get('val_loss', 'N/A')
            epoch = ckpt.get('epoch', 'N/A')
            checkpoints.append({
                'name': '📌 latest.pt',
                'epoch': epoch,
                'val_loss': val_loss,
                'size': size_mb,
                'time': mtime
            })
        except:
            pass
    
    # Epoch检查点
    for ckpt_file in sorted(checkpoint_dir.glob("model_epoch_*.pt")):
        stat = ckpt_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        try:
            ckpt = load_checkpoint(ckpt_file)
            val_loss = ckpt.get('val_loss', 'N/A')
            epoch = ckpt.get('epoch', 'N/A')
            checkpoints.append({
                'name': f'📦 {ckpt_file.name}',
                'epoch': epoch,
                'val_loss': val_loss,
                'size': size_mb,
                'time': mtime
            })
        except:
            pass
    
    # 打印表格
    if checkpoints:
        print(f"{'文件':<30} {'Epoch':<8} {'验证损失':<12} {'大小(MB)':<10} {'修改时间':<20}")
        print("-" * 80)
        for ckpt in checkpoints:
            epoch_str = str(ckpt['epoch']) if ckpt['epoch'] != 'N/A' else 'N/A'
            loss_str = f"{ckpt['val_loss']:.4f}" if isinstance(ckpt['val_loss'], float) else str(ckpt['val_loss'])
            print(f"{ckpt['name']:<30} {epoch_str:<8} {loss_str:<12} {ckpt['size']:<10.2f} {ckpt['time'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("未找到检查点")
    
    print("=" * 80)


def show_history(checkpoint_dir="checkpoints"):
    """显示训练历史"""
    history_file = Path(checkpoint_dir) / "training_history.json"
    
    if not history_file.exists():
        print("❌ 未找到训练历史文件")
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    print("\n" + "=" * 80)
    print("📈 训练历史")
    print("=" * 80)
    print(f"{'Epoch':<8} {'训练损失':<15} {'验证损失':<15}")
    print("-" * 80)
    
    for i, epoch in enumerate(history['epochs']):
        train_loss = history['train_loss'][i]
        val_loss = history['val_loss'][i]
        print(f"{epoch:<8} {train_loss:<15.4f} {val_loss:<15.4f}")
    
    print("=" * 80)
    
    # 统计信息
    if history['val_loss']:
        best_epoch = history['epochs'][history['val_loss'].index(min(history['val_loss']))]
        best_loss = min(history['val_loss'])
        print(f"\n📊 统计:")
        print(f"  最佳Epoch: {best_epoch}")
        print(f"  最佳验证损失: {best_loss:.4f}")
        print(f"  总训练轮数: {len(history['epochs'])}")


def compare_checkpoints(ckpt1, ckpt2):
    """比较两个检查点"""
    print(f"\n🔍 比较检查点:")
    print(f"  模型1: {ckpt1}")
    print(f"  模型2: {ckpt2}")
    print("-" * 80)
    
    try:
        c1 = load_checkpoint(ckpt1)
        c2 = load_checkpoint(ckpt2)
        
        print(f"{'指标':<20} {'模型1':<20} {'模型2':<20}")
        print("-" * 60)
        print(f"{'Epoch':<20} {c1.get('epoch', 'N/A'):<20} {c2.get('epoch', 'N/A'):<20}")
        print(f"{'训练损失':<20} {c1.get('train_loss', 'N/A'):<20} {c2.get('train_loss', 'N/A'):<20}")
        print(f"{'验证损失':<20} {c1.get('val_loss', 'N/A'):<20} {c2.get('val_loss', 'N/A'):<20}")
        print(f"{'最佳损失':<20} {c1.get('best_loss', 'N/A'):<20} {c2.get('best_loss', 'N/A'):<20}")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")


def clean_old_checkpoints(checkpoint_dir="checkpoints", keep_n=3):
    """清理旧的epoch检查点"""
    checkpoint_dir = Path(checkpoint_dir)
    epoch_files = sorted(checkpoint_dir.glob("model_epoch_*.pt"))
    
    if len(epoch_files) <= keep_n:
        print(f"✓ 当前有 {len(epoch_files)} 个检查点，无需清理")
        return
    
    print(f"🗑️  清理旧检查点 (保留最近 {keep_n} 个)...")
    for old_file in epoch_files[:-keep_n]:
        size_mb = old_file.stat().st_size / (1024 * 1024)
        print(f"  删除: {old_file.name} ({size_mb:.2f} MB)")
        old_file.unlink()
    
    print(f"✓ 清理完成，剩余 {keep_n} 个检查点")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练管理工具")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="检查点目录")
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # list命令
    list_parser = subparsers.add_parser("list", help="列出所有检查点")
    
    # history命令
    history_parser = subparsers.add_parser("history", help="显示训练历史")
    
    # compare命令
    compare_parser = subparsers.add_parser("compare", help="比较两个检查点")
    compare_parser.add_argument("ckpt1", help="第一个检查点")
    compare_parser.add_argument("ckpt2", help="第二个检查点")
    
    # clean命令
    clean_parser = subparsers.add_parser("clean", help="清理旧检查点")
    clean_parser.add_argument("--keep", type=int, default=3, help="保留最近N个")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_checkpoints(args.checkpoint_dir)
    elif args.command == "history":
        show_history(args.checkpoint_dir)
    elif args.command == "compare":
        compare_checkpoints(args.ckpt1, args.ckpt2)
    elif args.command == "clean":
        clean_old_checkpoints(args.checkpoint_dir, args.keep)
    else:
        parser.print_help()
