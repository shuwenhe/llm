#!/bin/bash

# LLM 项目自动设置脚本
# 自动创建虚拟环境并安装依赖

set -e  # 遇到错误时退出

echo "=================================="
echo "LLM 项目环境设置"
echo "=================================="
echo ""

# 检测操作系统
OS="$(uname -s)"
case "$OS" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    CYGWIN*|MINGW*|MSYS*)    OS_TYPE=Windows;;
    *)          OS_TYPE="Unknown";;
esac

echo "检测到操作系统: $OS_TYPE"
echo ""

# 检查 Python 版本
echo "检查 Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    echo "请先安装 Python 3.8 或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python 版本: $PYTHON_VERSION"
echo ""

# 检查是否已存在虚拟环境
if [ -d "venv" ]; then
    echo "⚠️  检测到已存在的虚拟环境"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧的虚拟环境..."
        rm -rf venv
    else
        echo "使用现有虚拟环境"
    fi
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    echo "✓ 虚拟环境创建成功"
else
    echo "✓ 使用现有虚拟环境"
fi
echo ""

# 激活虚拟环境并安装依赖
echo "安装依赖..."
echo "这可能需要几分钟时间..."
echo ""

# 根据操作系统选择激活脚本
if [ "$OS_TYPE" = "Windows" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# 升级 pip
echo "升级 pip..."
python -m pip install --upgrade pip -q

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

echo ""
echo "=================================="
echo "✅ 设置完成！"
echo "=================================="
echo ""
echo "下一步："
echo ""
if [ "$OS_TYPE" = "Windows" ]; then
    echo "1. 激活虚拟环境:"
    echo "   venv\\Scripts\\activate"
else
    echo "1. 激活虚拟环境:"
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. 测试模型:"
echo "   make test  (或 python test_model.py)"
echo ""
echo "3. 开始训练:"
echo "   make train  (或 python train.py)"
echo ""
echo "4. 查看更多命令:"
echo "   make help"
echo ""
