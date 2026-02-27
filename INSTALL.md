# 安装指南

本指南帮助你在不同操作系统上正确设置 LLM 开发环境。

## 📋 目录

- [快速开始](#快速开始)
- [Linux/Ubuntu](#linuxubuntu)
- [macOS](#macos)
- [Windows](#windows)
- [常见问题](#常见问题)

## 🚀 快速开始

### 最简单的方式（推荐）

```bash
# 进入项目目录
cd llm

# 一键设置（创建虚拟环境并安装依赖）
make setup-all

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 测试安装
make test
```

## 💻 Linux/Ubuntu

### 问题：externally-managed-environment 错误

在 Ubuntu 23.04+ 和 Debian 12+ 上，系统 Python 被标记为"外部管理"，必须使用虚拟环境。

#### 解决方案 1：使用虚拟环境（推荐）

```bash
# 确保安装了 python3-venv
sudo apt update
sudo apt install python3-full python3-venv

# 使用 Makefile 一键设置
make setup-all

# 激活虚拟环境
source venv/bin/activate

# 验证安装
make check-deps
```

#### 解决方案 2：手动创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

#### 解决方案 3：强制安装（不推荐）

```bash
# 只在确实需要时使用
make install-force
```

### 安装 CUDA（GPU 加速，可选）

如果你有 NVIDIA GPU：

```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应的 PyTorch CUDA 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🍎 macOS

### 使用 Homebrew（推荐）

```bash
# 安装 Python 3
brew install python@3.11

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 使用 Makefile 安装
make install
```

### Apple Silicon (M1/M2/M3) GPU 加速

```bash
# PyTorch 会自动支持 MPS (Metal Performance Shaders)
# 训练时模型会自动使用 GPU

# 验证 MPS 可用
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## 🪟 Windows

### 使用 PowerShell

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 升级 pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 使用 CUDA（GPU 加速）

```bash
# 先安装 CUDA Toolkit
# 下载地址: https://developer.nvidia.com/cuda-downloads

# 安装 PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 使用 WSL2（推荐）

在 Windows 上，推荐使用 WSL2 获得更好的性能：

```bash
# 在 PowerShell 中安装 WSL2
wsl --install

# 进入 WSL2
wsl

# 按照 Linux 安装步骤操作
```

## 🔧 常见问题

### Q: 如何知道我是否在虚拟环境中？

```bash
# 方法 1: 检查提示符
# 虚拟环境激活后，命令提示符前会显示 (venv)

# 方法 2: 检查环境变量
echo $VIRTUAL_ENV  # Linux/Mac
echo %VIRTUAL_ENV%  # Windows

# 方法 3: 检查 Python 路径
which python  # Linux/Mac
where python  # Windows
```

### Q: 如何退出虚拟环境？

```bash
deactivate
```

### Q: 虚拟环境可以删除吗？

可以，删除后重新创建：

```bash
# 退出虚拟环境
deactivate

# 删除虚拟环境目录
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows

# 重新创建
make setup-all
```

### Q: 如何检查依赖是否正确安装？

```bash
# 使用 Makefile
make check-deps

# 或手动检查
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

### Q: pip 安装太慢怎么办？

使用国内镜像源（中国用户）：

```bash
# 临时使用
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

推荐镜像源：
- 清华：https://pypi.tuna.tsinghua.edu.cn/simple
- 阿里云：https://mirrors.aliyun.com/pypi/simple
- 中科大：https://pypi.mirrors.ustc.edu.cn/simple

### Q: ModuleNotFoundError 错误？

确保：
1. 虚拟环境已激活
2. 依赖已安装
3. 使用正确的 Python 解释器

```bash
# 检查 Python 路径
which python

# 重新安装依赖
pip install -r requirements.txt
```

### Q: 权限错误 (Permission Denied)？

```bash
# 不要使用 sudo pip install
# 而是使用虚拟环境：
source venv/bin/activate
pip install -r requirements.txt
```

## 📦 依赖版本说明

主要依赖及其最低版本：

- Python: 3.8+
- PyTorch: 2.0.0+
- Transformers: 4.30.0+
- Datasets: 2.12.0+

完整依赖列表见 [requirements.txt](requirements.txt)

## 🧪 验证安装

安装完成后，运行测试脚本：

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/Mac

# 运行测试
make test

# 或直接运行
python test_model.py
```

如果看到 "✅ 所有测试通过！"，说明安装成功。

## 🆘 获取帮助

如果遇到问题：

1. 查看 [README.md](README.md) 的常见问题部分
2. 确保使用了最新版本的代码
3. 检查 Python 版本：`python --version`
4. 检查依赖安装：`make check-deps`

## 📚 下一步

安装完成后：

1. 阅读 [README.md](README.md) 了解项目结构
2. 运行 `make test` 测试模型
3. 运行 `make train` 开始训练
4. 查看 [config.py](config.py) 自定义模型配置
