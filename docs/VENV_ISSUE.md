# 虚拟环境问题解决方案

## 问题描述

在 Linux 系统（特别是 Ubuntu 23.04+, Debian 12+）上运行 `make install` 时，出现以下错误：

```
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.
```

## 原因

这是 PEP 668 引入的安全特性，防止用户破坏系统 Python 环境。必须使用虚拟环境来安装 Python 包。

## ✅ 解决方案

### 方案 1：使用自动设置脚本（最简单）

```bash
# 运行设置脚本
./setup.sh

# 激活虚拟环境
source venv/bin/activate

# 验证安装
make check-deps
```

### 方案 2：使用 Makefile 命令

```bash
# 一键创建虚拟环境并安装依赖
make setup-all

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 验证安装
make check-deps
```

### 方案 3：分步手动操作

```bash
# 1. 确保安装了 venv 模块
sudo apt update
sudo apt install python3-full python3-venv

# 2. 创建虚拟环境
make setup
# 或者: python3 -m venv venv

# 3. 激活虚拟环境
source venv/bin/activate

# 4. 安装依赖
make install
# 或者: pip install -r requirements.txt

# 5. 验证安装
make check-deps
```

## 📝 使用流程

安装完成后，每次使用项目都需要先激活虚拟环境：

```bash
# 进入项目目录
cd llm

# 激活虚拟环境
source venv/bin/activate

# 现在可以运行任何命令
make test
make train
python generate.py

# 完成后可以退出虚拟环境
deactivate
```

## 🔍 验证是否在虚拟环境中

```bash
# 方法 1: 查看命令提示符
# 激活后会显示: (venv) user@host:~/llm$

# 方法 2: 检查 Python 路径
which python
# 应该显示: /path/to/llm/venv/bin/python

# 方法 3: 检查环境变量
echo $VIRTUAL_ENV
# 应该显示: /path/to/llm/venv
```

## ⚠️ 注意事项

1. **不要使用 `sudo pip install`** - 这会污染系统 Python
2. **每次使用项目都要激活虚拟环境**
3. **虚拟环境是项目特定的** - 不同项目应该有各自的虚拟环境
4. **虚拟环境可以删除重建** - 只是包含安装的依赖

## 🚫 不推荐的方案

```bash
# 使用 --break-system-packages（可能破坏系统）
make install-force

# 直接修改系统配置（危险）
sudo rm /usr/lib/python3.*/EXTERNALLY-MANAGED
```

## 📚 更多信息

- 详细安装指南: [INSTALL.md](INSTALL.md)
- 项目文档: [README.md](../README.md)
- PEP 668 说明: https://peps.python.org/pep-0668/

## 🆘 仍然有问题？

1. 确保安装了 `python3-venv`:
   ```bash
   sudo apt install python3-full python3-venv
   ```

2. 检查 Python 版本（需要 3.8+):
   ```bash
   python3 --version
   ```

3. 完全清理后重试:
   ```bash
   rm -rf venv
   make setup-all
   source venv/bin/activate
   ```

4. 查看完整的错误信息并搜索具体问题
