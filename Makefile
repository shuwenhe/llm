.PHONY: help install test train generate clean clean-checkpoints clean-all

# Python解释器
PYTHON := python3

# 默认目标
help:
	@echo "LLM项目 - 可用命令:"
	@echo ""
	@echo "  make install          - 安装项目依赖"
	@echo "  make test             - 运行模型测试"
	@echo "  make train            - 开始训练模型"
	@echo "  make generate         - 运行文本生成"
	@echo "  make clean            - 清理Python缓存文件"
	@echo "  make clean-checkpoints - 删除所有checkpoint文件"
	@echo "  make clean-all        - 清理所有生成文件"
	@echo ""
	@echo "  make setup            - 完整设置(创建venv+安装依赖)"
	@echo "  make quick-test       - 快速测试(小数据集)"
	@echo ""

# 安装依赖
install:
	@echo "安装项目依赖..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ 依赖安装完成"

# 创建虚拟环境并安装依赖
setup:
	@echo "创建虚拟环境..."
	$(PYTHON) -m venv venv
	@echo "请激活虚拟环境后运行: make install"
	@echo ""
	@echo "Linux/Mac: source venv/bin/activate"
	@echo "Windows:   venv\\Scripts\\activate"

# 运行模型测试
test:
	@echo "运行模型测试..."
	$(PYTHON) test_model.py

# 训练模型
train:
	@echo "开始训练模型..."
	$(PYTHON) train.py

# 文本生成
generate:
	@echo "启动文本生成..."
	$(PYTHON) generate.py

# 快速测试（用于验证代码）
quick-test:
	@echo "快速测试模式..."
	$(PYTHON) -c "from model import GPT; from config import ModelConfig; \
		config = ModelConfig(n_layer=2, n_head=2, n_embd=128); \
		model = GPT(config); \
		print(f'模型参数: {model.get_num_params()/1e6:.2f}M'); \
		print('✓ 模型创建成功')"

# 清理Python缓存
clean:
	@echo "清理Python缓存..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ 缓存清理完成"

# 清理checkpoint文件
clean-checkpoints:
	@echo "删除checkpoint文件..."
	rm -rf checkpoints/*.pt
	@echo "✓ Checkpoint清理完成"

# 清理所有生成文件
clean-all: clean clean-checkpoints
	@echo "清理所有生成文件..."
	rm -rf logs/
	rm -rf wandb/
	rm -rf runs/
	rm -rf data/
	@echo "✓ 完全清理完成"

# 查看模型信息
info:
	@echo "模型信息:"
	@$(PYTHON) -c "from model import GPT; from config import ModelConfig; \
		config = ModelConfig(); \
		model = GPT(config); \
		print(f'参数量: {model.get_num_params()/1e6:.2f}M'); \
		print(f'层数: {config.n_layer}'); \
		print(f'嵌入维度: {config.n_embd}'); \
		print(f'注意力头数: {config.n_head}'); \
		print(f'序列长度: {config.block_size}')"

# 检查依赖
check-deps:
	@echo "检查依赖安装情况..."
	@$(PYTHON) -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "✗ PyTorch未安装"
	@$(PYTHON) -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" || echo "✗ Transformers未安装"
	@$(PYTHON) -c "import datasets; print(f'✓ Datasets {datasets.__version__}')" || echo "✗ Datasets未安装"
	@$(PYTHON) -c "import torch; print(f'✓ CUDA可用') if torch.cuda.is_available() else print('○ CUDA不可用')"

# 创建必要的目录
init:
	@echo "创建项目目录..."
	mkdir -p checkpoints
	mkdir -p logs
	mkdir -p data
	@echo "✓ 目录创建完成"
