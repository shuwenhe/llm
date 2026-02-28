.PHONY: help install test train train-multimodal serve serve-dev generate quick-generate quick-test-multimodal clean clean-checkpoints clean-all

# Python解释器（优先使用项目内虚拟环境）
PYTHON := $(shell if [ -x ./venv/bin/python ]; then echo ./venv/bin/python; else echo python3; fi)

# 默认目标
help:
	@echo "LLM项目 - 可用命令:"
	@echo ""
	@echo "环境设置:"
	@echo "  make setup            - 创建虚拟环境"
	@echo "  make setup-all        - 创建虚拟环境并安装依赖(推荐)"
	@echo "  make install          - 安装依赖(需要先激活虚拟环境)"
	@echo "  make install-force    - 强制安装(不推荐，跳过虚拟环境检查)"
	@echo ""
	@echo "开发与训练:"
	@echo "  make test             - 运行模型测试"
	@echo "  make train            - 开始训练模型"
	@echo "  make train-multimodal - 开始完整多模态训练(文本+图像+语音)"
	@echo "  make serve            - 启动推理API服务(生产模式)"
	@echo "  make serve-dev        - 启动推理API服务(开发热更新)"
	@echo "  make generate         - 运行交互式文本生成"
	@echo "  make quick-generate   - 批量测试生成参数"
	@echo "  make quick-test       - 快速测试(验证模型可用)"
	@echo "  make quick-test-multimodal - 快速测试多模态前向(文本+图像+语音)"
	@echo ""
	@echo "工具:"
	@echo "  make info             - 查看模型配置信息"
	@echo "  make check-deps       - 检查依赖安装情况"
	@echo "  make init             - 创建必要的项目目录"
	@echo ""
	@echo "清理:"
	@echo "  make clean            - 清理Python缓存文件"
	@echo "  make clean-checkpoints - 删除所有checkpoint文件"
	@echo "  make clean-all        - 清理所有生成文件"
	@echo ""

# 检查是否在虚拟环境中
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ 错误：未检测到虚拟环境！"; \
		echo ""; \
		echo "请先创建并激活虚拟环境："; \
		echo "  方式1: make setup && source venv/bin/activate"; \
		echo "  方式2: python3 -m venv venv && source venv/bin/activate"; \
		echo ""; \
		echo "然后再运行: make install"; \
		exit 1; \
	fi

# 安装依赖（需要在虚拟环境中）
install: check-venv
	@echo "安装项目依赖..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ 依赖安装完成"

# 强制安装（不检查虚拟环境，不推荐）
install-force:
	@echo "⚠️  强制安装依赖（不推荐，可能影响系统Python）..."
	$(PYTHON) -m pip install --upgrade pip --break-system-packages
	$(PYTHON) -m pip install -r requirements.txt --break-system-packages
	@echo "✓ 依赖安装完成"

# 创建虚拟环境并安装依赖
setup:
	@echo "创建虚拟环境..."
	$(PYTHON) -m venv venv
	@echo "✓ 虚拟环境创建完成: venv/"
	@echo ""
	@echo "激活虚拟环境："
	@echo "  Linux/Mac: source venv/bin/activate"
	@echo "  Windows:   venv\\Scripts\\activate"
	@echo ""
	@echo "然后运行: make install"

# 一键安装（创建venv并安装依赖）
setup-all:
	@echo "创建虚拟环境并安装依赖..."
	@if [ ! -d "venv" ]; then \
		echo "创建虚拟环境..."; \
		$(PYTHON) -m venv venv; \
	fi
	@echo "安装依赖到虚拟环境..."
	@./venv/bin/pip install --upgrade pip
	@./venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "✓ 设置完成！"
	@echo ""
	@echo "激活虚拟环境："
	@echo "  source venv/bin/activate"
	@echo ""
	@echo "然后可以运行："
	@echo "  make test    # 测试模型"
	@echo "  make train   # 训练模型"

# 运行模型测试
test:
	@echo "运行模型测试..."
	$(PYTHON) test_model.py

# 训练模型
train:
	@echo "开始训练模型..."
	$(PYTHON) train.py

# 多模态训练
train-multimodal:
	@echo "开始多模态训练模型..."
	LLM_MULTIMODAL=1 $(PYTHON) train.py

# 推理服务（生产）
serve:
	@echo "启动推理API服务(生产模式)..."
	$(PYTHON) -m uvicorn serve:app --host 0.0.0.0 --port 8000

# 推理服务（开发）
serve-dev:
	@echo "启动推理API服务(开发模式)..."
	$(PYTHON) -m uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

# 文本生成
generate:
	@echo "启动文本生成..."
	$(PYTHON) generate.py

# 快速生成测试（批量测试不同参数）
quick-generate:
	@echo "批量测试生成参数..."
	$(PYTHON) quick_generate.py

# 快速测试（用于验证代码）
quick-test:
	@echo "快速测试模式..."
	$(PYTHON) -c "from model import GPT; from config import ModelConfig; \
		config = ModelConfig(n_layer=2, n_head=2, n_embd=128); \
		model = GPT(config); \
		print(f'模型参数: {model.get_num_params()/1e6:.2f}M'); \
		print('✓ 模型创建成功')"

# 多模态快速测试（随机输入）
quick-test-multimodal:
	@echo "多模态快速测试模式..."
	$(PYTHON) -c "import torch; from model import GPT; from config import ModelConfig; \
		config = ModelConfig(multimodal_enabled=True, n_layer=2, n_head=2, n_embd=128, block_size=256); \
		model = GPT(config); \
		idx = torch.randint(0, config.vocab_size, (2, 32)); \
		img = torch.randn(2, 3, 64, 64); \
		aud = torch.randn(2, 50, config.audio_input_dim); \
		logits, loss = model(idx, idx, image=img, audio=aud); \
		print(f'logits形状: {tuple(logits.shape)}'); \
		print(f'loss: {loss.detach().item():.4f}'); \
		print('✓ 多模态前向测试成功')"

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
