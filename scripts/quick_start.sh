#!/bin/bash
# OpenAI风格训练快速启动脚本
# 使用方法: bash quick_start.sh quick|standard|extended|precision [data_file]

set -e

PROJECT_DIR="/home/shuwen/llm"
cd "$PROJECT_DIR"

# 获取参数
PRESET="${1:-standard}"
DATA_FILE="${2:-data/zh_sample.txt}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🚀 OpenAI风格工业级训练启动${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 验证参数
if [[ ! "$PRESET" =~ ^(quick|standard|extended|precision)$ ]]; then
    echo -e "${RED}❌ 错误：预设必须是 quick|standard|extended|precision${NC}"
    exit 1
fi

if [[ ! -f "$DATA_FILE" ]]; then
    echo -e "${RED}❌ 错误：数据文件不存在 $DATA_FILE${NC}"
    exit 1
fi

# 显示配置信息
echo -e "${GREEN}📋 训练配置:${NC}"
echo -e "  预设: ${YELLOW}$PRESET${NC}"
echo -e "  数据: ${YELLOW}$DATA_FILE${NC}"
echo -e "  数据大小: $(du -h "$DATA_FILE" | cut -f1)"
echo ""

# 显示预设详情
echo -e "${GREEN}⚙️  预设参数:${NC}"
python train_cli.py --preset "$PRESET" | grep -A 10 "^PRESET\|批次\|训练\|学习率" || true
echo ""

# 干运行
echo -e "${GREEN}🧪 干运行模式 (验证命令):${NC}"
python train_cli.py --preset "$PRESET" --data-file "$DATA_FILE" --dry-run
echo ""

# 询问确认
read -p "$(echo -e ${YELLOW}是否继续执行实际训练? [y/N] ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⏭️  已取消${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🎓 开始训练${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 执行训练
START_TIME=$(date +%s)
python train_cli.py --preset "$PRESET" --data-file "$DATA_FILE"
TRAIN_EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 训练成功完成！${NC}"
    echo -e "${GREEN}⏱️  总耗时: $((DURATION / 60))分钟 $((DURATION % 60))秒${NC}"
    echo ""
    
    echo -e "${GREEN}📊 检查结果:${NC}"
    echo ""
    
    # 显示检查点
    echo -e "${YELLOW}检查点列表:${NC}"
    python train_manager.py list | head -20
    echo ""
    
    # 显示训练历史
    echo -e "${YELLOW}训练历史:${NC}"
    python train_manager.py history | head -30
    echo ""
    
    # 显示日志位置
    LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        echo -e "${YELLOW}📝 训练日志:${NC}"
        echo "  $LATEST_LOG"
        echo ""
    fi
    
    echo -e "${GREEN}🎉 下一步:${NC}"
    echo "  1. 查看完整训练历史: python train_manager.py history"
    echo "  2. 对比不同模型: python train_manager.py compare model1.pt model2.pt"
    echo "  3. 清理旧检查点: python train_manager.py clean"
    echo "  4. 部署最优模型: LLM_CHECKPOINT=checkpoints/best_model.pt make serve"
else
    echo -e "${RED}❌ 训练失败！${NC}"
    echo -e "${RED}⏱️  总耗时: $((DURATION / 60))分钟 $((DURATION % 60))秒${NC}"
    
    LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        echo ""
        echo -e "${RED}📝 查看错误日志:${NC}"
        echo "  tail -50 $LATEST_LOG"
    fi
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

exit $TRAIN_EXIT_CODE
