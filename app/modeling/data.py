"""数据加载工具（wrapper，指向core实现）"""
# 从 core 模块导入所有实现
from app.core.data import (
    SimpleTokenizer,
    TextDataset,
    DataLoaderSimple as DataLoader,
    load_tokenizer,
    prepare_data,
    create_dataloader,
)

# 导出给外部使用
__all__ = ['SimpleTokenizer', 'TextDataset', 'DataLoader', 'load_tokenizer', 'prepare_data', 'create_dataloader']
