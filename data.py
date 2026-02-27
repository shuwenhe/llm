"""数据加载和处理"""
import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import numpy as np


def clean_wikitext(text):
    """
    清洗WikiText数据中的格式标记
    
    移除：
    - 维基百科分词标记 (@-@, @,@, @.@)
    - 标题分隔符 (= = =)
    - 多余的空白
    """
    if not isinstance(text, str):
        return ""
    
    # 替换维基标记
    text = text.replace(' @-@ ', '-')
    text = text.replace(' @,@ ', ',')
    text = text.replace(' @.@ ', '.')
    text = text.replace('@-@', '-')
    text = text.replace('@,@', ',')
    text = text.replace('@.@', '.')
    
    # 移除标题分隔符（= = = Title = = =）
    text = re.sub(r'\s*=+\s+([^=]+)\s+=+\s*', r'\n\1\n', text)
    text = re.sub(r'\s*=+\s*', '\n', text)
    
    # 清理多余换行
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # 清理多余空格
    text = re.sub(r' +', ' ', text)
    
    # 移除首尾空白
    text = text.strip()
    
    return text


class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
    
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        # 获取block_size长度的序列
        x = torch.from_numpy(self.tokens[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.tokens[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y


def load_tokenizer():
    """加载分词器"""
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return tokenizer


def prepare_data(dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", block_size=512, clean_data=True):
    """
    准备训练数据
    
    参数:
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        block_size: 序列块大小
        clean_data: 是否清洗数据（移除格式标记）
    
    返回: train_dataset, val_dataset, tokenizer
    """
    print(f"加载数据集: {dataset_name}/{dataset_config}")
    
    # 加载数据
    dataset = load_dataset(dataset_name, dataset_config)
    
    # 加载分词器
    tokenizer = load_tokenizer()
    
    # 清洗数据
    if clean_data and dataset_name == "wikitext":
        print("清洗WikiText格式标记...")
        
        def clean_function(examples):
            cleaned_texts = [clean_wikitext(text) for text in examples['text']]
            return {'text': cleaned_texts}
        
        dataset = dataset.map(
            clean_function,
            batched=True,
            desc="Cleaning data"
        )
    
    # 分词函数
    def tokenize_function(examples):
        return tokenizer(examples['text'])
    
    # 对数据集进行分词
    print("对数据集进行分词...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    # 将所有tokens拼接成一个大数组
    def concatenate_tokens(dataset_split):
        all_tokens = []
        for example in dataset_split:
            all_tokens.extend(example['input_ids'])
        return np.array(all_tokens, dtype=np.uint16)
    
    print("拼接tokens...")
    train_tokens = concatenate_tokens(tokenized_datasets['train'])
    val_tokens = concatenate_tokens(tokenized_datasets['validation'])
    
    print(f"训练集tokens数量: {len(train_tokens):,}")
    print(f"验证集tokens数量: {len(val_tokens):,}")
    
    # 创建数据集
    train_dataset = TextDataset(train_tokens, block_size)
    val_dataset = TextDataset(val_tokens, block_size)
    
    return train_dataset, val_dataset, tokenizer


def create_dataloader(dataset, batch_size, shuffle=True):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


if __name__ == "__main__":
    # 测试数据加载
    train_dataset, val_dataset, tokenizer = prepare_data()
    train_loader = create_dataloader(train_dataset, batch_size=8)
    
    # 打印一个batch
    x, y = next(iter(train_loader))
    print(f"\nBatch形状:")
    print(f"输入 x: {x.shape}")
    print(f"目标 y: {y.shape}")
    print(f"\n解码第一个样本:")
    print(tokenizer.decode(x[0].tolist()))
