"""数据加载工具"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SimpleTokenizer:
    """简单的字符级 tokenizer（当 transformers 不可用时使用）"""
    
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        # 构建字符到 ID 的映射
        self.stoi = {}
        self.itos = {}
        
        # 预定义常见的 ASCII 字符
        chars = list(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:\'"()-'))
        chars.sort()
        
        for i, c in enumerate(chars):
            self.stoi[c] = i
            self.itos[i] = c
        
        self.next_id = len(chars)
    
    def encode(self, text, return_tensors=None):
        """编码文本为 token IDs"""
        tokens = []
        for char in text:
            if char not in self.stoi:
                if self.next_id < self.vocab_size:
                    self.stoi[char] = self.next_id
                    self.itos[self.next_id] = char
                    self.next_id += 1
                else:
                    # 使用未知 token ID
                    self.stoi[char] = 0
            tokens.append(self.stoi.get(char, 0))
        
        if return_tensors == 'pt':
            return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        return tokens
    
    def decode(self, token_ids):
        """解码 token IDs 为文本"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy().tolist()
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        text = ''
        for token_id in token_ids:
            if isinstance(token_id, int):
                text += self.itos.get(token_id, '?')
            else:
                # 处理嵌套列表
                text += self.decode(token_id)
        return text


def load_tokenizer():
    """加载或创建 tokenizer"""
    try:
        # 尝试使用 transformers 库中的 GPT-2 tokenizer
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return tokenizer
    except (ImportError, Exception):
        # 如果 transformers 不可用，使用简单的字符级 tokenizer
        print("⚠️  Warning: transformers 库不可用，使用简单字符级 tokenizer")
        return SimpleTokenizer(vocab_size=50257)


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, tokens, block_size):
        """
        Args:
            tokens: numpy array 或列表，包含 token IDs
            block_size: 每个样本的序列长度
        """
        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=np.uint32)
        elif not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens, dtype=np.uint32)
        
        self.tokens = tokens
        self.block_size = block_size
    
    def __len__(self):
        """返回数据集大小"""
        # 每个起始位置对应一个样本，最后一个样本需要 block_size+1 个 tokens
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        """获取样本"""
        # 获取输入和目标
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()


def prepare_data(data_file, tokenizer, block_size=512, train_split=0.8):
    """准备训练数据"""
    # 读取文本文件
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize
    print(f"Tokenizing {data_file}...")
    if hasattr(tokenizer, 'encode_plus'):
        # transformers tokenizer
        tokens = tokenizer.encode(text, return_tensors=None)
        if isinstance(tokens, dict):
            tokens = tokens['input_ids']
    else:
        # 简单 tokenizer
        tokens = tokenizer.encode(text)
    
    tokens = np.array(tokens, dtype=np.uint32)
    print(f"Total tokens: {len(tokens):,}")
    
    # 分割训练集和验证集
    train_size = int(len(tokens) * train_split)
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]
    
    return train_tokens, val_tokens


def create_dataloader(tokens, batch_size, block_size, shuffle=False):
    """创建数据加载器"""
    dataset = TextDataset(tokens, block_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    return loader
