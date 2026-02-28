from app.core.nn import Module, Embedding, Linear, LayerNorm, TransformerBlock
from app.core.losses import cross_entropy


class TinyLM(Module):
    """最小可用语言模型：Embedding + Linear(vocab projection)"""

    def __init__(self, vocab_size, n_embd):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.tok_emb = Embedding(vocab_size, n_embd)
        self.lm_head = Linear(n_embd, vocab_size, bias=True)

    def __call__(self, input_ids, targets=None):
        x = self.tok_emb(input_ids)          # (B, T, C)
        logits = self.lm_head(x)             # (B, T, V)
        loss = None
        if targets is not None:
            loss = cross_entropy(logits, targets)
        return logits, loss


class TransformerLM(Module):
    """基于 Transformer 的语言模型（支持自注意力、位置编码、多层堆叠）"""

    def __init__(self, vocab_size, n_embd, n_layers=2, n_heads=4, max_seq_len=2048, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding + Position embedding
        self.tok_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Embedding(max_seq_len, n_embd)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(n_embd, n_heads, dropout)
            for _ in range(n_layers)
        ]
        
        # 最后的 LayerNorm 和输出投影
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = Linear(n_embd, vocab_size, bias=False)

    def __call__(self, input_ids, targets=None):
        B, T = input_ids.shape if hasattr(input_ids, 'shape') else (len(input_ids), len(input_ids[0]))
        
        # 位置索引
        import numpy as np
        pos = np.arange(0, T, dtype=np.int64)[None, :]  # (1, T)
        
        # Token + Position embeddings
        tok_emb = self.tok_emb(input_ids)  # (B, T, C)
        pos_emb = self.pos_emb(pos)        # (1, T, C)
        x = tok_emb + pos_emb              # (B, T, C)
        
        # 通过 Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 最终归一化和输出
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        
        loss = None
        if targets is not None:
            loss = cross_entropy(logits, targets)
        
        return logits, loss
