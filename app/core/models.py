from app.core.nn import Module, Embedding, Linear
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
