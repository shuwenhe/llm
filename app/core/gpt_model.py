"""GPT模型的core实现（纯numpy后端）"""
import math
import numpy as np
from app.core.tensor import Tensor
from app.core.nn import Module, Parameter, Embedding, Linear, LayerNorm, Dropout, GELU, ModuleList, ModuleDict


class CausalSelfAttention(Module):
    """多头因果自注意力机制（简化版）"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Q, K, V投影（合并为一个矩阵）
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 正则化
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # 因果mask（下三角矩阵）- 作为numpy数组存储
        self.causal_mask = np.tril(np.ones((config.block_size, config.block_size)))

    def __call__(self, x):
        B, T, C = x.data.shape  # batch, sequence length, embedding dim
        
        # 计算Q, K, V
        qkv = self.c_attn(x)
        # 将qkv分割为q, k, v
        qkv_data = qkv.data.reshape(B, T, 3, C)
        q_data = qkv_data[:, :, 0, :]  # (B, T, C)
        k_data = qkv_data[:, :, 1, :]
        v_data = qkv_data[:, :, 2, :]
        
        # 重塑为多头形式
        head_dim = C // self.n_head
        q_data = q_data.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k_data = k_data.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v_data = v_data.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        
        # 注意力计算（scaled dot-product attention）
        scale = 1.0 / math.sqrt(head_dim)
        att = np.matmul(q_data, k_data.transpose(0, 1, 3, 2)) * scale  # (B, nh, T, T)
        
        # 应用因果mask
        mask = self.causal_mask[:T, :T]
        att = np.where(mask[None, None, :, :] == 0, -1e10, att)
        
        # Softmax
        att_max = att.max(axis=-1, keepdims=True)
        att_exp = np.exp(att - att_max)
        att_probs = att_exp / (att_exp.sum(axis=-1, keepdims=True) + 1e-12)
        
        # Dropout (简化：直接在numpy上操作)
        if self.training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, size=att_probs.shape) / (1 - self.dropout)
            att_probs = att_probs * dropout_mask
        
        # 应用注意力权重
        y_data = np.matmul(att_probs, v_data)  # (B, nh, T, hs)
        
        # 重新组合所有头的输出
        y_data = y_data.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 创建Tensor并设置梯度传播（简化版）
        y = Tensor(y_data, requires_grad=x.requires_grad, _children=(x,), _op="attention")
        
        # 输出投影
        out = self.resid_dropout(self.c_proj(y))
        return out


class MLP(Module):
    """前馈神经网络"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = GELU()
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(Module):
    """Transformer块"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))  # 残差连接
        x = x + self.mlp(self.ln_2(x))   # 残差连接
        return x


class GPT(Module):
    """GPT语言模型（核心版本，不含多模态）"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token和位置嵌入
        self.wte = Embedding(config.vocab_size, config.n_embd)  # token embedding
        self.wpe = Embedding(config.block_size, config.n_embd)  # position embedding
        self.drop = Dropout(config.dropout)
        
        # Transformer块
        self.h = ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # 最终层归一化
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # 输出层（语言模型头）
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重共享：embedding和输出层共享权重
        # 注意：这里简化处理，不做严格的权重共享
        
        # 初始化权重（简化版）
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 对所有Linear层的权重进行特殊初始化
        for module in self.parameters():
            if isinstance(module, Parameter):
                # 简单的正态分布初始化
                pass  # 已在各层构造时初始化

    def __call__(self, idx, targets=None):
        """
        前向传播
        idx: (B, T) token索引
        targets: (B, T) 目标token (可选，用于训练)
        """
        if isinstance(idx, np.ndarray):
            idx_array = idx
        else:
            idx_array = idx.data if isinstance(idx, Tensor) else np.array(idx)
        
        B, T = idx_array.shape
        assert T <= self.config.block_size, f"序列长度{T}超过最大长度{self.config.block_size}"
        
        # Token嵌入
        tok_emb = self.wte(idx_array)  # (B, T, n_embd)
        
        # 位置编码
        pos = np.arange(0, T, dtype=np.int64)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        
        # 相加并应用dropout
        x = self.drop(tok_emb + pos_emb)
        
        # 通过所有Transformer块
        for block in self.h:
            x = block(x)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        if targets is not None:
            # 训练模式：计算所有位置的logits和损失
            logits = self.lm_head(x)
            # 计算交叉熵损失
            from app.core.losses import cross_entropy_loss
            loss = cross_entropy_loss(logits, Tensor(targets))
        else:
            # 推理模式：只计算最后一个token的logits
            # 取最后一个时间步
            x_last = Tensor(x.data[:, -1:, :], requires_grad=x.requires_grad, _children=(x,), _op="slice")
            logits = self.lm_head(x_last)
            loss = None
        
        return logits, loss

    def get_num_params(self):
        """返回模型参数总数"""
        total = 0
        for p in self.parameters():
            total += p.data.size
        return total

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        生成文本
        idx: (B, T) 当前上下文的token索引
        max_new_tokens: 要生成的新token数量
        temperature: 温度参数
        top_k: top-k采样（可选）
        """
        self.eval()  # 切换到评估模式
        
        if isinstance(idx, list):
            idx = np.array(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]  # 添加batch维度
        
        for _ in range(max_new_tokens):
            # 截断到block_size
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 前向传播
            logits, _ = self(idx_cond, targets=None)
            
            # 只取最后一个时间步，应用温度
            logits_data = logits.data[:, -1, :] / max(temperature, 1e-6)
            
            # Top-k采样（可选）
            if top_k is not None:
                # 保留top-k个最大值，其余设为-inf
                top_k_actual = min(top_k, logits_data.shape[-1])
                indices_to_remove = logits_data < np.partition(logits_data, -top_k_actual, axis=-1)[:, [-top_k_actual]]
                logits_data = np.where(indices_to_remove, -1e10, logits_data)
            
            # Softmax获取概率
            logits_max = logits_data.max(axis=-1, keepdims=True)
            exp_logits = np.exp(logits_data - logits_max)
            probs = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-12)
            
            # 采样
            idx_next = np.array([[np.random.choice(probs.shape[-1], p=probs[i])] for i in range(probs.shape[0])])
            
            # 拼接到序列
            idx = np.concatenate([idx, idx_next], axis=1)
        
        return idx


if __name__ == "__main__":
    # 测试模型
    from app.modeling.config import ModelConfig
    
    config = ModelConfig(
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=64,
        dropout=0.1,
        bias=True
    )
    
    model = GPT(config)
    print(f"模型参数量: {model.get_num_params()/1e6:.2f}M")
    
    # 测试前向传播
    x = np.random.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(x, x)
    print(f"输出形状: {logits.data.shape}")
    print(f"损失: {loss.data if loss else None}")
    
    # 测试生成
    start_ids = np.array([[1, 2, 3]], dtype=np.int64)
    generated = model.generate(start_ids, max_new_tokens=10, temperature=0.8)
    print(f"生成序列: {generated}")
