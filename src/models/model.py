"""GPT模型实现"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import ModelConfig


class LayerNorm(nn.Module):
    """带可选bias的LayerNorm"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """多头因果自注意力机制"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Q, K, V投影（合并为一个矩阵）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # 因果mask（下三角矩阵）
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # 计算Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # 注意力计算（scaled dot-product attention）
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hs)
        
        # 重新组合所有头的输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """前馈神经网络"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer块"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 残差连接
        x = x + self.mlp(self.ln_2(x))   # 残差连接
        return x


class VisualEncoder(nn.Module):
    """轻量视觉编码器：将图像转换为token序列"""
    def __init__(self, config):
        super().__init__()
        patch = getattr(config, 'vision_patch_size', 16)
        in_ch = getattr(config, 'vision_input_channels', 3)
        self.proj = nn.Conv2d(in_ch, config.n_embd, kernel_size=patch, stride=patch)
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, image):
        # image: (B, C, H, W)
        x = self.proj(image)  # (B, n_embd, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, T_img, n_embd)
        x = self.norm(x)
        return x


class AudioEncoder(nn.Module):
    """轻量语音编码器：将声学特征序列转换为token序列"""
    def __init__(self, config):
        super().__init__()
        in_dim = getattr(config, 'audio_input_dim', 80)
        kernel = getattr(config, 'audio_kernel_size', 3)
        stride = getattr(config, 'audio_stride', 2)
        padding = kernel // 2

        self.proj = nn.Conv1d(in_dim, config.n_embd, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, audio):
        # audio: (B, T_audio, F)
        x = audio.transpose(1, 2).contiguous()  # (B, F, T_audio)
        x = self.proj(x)  # (B, n_embd, T')
        x = x.transpose(1, 2).contiguous()  # (B, T', n_embd)
        x = self.norm(x)
        return x


class GPT(nn.Module):
    """GPT语言模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),  # position embedding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.multimodal_enabled = getattr(config, 'multimodal_enabled', False)
        self.modality_dropout = nn.Dropout(getattr(config, 'modality_dropout', 0.0))

        # 仅在启用多模态时创建参数，保证旧checkpoint兼容
        if self.multimodal_enabled:
            self.visual_encoder = VisualEncoder(config)
            self.audio_encoder = AudioEncoder(config)
        
        # 权重共享：embedding和输出层共享权重
        self.transformer.wte.weight = self.lm_head.weight
        
        # 初始化权重
        self.apply(self._init_weights)
        # 对残差投影使用特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _encode_modalities(self, image=None, audio=None):
        """编码视觉/语音输入并返回模态token序列"""
        if (image is not None or audio is not None) and not self.multimodal_enabled:
            raise ValueError("当前模型未启用多模态，请将 config.multimodal_enabled 设为 True")

        modality_tokens = []
        if self.multimodal_enabled and image is not None:
            modality_tokens.append(self.visual_encoder(image))
        if self.multimodal_enabled and audio is not None:
            modality_tokens.append(self.audio_encoder(audio))

        if not modality_tokens:
            return None

        fused = torch.cat(modality_tokens, dim=1)
        return self.modality_dropout(fused)

    def forward(self, idx, targets=None, image=None, audio=None):
        device = idx.device
        b, t = idx.size()

        # 文本token嵌入
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)

        # 编码模态token并与文本融合（前缀拼接）
        modal_emb = self._encode_modalities(image=image, audio=audio)
        if modal_emb is not None:
            prefix_len = modal_emb.size(1)
            if prefix_len >= self.config.block_size:
                # 至少保留1个文本token位置
                prefix_len = self.config.block_size - 1
                modal_emb = modal_emb[:, :prefix_len, :]

            max_text_len = self.config.block_size - prefix_len
            if tok_emb.size(1) > max_text_len:
                tok_emb = tok_emb[:, -max_text_len:, :]
                if targets is not None:
                    targets = targets[:, -max_text_len:]

            x_tokens = torch.cat([modal_emb, tok_emb], dim=1)
        else:
            x_tokens = tok_emb
            prefix_len = 0

            if x_tokens.size(1) > self.config.block_size:
                x_tokens = x_tokens[:, -self.config.block_size:, :]
                if targets is not None:
                    targets = targets[:, -self.config.block_size:]

        total_len = x_tokens.size(1)
        assert total_len <= self.config.block_size, f"序列长度{total_len}超过最大长度{self.config.block_size}"

        # 位置编码
        pos = torch.arange(0, total_len, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)  # (total_len, n_embd)

        # 前向传播
        x = self.transformer.drop(x_tokens + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # 训练模式：计算损失
            logits = self.lm_head(x)
            if prefix_len > 0:
                logits = logits[:, prefix_len:, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            # 推理模式：只计算最后一个token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss

    def get_num_params(self):
        """返回模型参数总数"""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0,
                 image=None, audio=None):
        """
        生成文本
        idx: (b, t) 当前上下文的token索引
        max_new_tokens: 要生成的新token数量
        """
        for _ in range(max_new_tokens):
            # 截断到block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向传播
            logits, _ = self(idx_cond, image=image, audio=audio)
            # 只取最后一个时间步
            logits = logits[:, -1, :] / temperature
            # 重复惩罚（降低已出现token再次被采样的概率）
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    prev_tokens = torch.unique(idx[b])
                    logits[b, prev_tokens] = logits[b, prev_tokens] / repetition_penalty

            # 可选：top-k采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 可选：top-p (nucleus) 采样
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float('Inf'))

            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 拼接到序列
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


if __name__ == "__main__":
    # 测试模型
    config = ModelConfig()
    model = GPT(config)
    print(f"模型参数量: {model.get_num_params()/1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(x, x)
    print(f"输出形状: {logits.shape}")
