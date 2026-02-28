import numpy as np

from app.core.tensor import Tensor


class Module:
    def __init__(self):
        self.training = True

    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        """切换到训练模式"""
        self.training = True
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train()

    def eval(self):
        """切换到评估模式"""
        self.training = False
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.eval()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.eval()


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.02)

    def __call__(self, input_ids):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        out_data = self.weight.data[input_ids]
        out = Tensor(out_data, requires_grad=self.weight.requires_grad, _children=(self.weight,), _op="embedding")

        def _backward():
            if self.weight.requires_grad:
                grad = np.zeros_like(self.weight.data)
                np.add.at(grad, input_ids, out.grad)
                self.weight.grad += grad

        out._backward = _backward
        return out


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        scale = (2.0 / max(1, in_features)) ** 0.5
        self.weight = Parameter(np.random.randn(in_features, out_features) * scale)
        self.bias = Parameter(np.zeros((1, 1, out_features))) if bias else None

    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    """层归一化：对最后一个维度进行归一化"""
    def __init__(self, normalized_shape, eps=1e-5, bias=True):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape)) if bias else None

    def __call__(self, x):
        # x.shape = (..., normalized_shape)
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        x_normalized = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = x_normalized * self.weight.data + (self.bias.data if self.bias else 0)
        
        requires_grad = x.requires_grad or self.weight.requires_grad or (self.bias and self.bias.requires_grad)
        children = [c for c in [x, self.weight, self.bias] if c is not None and getattr(c, 'requires_grad', False)]
        out = Tensor(out_data, requires_grad=requires_grad, _children=tuple(children), _op="layernorm")

        def _backward():
            if not out.grad.any():
                return
            # 简化的梯度计算
            if self.weight.requires_grad:
                self.weight.grad += (out.grad * x_normalized).sum(axis=tuple(range(out.grad.ndim - len(self.normalized_shape))))
            if self.bias and self.bias.requires_grad:
                self.bias.grad += out.grad.sum(axis=tuple(range(out.grad.ndim - len(self.normalized_shape))))
            if x.requires_grad:
                # 简化版本：主要传播 weight 的梯度
                dxhat = out.grad * self.weight.data
                std = np.sqrt(var + self.eps)
                x.grad += dxhat / std

        out._backward = _backward
        return out


class Dropout(Module):
    """Dropout正则化"""
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        # 训练时随机置零
        mask = np.random.binomial(1, 1 - self.p, size=x.data.shape) / (1 - self.p)
        out_data = x.data * mask
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="dropout")

        def _backward():
            if x.requires_grad:
                x.grad += out.grad * mask

        out._backward = _backward
        return out

    def eval(self):
        """切换到评估模式"""
        self.training = False

    def train(self):
        """切换到训练模式"""
        self.training = True


class GELU(Module):
    """GELU激活函数（近似版本）"""
    def __call__(self, x):
        # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        # 简化版本: GELU(x) ≈ x * sigmoid(1.702 * x)
        out_data = x.data * (1.0 / (1.0 + np.exp(-1.702 * x.data)))
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="gelu")

        def _backward():
            if x.requires_grad:
                # 近似梯度
                sigmoid = 1.0 / (1.0 + np.exp(-1.702 * x.data))
                grad_sigmoid = 1.702 * sigmoid * (1 - sigmoid)
                x.grad += out.grad * (sigmoid + x.data * grad_sigmoid)

        out._backward = _backward
        return out


class ModuleList(Module):
    """存储Module列表"""
    def __init__(self, modules=None):
        self._modules = []
        if modules:
            self._modules.extend(modules)

    def append(self, module):
        self._modules.append(module)

    def __iter__(self):
        return iter(self._modules)

    def __getitem__(self, idx):
        return self._modules[idx]

    def __len__(self):
        return len(self._modules)

    def parameters(self):
        params = []
        for module in self._modules:
            if isinstance(module, Module):
                params.extend(module.parameters())
            elif isinstance(module, Parameter):
                params.append(module)
        return params


class ModuleDict(Module):
    """存储Module字典"""
    def __init__(self, modules=None):
        self._modules = {}
        if modules:
            self._modules.update(modules)

    def __setitem__(self, key, module):
        self._modules[key] = module

    def __getitem__(self, key):
        return self._modules[key]

    def __contains__(self, key):
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def parameters(self):
        params = []
        for module in self._modules.values():
            if isinstance(module, Module):
                params.extend(module.parameters())
            elif isinstance(module, Parameter):
                params.append(module)
        return params
