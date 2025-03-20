"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = None
        if bias:
          self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose())

    def forward(self, X: Tensor) -> Tensor:
        out = X @ self.weight
        if self.bias:
          out += self.bias.broadcast_to(out.shape)
        return out


class Flatten(Module):
    def forward(self, X):
        new_shape = (X.shape[0], np.prod(X.shape[1:]))
        return X.reshape(new_shape)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
          x = module.forward(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        one_hot_y = init.one_hot(logits.shape[1], y, device=logits.device)
        return ops.summation(ops.logsumexp(logits, (1,)) - ops.summation(one_hot_y * logits, (1,))) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
          m, n = x.shape
          mean = x.sum((0,)) / m
          # self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
          # 别把输入tensor引用了，回收不了！
          self.running_mean = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
          mean = mean.reshape((1, n)).broadcast_to(x.shape)
          var = ((x - mean) ** 2).sum((0,)) / m
          # self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
          self.running_var = (1 - self.momentum) * self.running_var.data + self.momentum * var.data
          var = var.reshape((1, n)).broadcast_to(x.shape)
          w = self.weight.broadcast_to(x.shape)
          b = self.bias.broadcast_to(x.shape)
          return w * (x - mean) / ((var + self.eps) ** 0.5) + b
        else:
          w = self.weight.broadcast_to(x.shape)
          b = self.bias.broadcast_to(x.shape)
          mean = self.running_mean.reshape((1,self.dim)).broadcast_to(x.shape)
          var = self.running_var.reshape((1,self.dim)).broadcast_to(x.shape)
          return w * (x - mean) / ((var + self.eps) ** 0.5) + b

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        m, n = x.shape
        mean = x.sum((1,)) / n
        mean = mean.reshape((m, 1)).broadcast_to(x.shape)
        var = ((x - mean) ** 2).sum((1,)) / n
        var = var.reshape((m, 1)).broadcast_to(x.shape)
        w = self.weight.broadcast_to(x.shape)
        b = self.bias.broadcast_to(x.shape)
        return w * (x - mean) / ((var + self.eps) ** 0.5) + b


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
          mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
          return (x * mask) / (1 - self.p)
        else:
          return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
