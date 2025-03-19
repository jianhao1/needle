from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        max_z = array_api.max(Z, 1, keepdims=True)
        exp_z = array_api.exp(Z-max_z)
        sum_exp_z = array_api.sum(exp_z, 1, keepdims=True)
        log_sum_exp_z = array_api.log(sum_exp_z) + max_z
        return Z - log_sum_exp_z

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(1, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, 1)
        # 先reshape回与输入相同的维度数量，再broadcast成输入维度
        return out_grad - (summation(out_grad, 1) / sum_exp_z).reshape((z.shape[0], 1)).broadcast_to(z.shape) * exp_z


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_z = Z.max(self.axes, keepdims=True).broadcast_to(Z.shape)
        max_z2 = Z.max(self.axes)
        self.exp_z = array_api.exp(Z-max_z)
        self.sum_exp_z = array_api.sum(self.exp_z, self.axes)
        return array_api.log(self.sum_exp_z) + max_z2

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        # 先reshape回与输入相同的维度数量，再broadcast成输入维度
        axes = range(len(z.shape)) if self.axes is None else self.axes
        new_shape = list(z.shape)
        for i in axes:
          new_shape[i] = 1
        exp_z = Tensor(self.exp_z, device=out_grad.device)
        sum_exp_z = Tensor(self.sum_exp_z, device=out_grad.device)
        return (out_grad / sum_exp_z).reshape(new_shape).broadcast_to(z.shape) * exp_z


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

