"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        # return out_grad * node / a * self.scalar  # a为0时会变成NaN
        return out_grad * power_scalar(a, self.scalar - 1) * self.scalar


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad / b
        grad_b = -out_grad * a / (b * b)
        return grad_a, grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # return array_api.true_divide(a, self.scalar, dtype=a.dtype)
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
          i, j = -1, -2
        elif len(self.axes) == 2:
          i, j = self.axes[0], self.axes[1]
        else:
          raise ValueError()
        l_axes = list(range(len(a.shape)))
        l_axes[i], l_axes[j] = l_axes[j], l_axes[i]
        return a.permute(l_axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a.compact(), self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # broadcast就相当于把一个变量输入到多个函数里，那梯度就应该是后面多个梯度相加
        # 这里统一认为广播出来的维度都在前面
        input_shape = node.inputs[0].shape
        less_axes = len(self.shape) - len(input_shape) # 多了的维度数
        reduce_axes = list(range(less_axes)) #要去掉的维度
        for i in range(len(input_shape)):
          if input_shape[i] != self.shape[i+less_axes]:
            reduce_axes.append(i+less_axes)
        if len(reduce_axes) == 0: # 有可能shape没有变化
          return out_grad
        return reshape(summation(out_grad, tuple(reduce_axes)), input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
          for axis in reversed(sorted(self.axes)): # 没有keepdims所以要从后往前，如果先sum了前面的维度，后面的维度就不对了
            a = a.sum(axis=axis)
          return a
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        # 和对于被求和的每一项的偏导都是1，所以summation的梯度是broadcast
        input_shape = node.inputs[0].shape
        new_shape = list(input_shape)
        axes = range(len(input_shape)) if self.axes is None else self.axes
        if isinstance(axes, int):
          axes = (axes,)
        for axis in axes:
          new_shape[axis] = 1
        return broadcast_to(reshape(out_grad, new_shape), input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        # 大于2维的numpy会把最后两维当成矩阵，如果一方2维一方大于2维，2维的就相当于被broadcast了，所以反过来就是summation
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        if (len(a.shape) < len(grad_a.shape)):
          grad_a = summation(grad_a, tuple(range(len(grad_a.shape) - len(a.shape))))
        if (len(b.shape) < len(grad_b.shape)):
          grad_b = summation(grad_b, tuple(range(len(grad_b.shape) - len(b.shape))))
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * node


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        out = node.realize_cached_data()
        # adjoint = (out > 0).astype(float) # 运算要保证类型不变，这里会变成float64
        # adjoint = (out > 0).astype(out.dtype)
        adjoint = out > 0
        return out_grad * Tensor(adjoint, device=out_grad.device)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return a.tanh()

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * (1 - tanh(a) ** 2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        n = len(args)
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, n)
        out = array_api.empty(new_shape, device=args[0].device)
        slices = [slice(None) for i in range(len(new_shape))]
        for i, arr in enumerate(args):
          slices[self.axis] = i
          out[tuple(slices)] = arr
        return out

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(None) for i in range(len(A.shape))]
        splits = []
        for i in range(n):
          slices[self.axis] = i
          splits.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(splits)

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = [a.shape[i]*(self.dilation+1) if i in self.axes else a.shape[i] for i in range(len(a.shape))]
        out = array_api.full(new_shape, 0, device=a.device)
        slices = [slice(None) for i in range(len(new_shape))]
        for axis in self.axes:
          slices[axis] = slice(0, new_shape[axis], self.dilation+1)
        out[tuple(slices)] = a
        return out

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        slices = [slice(None) for i in range(len(a.shape))]
        for axis in self.axes:
          slices[axis] = slice(0, a.shape[axis], self.dilation+1)
        return a[tuple(slices)]

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        A = A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K * K * C_in
        out_H, out_W = (H-K+1)//self.stride, (W-K+1)//self.stride
        # im2col
        A = A.as_strided((N, out_H, out_W, K, K, C_in), (Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs))
        A = A.compact().reshape((N*out_H*out_W, inner_dim))
        B = B.compact().reshape((inner_dim, C_out))
        out = A @ B
        return out.reshape((N,out_H,out_W,C_out))

    def gradient(self, out_grad, node):
        X, W = node.inputs
        K,_,_,_ = W.shape

        # X(N,H,W,Cin) W(K,K,Cin,Cout) -> conv(X,W,s,p)(N, (H+2p-K+1)/s, (W+2p-K+1)/s, Cout)

        if self.stride > 1:
          # (N, (H+2p-K+1)/s, (W+2p-K+1)/s, Cout) -> (N, (H+2p-K+1), (W+2p-K+1), Cout)
          out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        
        # W(K,K,Cin,Cout) -> W2(K,K,Cout,Cin)
        W2 = transpose(flip(W, (0, 1)))
        # out_grad(N,(H+2p-K+1),(W+2p-K+1),Cout) W2(K,K,Cout,Cin) -> conv(out_grad,W2,1,q)(N,H,W,Cin)
        # 2p-K+1+2q-K+1=0 q=k-1-p
        X_grad = conv(out_grad, W2, padding=K-1-self.padding)

        # X(N,H,W,Cin) -> X2(Cin,H,W,N)
        X2 = transpose(X, (0, 3))
        # out_grad(N,(H+2p-K+1),(W+2p-K+1),Cout) -> out_grad2((H+2p-K+1),(W+2p-K+1),N,Cout)
        out_grad2 = transpose(transpose(out_grad, (0, 1)), (1, 2))
        # X2(Cin,H,W,N) out_grad2((H+2p-K+1),(W+2p-K+1),N,Cout) -> conv(X2,out_grad2,1,q)(Cin,K,K,Cout)
        # H+2q-(H+2p-K+1)+1=K q=p
        W_grad = conv(X2, out_grad2, padding=self.padding)
        # (Cin,K,K,Cout) -> (K,K,Cin,Cout)
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))
        
        return X_grad, W_grad


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


