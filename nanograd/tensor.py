from __future__ import annotations

import numpy as np
from typing import Any
from nanograd.topo_sort import topo_sort_iterative


def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sums out broadcasted dimensions to match the target_shape."""
    if grad.shape == target_shape:
        return grad

    # First, sum out leading dimensions if target_shape has fewer dimensions
    grad_ndim = grad.ndim
    target_ndim = len(target_shape)
    for _ in range(grad_ndim - target_ndim):
        grad = grad.sum(axis=0)

    # Second, sum along dimensions that were broadcasted from 1
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

class Tensor:
    def __init__(self, data: Any, _children: tuple = (), _op: str = '', label: str = ''):
        # Ensure data is a float32 NumPy array
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def backward(self):
        """Topological sort and backward pass execution loop."""
        topo = topo_sort_iterative(self)

        # Seed the global gradient (dOut/dOut = 1)
        self.grad = np.ones_like(self.data)

        # Go through the computational graph in reverse topological order
        for v in reversed(topo):
            v._backward()

    def add(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def mul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def matmul(self, other):
        """Matrix Multiplication (The primary driver for Tensor Train operations)."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), 'matmul')

        def _backward():
            # Gradients derived using transposed matrices:
            # dL/dX = dL/dY @ W.T  and  dL/dW = X.T @ dL/dY
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def reshape(self, *shape):
        """Changes the view of the tensor without altering its data."""
        # Handle both unpack formatting: reshape(1, 2) or reshape((1, 2))
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]

        out = Tensor(self.data.reshape(shape), (self,), 'reshape')

        def _backward():
            # Bring the gradient back to the original array shape layout
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        """Permutes/swaps the axes of the tensor."""
        out = Tensor(self.data.transpose(axes), (self,), 'transpose')

        def _backward():
            if axes is None:
                # Standard matrix transpose inverse is just another transpose
                self.grad += out.grad.transpose()
            else:
                # For higher-order tensors, we invert the custom axis permutation
                inv_axes = np.argsort(axes)
                self.grad += out.grad.transpose(inv_axes)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def __add__(self, other): return self.add(other)
    def __radd__(self, other): return self.add(other)

    def __mul__(self, other): return self.mul(other)
    def __rmul__(self, other): return self.mul(other)

    def __matmul__(self, other): return self.matmul(other)
    def __rmatmul__(self, other): return Tensor(other).matmul(self)

    @property
    def T(self): return self.transpose()

    def __repr__(self):
        return f"Tensor(data={self.data.tolist()}, shape={self.data.shape}, grad={self.grad.tolist() if self.grad is not None else None})"
