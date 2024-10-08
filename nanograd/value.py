from __future__ import annotations

from math import exp

from nanograd.plot_graph import draw_dot
from nanograd.topo_sort import topo_sort_iterative


class Value:
    """Value class which stores a single scalar and it's gradient.

    """

    def __init__(self, data: int | float, _children: tuple = (), _op: str = "", label: str = "") -> None:
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: int | float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other: Value) -> Value:
        return self.__add__(other)

    def __mul__(self, other: int | float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other: int | float) -> Value:
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other: int | float | Value) -> Value:
        return self * other

    def __truediv__(self, other: int | float | Value) -> Value:
        return self * other**(-1)

    def __neg__(self) -> Value:
        return self * (-1)

    def __sub__(self, other: int | float | Value) -> Value:
        return self + (-other)

    def tanh(self) -> Value:
        x = self.data
        t = (exp(2*x) - 1) / (exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> Value:
        x = self.data
        out = Value(exp(x), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        topo = topo_sort_iterative(self)

        # Backpropagation step
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


if __name__ == '__main__':
    # inputs x1, x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights w1, w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bias of the neuron
    b = Value(6.8813735870195432, label="b")
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    e = (2*n).exp()
    o = (e - 1) / (e + 1)
    # o = n.tanh()
    o.backward()
    # render graph
    dot = draw_dot(o)
    dot.render("graph", format="jpg", view=True)
