from __future__ import annotations

from value import Value
import random


class Neuron:
    def __init__(self, inputs: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: list[int | float]):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out


class Layer:
    def __init__(self, inputs: int, outputs: int) -> None:
        self.neurons = [Neuron(inputs) for _ in range(outputs)]

    def __call__(self, x: list[int | float]):
        outs = [n(x) for n in self.neurons]
        return outs


if __name__ == '__main__':
    x = [2.0, 3.0]
    l = Layer(2, 3)
    print(l(x))
