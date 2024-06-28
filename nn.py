from __future__ import annotations

import random
from abc import ABC

from value import Value


class Module(ABC):
    def zero_grad(self):
        """Zeros the gradient  before it can be accumulated"""
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        """Gets the network parameters"""
        return []


class Neuron(Module):
    def __init__(self, inputs: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: list[float]):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        self.neurons = [Neuron(inputs) for _ in range(outputs)]

    def __call__(self, x: list[float]):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, inputs: int, outputs: list[float]):
        sz = [inputs] + outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(outputs))]

    def __call__(self, x: list[float]):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self. layers for p in layer.parameters()]


if __name__ == '__main__':
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1])

    num_epochs = 20
    lr = 0.05
    for k in range(num_epochs):
        # forward
        y_pred = [model(x) for x in xs]
        loss = sum((y_out[0] - y_gt)**2 for y_gt, y_out in zip(ys, y_pred))

        # backward
        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data += - lr * p.grad

        print(k, loss.data)
