import random
from typing import Sequence

from nanograd.core.value import Value
from nanograd.module import Module


class Neuron(Module):
    def __init__(self, inputs: int) -> None:
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(inputs)]
        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x: Sequence[float | Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]
