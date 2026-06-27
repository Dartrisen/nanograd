from typing import Sequence

from nanograd.core.value import Value
from nanograd.layer import Layer
from nanograd.module import Module


class MLP(Module):
    """A multi-layer perceptron built from stacked layers."""

    def __init__(self, inputs: int, outputs: list[int]) -> None:
        if not outputs:
            raise ValueError("MLP requires at least one output dimension")

        sizes = [inputs, *outputs]
        self.layers: list[Layer] = [Layer(sizes[i], sizes[i + 1]) for i in range(len(outputs))]

    def __call__(self, x: Sequence[float | Value]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [param for layer in self.layers for param in layer.parameters()]
