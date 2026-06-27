from typing import Sequence

from nanograd.core.value import Value
from nanograd.module import Module
from nanograd.neuron import Neuron


class Layer(Module):
    """A layer of neurons that maps one vector to another."""

    def __init__(self, inputs: int, outputs: int) -> None:
        self.neurons: list[Neuron] = [Neuron(inputs) for _ in range(outputs)]

    def __call__(self, x: Sequence[float | Value]) -> list[Value]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        return [param for neuron in self.neurons for param in neuron.parameters()]
