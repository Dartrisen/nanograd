from nanograd.module import Module
from nanograd.neuron import Neuron


class Layer(Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        self.neurons = [Neuron(inputs) for _ in range(outputs)]

    def __call__(self, x: list[float]):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
