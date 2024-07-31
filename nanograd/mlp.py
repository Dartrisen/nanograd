from nanograd.module import Module
from nanograd.layer import Layer


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
