from abc import ABC


class Module(ABC):
    def zero_grad(self):
        """Zeros the gradient  before it can be accumulated"""
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        """Gets the network parameters"""
        return []
