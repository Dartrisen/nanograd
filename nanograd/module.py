from abc import ABC


class Module(ABC):
    def zero_grad(self) -> None:
        """Zeros the gradient  before it can be accumulated"""
        for param in self.parameters():
            param.grad = 0

    def parameters(self) -> list:
        """Gets the network parameters"""
        return []
