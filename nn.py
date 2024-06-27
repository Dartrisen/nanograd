from value import Value
import random


class Neuron:
    def __init__(self, inputs: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = Value(random.uniform(-1, 1))
