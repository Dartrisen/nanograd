from typing import List, Union
import numpy as np
from nanograd.value import Value
from nanograd.tensor import Tensor

Parameter = Union[Value, Tensor]

class Module:
    """Base class for all neural network modules (Scalar or Tensor based)."""

    def zero_grad(self) -> None:
        """Resets gradients for all registered parameters."""
        for p in self.parameters():
            if isinstance(p, Tensor):
                p.grad = np.zeros_like(p.data)
            else:
                p.grad = 0.0  # For classic scalar Value objects

    def parameters(self) -> List[Parameter]:
        """Override this in subclasses to return parameters."""
        return []
