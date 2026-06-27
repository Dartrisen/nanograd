import numpy as np
from nanograd.core.value import Value
from nanograd.core.tensor import Tensor


class Module:
    """Base class for all neural network modules (Scalar or Tensor based)."""

    def zero_grad(self) -> None:
        """Resets gradients for all registered parameters."""
        for p in self.parameters():
            if isinstance(p, Tensor):
                p.grad = np.zeros_like(p.data)
            else:
                p.grad = 0.0  # For classic scalar Value objects

    def parameters(self) -> list[Value | Tensor]:
        """Override this in subclasses to return parameters."""
        return []
