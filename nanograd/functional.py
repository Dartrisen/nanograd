"""
Functional tracking operations and loss objectives for the NanoGrad autograd engine.
"""

from __future__ import annotations
import numpy as np
from nanograd.core.tensor import Tensor

def cross_entropy_loss(logits_tensor: Tensor, target_idx: int) -> Tensor:
    """
    Computes numerical categorical cross-entropy.
    Hooks a backward operation directly into the NanoGrad autograd trace.
    """
    raw_logits = logits_tensor.data.flatten()
    # Normalize to prevent exponent overflow errors
    stabilized_logits = raw_logits - np.max(raw_logits)
    exps = np.exp(stabilized_logits)
    probabilities = exps / np.sum(exps)
    
    loss_scalar = -np.log(probabilities[target_idx] + 1e-15)
    
    # Pack result back into the tracking graph
    out = Tensor(np.array([[loss_scalar]]), _children=(logits_tensor,), _op='cross_entropy')
    
    def _backward():
        derivative_array = probabilities.copy()
        derivative_array[target_idx] -= 1.0
        logits_tensor.grad += out.grad * derivative_array.reshape(logits_tensor.shape)
        
    out._backward = _backward
    return out
