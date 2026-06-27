"""
Input-guided extraction systems mapping tokens into active transformation planes.
"""

from __future__ import annotations
import numpy as np
from nanograd.core.tensor import Tensor


class ContractionRouter:
    """
    Routes structural inputs into square operational operators.
    """
    @staticmethod
    def route_token(token_idx: int, vocab_size: int, W_core: Tensor, bond_dim: int) -> Tensor:
        """
        Constructs an explicit algebraic extraction path to isolate a single
        token matrix out of a global matrix storage block.
        """
        # 1. Instantiate a one-hot vector trace: Shape (1, vocab_size)
        one_hot = np.zeros((1, vocab_size))
        one_hot[0, token_idx] = 1.0
        one_hot_tensor = Tensor(one_hot, label=f"router_token_{token_idx}")
        
        # 2. Extract the associated flattened slice via matrix multiplication
        # (1, vocab_size) @ (vocab_size, bond_dim * bond_dim) -> (1, bond_dim * bond_dim)
        flat_transition = one_hot_tensor.matmul(W_core)
        
        # 3. Shape the localized operator back into a square transition matrix
        # (bond_dim, bond_dim)
        return flat_transition.reshape(bond_dim, bond_dim)
