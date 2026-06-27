"""
Decoupled parametric memory clusters for the Tensor Network State Machine.
"""

from __future__ import annotations
import numpy as np
from typing import List
from nanograd import Tensor
from nanograd import Module


class TNMemory(Module):
    """
    Houses the low-rank core blocks and prediction projections.
    """
    def __init__(self, vocab_size: int, bond_dim: int) -> None:
        self.vocab_size = vocab_size
        self.bond_dim = bond_dim
        
        # Structural Core: Maps tokens to spatial matrix operations
        # Shape layout: (vocab_size, bond_dim * bond_dim)
        raw_core = np.random.randn(vocab_size, bond_dim * bond_dim) * 0.1
        self.W_core = Tensor(raw_core, label="W_core")
        
        # Predictive Projection Head: Decodes active state back to vocabulary spaces
        # Shape layout: (bond_dim, vocab_size)
        raw_head = np.random.randn(bond_dim, vocab_size) * 0.1
        self.W_head = Tensor(raw_head, label="W_head")

    def parameters(self) -> List[Tensor]:
        """Returns the active parameter allocations."""
        return [self.W_core, self.W_head]
