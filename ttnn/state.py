"""
Live hidden state buffers and memory trace controllers for the TNSM framework.
"""

from __future__ import annotations
import numpy as np
from nanograd.core.tensor import Tensor


class MachineState:
    """
    Tracks and updates the active low-rank bond memory configurations with norm-stabilization.
    """
    def __init__(self, bond_dim: int, batch_size: int = 1) -> None:
        self.bond_dim = bond_dim
        self.batch_size = batch_size
        self.h = None
        self.reset()

    def reset(self) -> None:
        """Anchors the boundary condition to a normalized uniform state."""
        initial_block = np.ones((self.batch_size, self.bond_dim)) / np.sqrt(self.bond_dim)
        self.h = Tensor(initial_block, label="live_bond_state")

    def update(self, new_h: Tensor) -> None:
        """
        Overwrites the internal state, applying a standard L2 vector normalization 
        to prevent exponential numerical explosion or decay.
        """
        # Calculate the raw vector norm directly out of the underlying numpy array
        raw_norm = np.linalg.norm(new_h.data, ord=2, axis=1, keepdims=True)
        # Avoid zero division
        normalized_data = new_h.data / (raw_norm + 1e-8)
        
        # Instantiate an unlinked Tensor to house the stabilized values
        self.h = Tensor(normalized_data, label="live_bond_state")
