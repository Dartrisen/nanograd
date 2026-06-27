"""
Topology Evolution and singular-value validation loops.
"""

from __future__ import annotations
import numpy as np
from ttnn.memory import TNMemory
from nanograd.core.tensor import Tensor


class TopologyEvolver:
    """
    Dynamically resizes core matrix tracks by evaluating internal entropy.
    """
    def __init__(self, eps: float = 1e-2, min_rank: int = 2, max_rank: int = 16) -> None:
        self.eps = eps
        self.min_rank = min_rank
        self.max_rank = max_rank

    def evolve_memory(self, memory: TNMemory) -> bool:
        """
        Computes an SVD over the model's active transition matrix data blocks.
        Truncates or inflates dimensions based on informational variance.
        
        :param memory: The memory module containing the transition matrix data.
        :return: True if structural properties were rewritten, otherwise False.
        """
        vocab_size = memory.vocab_size
        curr_bond = memory.bond_dim
        
        # 1. Unfold active weights into a matrix mapping across mode profiles
        # Shape: (vocab_size * curr_bond, curr_bond)
        matrix_view = memory.W_core.data.reshape(vocab_size * curr_bond, curr_bond)
        
        # 2. Execute economy Singular Value Decomposition
        U, S, Vt = np.linalg.svd(matrix_view, full_matrices=False)
        
        # 3. Measure relative energy allocations to determine the required rank
        total_energy = np.sum(S ** 2)
        if total_energy < 1e-10:
            return False
            
        cumulative_energy = np.cumsum(S ** 2) / total_energy
        ideal_rank = np.argmax(cumulative_energy >= (1.0 - self.eps ** 2)) + 1
        ideal_rank = max(self.min_rank, min(ideal_rank, self.max_rank))
        
        # 4. If a structural change is required, alter the model's internal parameters
        if ideal_rank != curr_bond:
            print(f"\n[Reasoning Engine] Structural evolution triggered! Changing bond_dim: {curr_bond} -> {ideal_rank}")
            
            # Truncate components to match target parameters
            U_trunc = U[:, :ideal_rank]
            S_trunc = S[:ideal_rank]
            Vt_trunc = Vt[:ideal_rank, :]
            
            # Construct a brand new parameter matrix layout
            evolved_matrix = U_trunc @ np.diag(S_trunc) @ Vt_trunc
            new_core_data = evolved_matrix.reshape(vocab_size, ideal_rank * ideal_rank)
            
            # Adapt output predictive tracking head to support new incoming state size
            new_head_data = np.zeros((ideal_rank, vocab_size))
            copy_bound = min(curr_bond, ideal_rank)
            new_head_data[:copy_bound, :] = memory.W_head.data[:copy_bound, :]
            
            # Re-register variables into the memory module
            memory.bond_dim = ideal_rank
            memory.W_core = Tensor(new_core_data, label="W_core_evolved")
            memory.W_head = Tensor(new_head_data, label="W_head_evolved")
            return True
            
        return False
