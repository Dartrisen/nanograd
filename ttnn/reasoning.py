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
        """
        vocab_size = memory.vocab_size
        curr_bond = memory.bond_dim
        
        # 1. Unfold active weights into a matrix mapping across mode profiles
        matrix_view = memory.W_core.data.reshape(vocab_size * curr_bond, curr_bond)
        
        # 2. Execute economy Singular Value Decomposition
        try:
            U, S, Vt = np.linalg.svd(matrix_view, full_matrices=False)
        except np.linalg.LinAlgError:
            return False
        
        # 3. Measure relative energy allocations to determine the required rank
        total_energy = np.sum(S ** 2)
        if total_energy < 1e-10:
            return False
            
        cumulative_energy = np.cumsum(S ** 2) / total_energy
        ideal_rank = np.argmax(cumulative_energy >= (1.0 - self.eps ** 2)) + 1
        ideal_rank = max(self.min_rank, min(ideal_rank, self.max_rank))
        
        # 4. If a structural change is required, alter the model's internal parameters
        if ideal_rank != curr_bond:
            
            # --- ASCII VISUALIZATION OF NETWORK BRANCHES ---
            print("\n" + "≈" * 55)
            print("🧬 TOPOLOGY MUTATION TRIGGERED")
            print(f"Shift: Bond Dimension {curr_bond} -> {ideal_rank}")
            print("Singular Value Energy Spectrum:")
            
            max_s = np.max(S) if len(S) > 0 else 1.0
            for i, val in enumerate(S):
                bar_length = int((val / max_s) * 30)
                if i < ideal_rank:
                    print(f"  Branch {i:02d} | {'█' * bar_length:<30} | [ACTIVE]")
                else:
                    print(f"  Branch {i:02d} | {'░' * bar_length:<30} | [PRUNED]")
            print("≈" * 55 + "\n")
            # -----------------------------------------------

            # MATHEMATICAL FIX: Safely project the 3D tensor to the new dimensions
            # Extract current 3D state
            old_core_3d = memory.W_core.data.reshape(vocab_size, curr_bond, curr_bond)
            
            # Create a properly sized empty container for the new rank
            new_core_3d = np.zeros((vocab_size, ideal_rank, ideal_rank))
            
            # Transfer the existing active structural weights safely
            copy_bound = min(curr_bond, ideal_rank)
            new_core_3d[:, :copy_bound, :copy_bound] = old_core_3d[:, :copy_bound, :copy_bound]
            
            # Flatten to 2D for the standard routing engine
            new_core_data = new_core_3d.reshape(vocab_size, ideal_rank * ideal_rank)
            
            # Adapt output predictive tracking head (Your original logic!)
            new_head_data = np.zeros((ideal_rank, vocab_size))
            new_head_data[:copy_bound, :] = memory.W_head.data[:copy_bound, :]
            
            # Re-register variables into the memory module
            memory.bond_dim = ideal_rank
            memory.W_core = Tensor(new_core_data, label="W_core_evolved")
            memory.W_head = Tensor(new_head_data, label="W_head_evolved")
            return True
            
        return False
