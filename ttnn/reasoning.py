from __future__ import annotations
import numpy as np
from ttnn.memory import TNMemory
from nanograd.core.tensor import Tensor


class TopologyEvolver:
    """
    Dynamically resizes core matrix tracks by evaluating internal entropy.
    Supports both pruning uninformative paths and inflating capacity on demand.
    """
    def __init__(self, eps: float = 1e-2, min_rank: int = 4, max_rank: int = 128) -> None:
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

        # Unfold active weights into a matrix mapping across mode profiles
        matrix_view = memory.W_core.data.reshape(vocab_size * curr_bond, curr_bond)

        # Execute economy Singular Value Decomposition
        try:
            U, S, Vt = np.linalg.svd(matrix_view, full_matrices=False)
        except np.linalg.LinAlgError:
            return False

        # Measure relative energy allocations to determine the required rank
        total_energy = np.sum(S ** 2)
        if total_energy < 1e-10:
            return False

        cumulative_energy = np.cumsum(S ** 2) / total_energy

        # Calculate optimal structural rank based on energy threshold
        ideal_rank = np.argmax(cumulative_energy >= (1.0 - self.eps ** 2)) + 1

        # Look at the final singular value. If it holds significant energy,
        # the network is choking; signal a controlled expansion step.
        max_s = S[0] if S[0] > 0 else 1e-8
        if S[-1] > (max_s * 0.1) and curr_bond < self.max_rank:
            ideal_rank = min(curr_bond + 8, self.max_rank)

        # Constrain rank strictly within safety boundaries
        ideal_rank = max(self.min_rank, min(ideal_rank, self.max_rank))

        # Exit early if current topology matches the target configuration
        if ideal_rank == curr_bond:
            return False

        # --- ASCII VISUALIZATION OF NETWORK BRANCHES ---
        print("\n" + "≈" * 55)
        print("🧬 TOPOLOGY MUTATION TRIGGERED")
        print(f"Shift: Bond Dimension {curr_bond} -> {ideal_rank}")
        print("Singular Value Energy Spectrum:")

        for i, val in enumerate(S):
            bar_length = int((val / max_s) * 30)
            if i < ideal_rank:
                print(f"  Branch {i:02d} | {'█' * bar_length:<30} | [ACTIVE]")
            else:
                print(f"  Branch {i:02d} | {'░' * bar_length:<30} | [PRUNED]")
        print("≈" * 55 + "\n")
        # -----------------------------------------------

        # Extract current 3D state
        old_core_3d = memory.W_core.data.reshape(vocab_size, curr_bond, curr_bond)

        # Create properly scaled container for mutated operations
        new_core_3d = np.zeros((vocab_size, ideal_rank, ideal_rank))

        # Transfer identical active structural slices
        copy_bound = min(curr_bond, ideal_rank)
        new_core_3d[:, :copy_bound, :copy_bound] = old_core_3d[:, :copy_bound, :copy_bound]

        # EXPANSION TRICK: If growing, fill new channels with tiny exploration noise
        if ideal_rank > curr_bond:
            # Generate subtle discovery perturbations
            noise = np.random.randn(vocab_size, ideal_rank, ideal_rank) * 0.01
            # Apply only to the newly appended index boundaries
            new_core_3d[:, copy_bound:, :] = noise[:, copy_bound:, :]
            new_core_3d[:, :, copy_bound:] = noise[:, :, copy_bound:]

        # Flatten back to standard structural dimensions
        new_core_data = new_core_3d.reshape(vocab_size, ideal_rank * ideal_rank)

        # Adapt output predictive tracking head
        new_head_data = np.zeros((ideal_rank, vocab_size))
        new_head_data[:copy_bound, :] = memory.W_head.data[:copy_bound, :]

        # Adapt embedding matrix to match the new tracking shape
        new_embed_data = np.zeros((memory.W_embed.data.shape[0], ideal_rank))
        new_embed_data[:, :copy_bound] = memory.W_embed.data[:, :copy_bound]
        if ideal_rank > curr_bond:
            new_embed_data[:, copy_bound:] = np.random.randn(memory.W_embed.data.shape[0], ideal_rank - copy_bound) * 0.01

        # Re-register variables securely into the memory module
        memory.bond_dim = ideal_rank
        memory.W_core = Tensor(new_core_data, label="W_core_evolved")
        memory.W_head = Tensor(new_head_data, label="W_head_evolved")
        memory.W_embed = Tensor(new_embed_data, label="W_embed_evolved")

        return True
