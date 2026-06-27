"""
Container abstractions representing a global Tensor Train Network topology.
Manages validation routines, structural properties, and full tensor reconstructions.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from tt.core import TTCore


class TensorTrain:
    """
    A structural sequence container enclosing an ordered chain of TTCore matrices.
    """

    def __init__(self, cores: List[TTCore]) -> None:
        """
        Instantiates a TensorTrain instance.

        :param cores: Ordered sequence of individual TTCore nodes.
            
        :raises ValueError: If the core chain fails topological validation rules.
        """
        if not cores:
            raise ValueError("A TensorTrain network must contain at least one TTCore.")
            
        self._cores = list(cores)
        self._validate_topology()

    def _validate_topology(self) -> None:
        """
        Enforces standard boundary conditions and core continuity checks.
        
        :raises ValueError: If individual core dimensions fail to connect sequentially,
                            or if boundary dimensions do not equal 1.
        """
        # Boundary Rule: Extreme outer dimensions must be anchored to 1
        if self._cores[0].rank_left != 1:
            raise ValueError(
                f"Boundary failure: First core rank_left must be 1. Got {self._cores[0].rank_left}"
            )
        if self._cores[-1].rank_right != 1:
            raise ValueError(
                f"Boundary failure: Last core rank_right must be 1. Got {self._cores[-1].rank_right}"
            )

        # Continuity Rule: Interconnected core rank interfaces must match exactly
        for i in range(len(self._cores) - 1):
            left_out = self._cores[i].rank_right
            right_in = self._cores[i+1].rank_left
            if left_out != right_in:
                raise ValueError(
                    f"Topology linkage mismatch at interface {i} -> {i+1}: "
                    f"Left core outputs rank {left_out}, but Right core demands rank {right_in}."
                )

    @property
    def cores(self) -> List[TTCore]:
        """Returns a copy of the list of underlying cores to maintain structural safety."""
        return list(self._cores)

    @property
    def order(self) -> int:
        """Returns the total number of tensor dimensions (dimensionality / order)."""
        return len(self._cores)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Computes the original uncompressed multi-index dense target shape."""
        return tuple(core.mode_size for core in self._cores)

    @property
    def ranks(self) -> Tuple[int, ...]:
        """Returns the complete structural rank footprint sequence."""
        bonds = [self._cores[0].rank_left]
        for core in self._cores:
            bonds.append(core.rank_right)
        return tuple(bonds)

    @property
    def num_parameters(self) -> int:
        """Calculates the total scalar parameter allocation across the matrix chain."""
        return sum(core.data.size for core in self._cores)

    def to_tensor(self) -> np.ndarray:
        """
        Contracts the sequence of cores to reconstruct the original dense tensor.

        :return: A dense multi-dimensional NumPy array matching self.shape.
        """
        # Start contraction with the first core
        # Shape: (1, n_1, r_1) -> squeeze left boundary -> (n_1, r_1)
        current = self._cores[0].data
        current = current.reshape(current.shape[1], current.shape[2])
        
        # Contract remaining cores sequentially from left to right
        for i in range(1, len(self._cores)):
            next_core = self._cores[i].data  # Shape: (r_{i-1}, n_i, r_i)
            r_left, n_i, r_right = next_core.shape
            
            # Reshape next core to matrix for clean multiplication: (r_{i-1}, n_i * r_i)
            next_matrix = next_core.reshape(r_left, n_i * r_right)
            
            # Contract current intermediate tracking state with the next matrix block
            # (Product of all previous modes, r_{i-1}) @ (r_{i-1}, n_i * r_i)
            current = current @ next_matrix
            
            # Group newly accumulated mode dimensions together with the active right rank
            current = current.reshape(-1, r_right)
            
        # Strip final trailing outer rank boundary dimension (which equals 1)
        final_shape = self.shape
        reconstructed = current.reshape(final_shape)
        reconstructed[np.abs(reconstructed) < 1e-12] = 0.0
        return reconstructed

    def __repr__(self) -> str:
        dense_elements = int(np.prod(self.shape))
        compression = dense_elements / max(1, self.num_parameters)
        
        lines = [
            "TensorTrain",
            "-----------",
            f"Order      : {self.order}",
            f"Shape      : {self.shape}",
            f"Ranks      : {self.ranks}",
            f"Parameters : {self.num_parameters}",
            f"Compression: {compression:.1f}x"
        ]
        return "\n".join(lines)
