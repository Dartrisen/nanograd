"""
Core datastructures for the Tensor Train network framework.
Manages individual TT-cores, encapsulating metadata and tracking orthogonality.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


class TTCore:
    """
    An immutable representation of a single third-order Tensor Train Core.
    
    A TT-core is a 3-dimensional array with structural layout:
    Shape: (rank_left, mode_size, rank_right)
    """

    def __init__(
        self, 
        data: np.ndarray, 
        orthogonalized: bool = False
    ) -> None:
        """
        Initializes a TTCore instance.

        :param data: A 3D NumPy array representing the core content.
        :param orthogonalized: Flag indicating if the core satisfies orthogonal constraints.

        :raises ValueError: If the input array does not have exactly 3 dimensions.
        """
        if data.ndim != 3:
            raise ValueError(
                f"TTCore data must be exactly 3-dimensional. Got shape {data.shape}"
            )
            
        # Ensure underlying data block is immutable to prevent accidental shifts
        self._data = data.copy()
        self._data.setflags(write=False)
        
        self._shape = self._data.shape
        self._orthogonalized = orthogonalized

    @property
    def data(self) -> np.ndarray:
        """Exposes the internal read-only NumPy multidimensional array."""
        return self._data

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Returns the 3D shape tuple of the core: (rank_left, mode_size, rank_right)."""
        return self._shape

    @property
    def rank_left(self) -> int:
        """Returns the left bond dimension (r_{k-1})."""
        return self._shape[0]

    @property
    def mode_size(self) -> int:
        """Returns the physical tensor dimension size (n_k) of this coordinate node."""
        return self._shape[1]

    @property
    def rank_right(self) -> int:
        """Returns the right bond dimension (r_k)."""
        return self._shape[2]

    @property
    def orthogonalized(self) -> bool:
        """Indicates whether this core has been swept into an orthogonal state."""
        return self._orthogonalized

    @property
    def dtype(self) -> np.dtype:
        """Returns the numeric precision datatype of the allocation block."""
        return self._data.dtype

    def __repr__(self) -> str:
        return (
            f"TTCore(shape={self.shape}, "
            f"orthogonalized={self.orthogonalized}, "
            f"dtype={self.dtype})"
        )
