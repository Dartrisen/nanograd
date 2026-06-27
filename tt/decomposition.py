"""
Algorithms for decomposing dense, higher-order tensors into Tensor Train structures.
Implements standard TT-SVD routines with robust rank-truncation mechanisms.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from tt.core import TTCore
from tt.tensor_train import TensorTrain


def from_tensor(
    tensor: np.ndarray, 
    max_rank: Optional[int] = None, 
    eps: Optional[float] = None
) -> TensorTrain:
    """
    Decomposes a dense multidimensional array into a TensorTrain instance via TT-SVD.

    :param tensor: The dense input source ndarray to compress.
    :param max_rank: Upper limit constraint for individual core internal ranks.
    :param eps: Singular value precision threshold. Truncates ranks dynamically
                when cumulative energy falls below this margin.

    :return: A completely validated, optimally compressed TensorTrain sequence instance.
    :raises ValueError: If the input tensor is not at least 1-dimensional.
    """
    src_shape = tensor.shape
    order = len(src_shape)
    
    if order < 1:
        raise ValueError("Cannot perform TT decomposition on a scalar value.")

    # Use a higher-precision working dtype to reduce reconstruction drift
    # from repeated SVD truncations.
    active_matrix = np.array(tensor, dtype=np.float64, copy=True)
    current_rank_left = 1
    cores_accumulator = []
    
    for k in range(order - 1):
        mode_n = src_shape[k]
        
        # Reshape active block into a 2D matrix splitting the current mode index
        # Target shape dimension: (current_rank_left * mode_n, remaining_dense_elements)
        rows_dimension = current_rank_left * mode_n
        active_matrix = active_matrix.reshape(rows_dimension, -1)
        
        # Compute economy Singular Value Decomposition
        U, S, Vt = np.linalg.svd(active_matrix, full_matrices=False)
        
        # Determine the target rank under constraints
        chosen_rank = len(S)
        if max_rank is not None:
            chosen_rank = min(chosen_rank, max_rank)
            
        if eps is not None and len(S) > 1:
            # Analyze relative cumulative energy across singular values
            total_energy = np.sum(S ** 2)
            if total_energy > 1e-15:
                # Find minimum rank required to satisfy precision threshold
                cumulative_energy = np.cumsum(S ** 2) / total_energy
                energy_mask = cumulative_energy >= (1.0 - eps ** 2)
                if np.any(energy_mask):
                    truncated_rank = np.argmax(energy_mask) + 1
                    chosen_rank = min(chosen_rank, truncated_rank)
        
        # Enforce valid minimum rank boundaries
        chosen_rank = max(1, chosen_rank)
        
        # Isolate components tracking back to chosen rank constraints
        U_truncated = U[:, :chosen_rank]
        S_truncated = S[:chosen_rank]
        Vt_truncated = Vt[:chosen_rank, :]
        
        # Reconstruct the current localized TTCore block
        core_data = U_truncated.reshape(current_rank_left, mode_n, chosen_rank)
        cores_accumulator.append(TTCore(core_data))
        
        # Pass the residual matrix block downstream to the next mode index loop
        active_matrix = np.diag(S_truncated) @ Vt_truncated
        current_rank_left = chosen_rank

    # Handle the final boundary core node from remaining data blocks
    final_mode_n = src_shape[-1]
    final_core_data = active_matrix.reshape(current_rank_left, final_mode_n, 1)
    cores_accumulator.append(TTCore(final_core_data))
    
    return TensorTrain(cores_accumulator)
