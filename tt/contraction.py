"""
Core contraction runtime orchestration engine.
Executes the optimal path computed by the contraction optimizer using NanoGrad primitives.
"""

from __future__ import annotations
from nanograd.core.tensor import Tensor
import tt.ops as ops
from tt.optimizer import ContractionPathOptimizer


class TTContractionEngine:
    """
    Executes localized tensor train node contractions dynamically.
    """

    @staticmethod
    def contract_step(
        state: Tensor, 
        embedding: Tensor, 
        core_weight: Tensor, 
        r_left: int, 
        r_right: int
    ) -> Tensor:
        """
        Contracts an incoming embedding vector into the existing hidden state.

        :param state: Hidden state tensor of shape (B, r_left).
        :param embedding: Normalized token projection tensor of shape (B, D).
        :param core_weight: Core parameter memory matrix of shape (D, r_left * r_right).
        :param r_left: Left bond dimension of the hidden state.
        :param r_right: Right bond dimension of the hidden state.

        :return: Updated hidden state tensor of shape (B, r_right).
        """
        batch_size = embedding.data.shape[0]
        embed_dim = embedding.data.shape[1]

        path = ContractionPathOptimizer.optimize_step_path(
            batch_size, embed_dim, r_left, r_right
        )

        if path == 'extract_first':
            # 1. Isolate the localized transition operator
            # (B, D) @ (D, r_left * r_right) -> (B, r_left * r_right)
            flat_transition = ops.project_embedding_to_core(
                embedding, core_weight, r_left, r_right
            )
            
            # For B=1, reshape directly to a square matrix for clean state transformation
            # (r_left, r_right)
            transition_matrix = flat_transition.reshape(r_left, r_right)
            
            # 2. Update state memory via localized matrix multiplication
            # (B, r_left) @ (r_left, r_right) -> (B, r_right)
            new_state = ops.apply_transition_to_state(
                state, transition_matrix, r_left, r_right
            )
            return new_state

        else:
            # Fallback path: Reshape state into core dimensions first
            # (B, r_left) -> processed as matrix transformations
            flat_transition = ops.project_embedding_to_core(
                embedding, core_weight, r_left, r_right
            )
            transition_matrix = flat_transition.reshape(r_left, r_right)
            return state.matmul(transition_matrix)
