"""
Contraction path optimizer for NanoGrad Tensor Train operations.
Analyzes shape metadata to prevent memory bottlenecks during sequence processing.
"""

from __future__ import annotations


class ContractionPathOptimizer:
    """
    Evaluates equivalent algebraic paths to determine the lowest FLOP count.
    """

    @staticmethod
    def optimize_step_path(
        batch_size: int, 
        embed_dim: int, 
        r_left: int, 
        r_right: int
    ) -> str:
        """
        Determines whether it is cheaper to contract the embedding first, or contract the hidden state memory first.

        :param batch_size: Number of samples in the current batch.
        :param embed_dim: Dimensionality of the embedding vector.
        :param r_left: Left bond dimension of the hidden state.
        :param r_right: Right bond dimension of the hidden state.

        :return: 'extract_first' or 'state_first'
        """
        # Path 1: Project embedding to matrix, then multiply by state
        # Cost = (Batch * EmbedDim * r_left * r_right) + (Batch * r_left * r_right)
        cost_extract = (batch_size * embed_dim * r_left * r_right) + (batch_size * r_left * r_right)

        # Path 2: Contract state with core matrix along bonds first, then contract embedding
        # Cost = (Batch * r_left * EmbedDim * r_right) + (Batch * EmbedDim * r_right)
        cost_state_first = (batch_size * r_left * embed_dim * r_right) + (batch_size * embed_dim * r_right)

        if cost_extract <= cost_state_first:
            return 'extract_first'
        return 'state_first'