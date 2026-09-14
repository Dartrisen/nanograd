from __future__ import annotations
import numpy as np
from nanograd.core.tensor import Tensor


class ContractionRouter:
    """
    Executes sequence routing by contracting inputs sequentially across TT core chains.
    """
    @staticmethod
    def route_token(token_idx: int, vocab_size: int, G1_core: Tensor, G2_core: Tensor, bond_dim: int) -> Tensor:
        """
        Contracts token one-hot selection through (G1, G2) TT cores to yield a (bond_dim, bond_dim) matrix.
        """
        one_hot = np.zeros((1, vocab_size), dtype=np.float32)
        one_hot[0, token_idx] = 1.0
        one_hot_tensor = Tensor(one_hot, label=f"router_token_{token_idx}")

        # Extract token latent rank vector: (1, vocab_size) @ (vocab_size, tt_rank) -> (1, tt_rank)
        extracted_g1 = one_hot_tensor.matmul(G1_core)

        # Contract extracted rank vector into G2 core to obtain square state transition matrix:
        # (1, tt_rank) @ (tt_rank, bond_dim * bond_dim) -> (1, bond_dim * bond_dim)
        tt_rank = G1_core.data.shape[1]
        g2_flat = G2_core.reshape(tt_rank, bond_dim * bond_dim)
        transition_flat = extracted_g1.matmul(g2_flat)

        return transition_flat.reshape(bond_dim, bond_dim)
