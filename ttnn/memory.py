from __future__ import annotations
import numpy as np
from nanograd import Tensor, Module
from tt.linear import TTLinear


class TNMemory(Module):
    """
    Parametric memory cluster constructed with TT-Linear layers and multi-core transition tensors.
    Includes Bilinear TT gating structures for long-context (L > 500) recurrent stability.
    """
    def __init__(self, vocab_size: int, bond_dim: int, tt_rank: int = 16) -> None:
        self.vocab_size = vocab_size
        self.bond_dim = bond_dim
        self.tt_rank = tt_rank

        # TT-Linear Embedding layer
        self.W_embed = TTLinear(in_modes=(vocab_size,), out_modes=(bond_dim,), tt_rank=tt_rank)

        # TT-Linear Gate layer for bilinear state retention control
        self.W_gate = TTLinear(in_modes=(vocab_size,), out_modes=(bond_dim,), tt_rank=tt_rank)

        # Decomposed 2-core Tensor Train state transition operator chain
        raw_g1 = np.random.randn(vocab_size, tt_rank).astype(np.float32) * 0.05
        raw_g2 = np.random.randn(tt_rank, bond_dim, bond_dim).astype(np.float32) * 0.05
        self.G1_core = Tensor(raw_g1, label="G1_core")
        self.G2_core = Tensor(raw_g2, label="G2_core")

        # TT-Linear Projection and Output Prediction layers
        self.W_proj = TTLinear(in_modes=(bond_dim,), out_modes=(bond_dim,), tt_rank=tt_rank)
        self.W_head = TTLinear(in_modes=(bond_dim,), out_modes=(vocab_size,), tt_rank=tt_rank)

    def parameters(self) -> list[Tensor]:
        params = self.W_embed.parameters() + self.W_gate.parameters()
        params += [self.G1_core, self.G2_core]
        params += self.W_proj.parameters() + self.W_head.parameters()
        return params
