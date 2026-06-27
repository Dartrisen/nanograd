"""
Mathematical contraction primitives implemented using NanoGrad Tensor nodes.
Bypasses high-dimensional operations by casting contractions into 2D spaces.
"""

from __future__ import annotations
from nanograd.core.tensor import Tensor


def project_embedding_to_core(embedding: Tensor, core_weight: Tensor, r_left: int, r_right: int) -> Tensor:
    """
    Blends an input embedding vector with a third-order TT core matrix.
    
    :param embedding: Tensor of shape (B, D) where B is batch, D is embedding dim.
    :param core_weight: Tensor of shape (D, r_left * r_right).
    :return: A flattened transition operator block of shape (B, r_left * r_right).
    """
    return embedding.matmul(core_weight)


def apply_transition_to_state(state: Tensor, transition: Tensor, r_left: int, r_right: int) -> Tensor:
    """
    Updates the historical hidden state vector using a localized transition matrix.
    
    :param state: Old internal bond memory tensor of shape (B, r_left).
    :param transition: Flattened transition operator of shape (r_left, r_right).
    :return: An updated bond memory tensor of shape (B, r_right).
    """
    return state.matmul(transition)
