"""
Adaptive Moment Estimation (Adam) optimizer module for NanoGrad Tensor networks.
Manages running momentum histories across dynamically evolving structural weights.
"""

from __future__ import annotations
import numpy as np
from nanograd.core.tensor import Tensor
from nanograd.module import Module


class Adam(Module):
    """
    Implements the Adam optimization algorithm tailored for NanoGrad Tensor instances.
    """
    def __init__(self, parameters: list[Tensor], lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize moment vectors using parameter memory addresses as tracking keys
        self.m = {}
        self.v = {}
        self._sync_moments()

    def _sync_moments(self) -> None:
        """Ensures fresh tracking slots and correct shapes exist for all parameters."""
        for p in self.parameters:
            p_id = id(p)
            # Check if ID is missing OR if the underlying shape has mutated
            if p_id not in self.m or self.m[p_id].shape != p.data.shape:
                self.m[p_id] = np.zeros_like(p.data)
                self.v[p_id] = np.zeros_like(p.data)

    def step(self, max_norm: float = 1.0) -> None:
        """
        Applies gradient clipping and updates weight allocations via bias-corrected moments.
        """
        self.t += 1
        # Synchronize in case the reasoning engine mutated core shapes mid-flight
        self._sync_moments()
        
        for p in self.parameters:
            if p.grad is None:
                continue
                
            p_id = id(p)
            
            # Element-wise gradient clipping to stabilize deep unrolled sequence steps
            clipped_grad = np.clip(p.grad, -max_norm, max_norm)
            
            # Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            self.m[p_id] = self.beta1 * self.m[p_id] + (1.0 - self.beta1) * clipped_grad
            
            # Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            self.v[p_id] = self.beta2 * self.v[p_id] + (1.0 - self.beta2) * (clipped_grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[p_id] / (1.0 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[p_id] / (1.0 - self.beta2 ** self.t)
            
            # Update parameter data values
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
