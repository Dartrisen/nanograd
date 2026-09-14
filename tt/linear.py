from __future__ import annotations
import numpy as np
from nanograd import Tensor, Module


class TTLinear(Module):
    """
    Tensor Train Linear layer mapping input modes (n_1, ..., n_d) to output modes (m_1, ..., m_d)
    parameterized by a chain of 4D core tensors of shape (r_{k-1}, n_k, m_k, r_k).
    """
    def __init__(self, in_modes: tuple[int, ...], out_modes: tuple[int, ...], tt_rank: int = 4) -> None:
        assert len(in_modes) == len(out_modes), "Input and output mode dimensions must match."
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.d = len(in_modes)
        self.ranks = (1,) + (tt_rank,) * (self.d - 1) + (1,)

        self.cores: list[Tensor] = []
        for k in range(self.d):
            shape = (self.ranks[k], in_modes[k], out_modes[k], self.ranks[k+1])
            scale = 1.0 / np.sqrt(self.ranks[k] * in_modes[k])
            raw_data = np.random.randn(*shape).astype(np.float32) * scale
            self.cores.append(Tensor(raw_data, label=f"tt_linear_core_{k}"))

    def parameters(self) -> list[Tensor]:
        return list(self.cores)

    def forward(self, x: Tensor) -> Tensor:
        """
        Contracts input vector x through the TT core chain using block matrix products.
        """
        batch_size = x.data.shape[0]

        # Contract first core: (1, n_1, m_1, r_1) -> (n_1, m_1 * r_1)
        W_flat = self.cores[0].reshape(self.in_modes[0], self.out_modes[0] * self.ranks[1])

        for k in range(1, self.d):
            r_prev, n_k, m_k, r_next = self.cores[k].data.shape
            next_core_flat = self.cores[k].reshape(r_prev, n_k * m_k * r_next)
            W_flat = W_flat.matmul(next_core_flat).reshape(-1, r_next)

        W_final = W_flat.reshape(int(np.prod(self.in_modes)), int(np.prod(self.out_modes)))
        return x.matmul(W_final)
