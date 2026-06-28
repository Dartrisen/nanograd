import unittest
import numpy as np

from nanograd.core.tensor import Tensor
from run_gpt_benchmark import cross_entropy_loss
from ttnn.model import TensorNetworkStateMachine


class TensorNetworkTrainingTests(unittest.TestCase):
    def test_state_update_preserves_gradients_for_core_parameters(self) -> None:
        model = TensorNetworkStateMachine(vocab_size=6, initial_bond_dim=3)
        seq = [0, 1, 2]
        targets = [1, 2, 3]

        model.zero_grad()
        logits_history = model.forward(seq)

        loss_node = Tensor(np.array([[0.0]]))
        for t in range(len(seq)):
            loss_node = loss_node + cross_entropy_loss(logits_history[t], targets[t])

        loss_node.backward()

        core_grad_norm = np.linalg.norm(model.memory.W_core.grad)
        self.assertGreater(core_grad_norm, 1e-6, "Core transition weights should receive non-zero gradients")

        embed_grad_norm = np.linalg.norm(model.memory.W_embed.grad)
        self.assertGreater(embed_grad_norm, 1e-6, "Token embedding weights should receive non-zero gradients")


if __name__ == "__main__":
    unittest.main()
