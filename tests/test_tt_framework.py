"""
Validation suite testing correctness across the Tensor Train framework modules.
"""

import unittest
import numpy as np
from tt.core import TTCore
from tt.tensor_train import TensorTrain
import tt.decomposition as decomposition


class TestTensorTrainFramework(unittest.TestCase):
    """Unit tests for the Tensor Train framework, including core validation, topology checks, and reconstruction fidelity."""
    def setUp(self) -> None:
        # Seed random initialization loops for test consistency
        np.random.seed(42)

    def test_core_immutability(self) -> None:
        """Verifies that TTCore content blocks reject internal mutations."""
        raw_data = np.random.randn(2, 3, 4)
        core = TTCore(raw_data)
        
        with self.assertRaises(ValueError):
            core.data[0, 0, 0] = 99.0

    def test_invalid_topology_detection(self) -> None:
        """Verifies that the container intercepts broken dimension linkages."""
        c1 = TTCore(np.random.randn(1, 4, 3))
        c2_bad = TTCore(np.random.randn(2, 4, 1))  # Invalid link dimension: 3 != 2
        
        with self.assertRaises(ValueError):
            TensorTrain([c1, c2_bad])

    def test_boundary_rank_enforcement(self) -> None:
        """Ensures extreme outer borders are anchored to 1."""
        c_bad_left = TTCore(np.random.randn(2, 3, 1))
        with self.assertRaises(ValueError):
            TensorTrain([c_bad_left])

    def test_perfect_reconstruction(self) -> None:
        """Ensures dense tensors reconstruct flawlessly when rank limits are omitted."""
        dense_target = np.random.randn(3, 4, 5)
        tt = decomposition.from_tensor(dense_target)
        
        reconstructed = tt.to_tensor()
        np.testing.assert_array_almost_equal(dense_target, reconstructed, decimal=4)

    def test_rank_truncation_limits(self) -> None:
        """Verifies hard ceiling constraint limits across internal core nodes."""
        dense_target = np.random.randn(4, 4, 4, 4)
        max_allowed_rank = 2
        
        tt = decomposition.from_tensor(dense_target, max_rank=max_allowed_rank)
        
        for rank in tt.ranks:
            self.assertTrue(rank <= max_allowed_rank)

    def test_precision_energy_truncation(self) -> None:
        """Validates that epsilon tolerance drops near-zero singular values."""
        # Create a low-rank matrix structure manually obscured by noise
        r1 = np.random.randn(4, 1)
        r2 = np.random.randn(1, 4)
        low_rank_matrix = r1 @ r2  # Explicit rank-1 structure
        
        # Expand across an extra dimension to form an order-3 tensor
        dense_tensor = np.stack([low_rank_matrix, low_rank_matrix * 0.5], axis=-1)
        
        # Decompose using an absolute filtering threshold
        tt = decomposition.from_tensor(dense_tensor, eps=1e-1)
        
        # Verify compression successfully drops negligible trailing components
        self.assertTrue(max(tt.ranks) < 4)

    def test_representation_output(self) -> None:
        """Confirms that visual string tracking maps correctly without crashes."""
        dense_target = np.random.randn(2, 2, 2)
        tt = decomposition.from_tensor(dense_target)
        info_string = str(tt)
        
        self.assertIn("TensorTrain", info_string)
        self.assertIn("Shape", info_string)
        self.assertIn("Ranks", info_string)


if __name__ == "__main__":
    unittest.main()