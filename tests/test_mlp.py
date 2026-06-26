import unittest
from unittest.mock import patch

from nanograd.mlp import MLP
from nanograd.layer import Layer
from nanograd.value import Value


class TestMLP(unittest.TestCase):
    """Tests for MLP class."""

    def setUp(self):
        """Set up MLP."""
        self.inputs = 3
        self.outputs = [4, 3, 3]

        with patch('nanograd.mlp.MLP', autospec=True):
            self.mock_mlp = MLP(self.inputs, self.outputs)

            for layer in self.mock_mlp.layers:
                for param in layer.parameters():
                    param.grad = 1.0

    def test_initialization(self):
        self.assertEqual(len(self.mock_mlp.layers), len(self.outputs), "Number of layers should match the number of outputs")
        for layer in self.mock_mlp.layers:
            self.assertIsInstance(layer, Layer, "Each layer should be an instance of Layer")

    def test_call(self):
        with patch('nanograd.neuron.Neuron.__call__', return_value=Value(0.5)):
            x = [0.5, -0.3, 0.8]
            result = self.mock_mlp(x)
            print(result)
            self.assertEqual(len(result), self.outputs[-1], "Output should have the same length as the last layer's output size")
            for out in result:
                self.assertIsInstance(out, Value, "Each output should be an instance of Value")
                self.assertEqual(out.data, 0.5, "The output value should be 0.5")

    def test_parameters(self):
        with patch('nanograd.neuron.Neuron.parameters', return_value=[Value(0.5), Value(0.5), Value(0.5)]) as mock_parameters:
            params = self.mock_mlp.parameters()
            expected_params = sum(len(mock_parameters.return_value) * neurons for neurons in self.outputs)
            self.assertEqual(len(params), expected_params, "Parameters length should match the sum of parameters from all layers")
            for param in params:
                self.assertIsInstance(param, Value, "Each parameter should be an instance of Value")

    def test_zero_grad(self):
        self.mock_mlp.zero_grad()

        for layer in self.mock_mlp.layers:
            for param in layer.parameters():
                self.assertEqual(param.grad, 0, "Parameter gradient should be zeroed by zero_grad method")


if __name__ == '__main__':
    unittest.main()
