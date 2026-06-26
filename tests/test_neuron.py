import unittest
from unittest.mock import patch

from nanograd.neuron import Neuron
from nanograd.value import Value


class TestNeuron(unittest.TestCase):
    """Tests for the Neuron class."""

    @patch('random.uniform', return_value=0.5)
    def setUp(self, mock_random):
        """Set up Neuron."""
        self.inputs = 3
        self.x = [0.5, -0.3, 0.8]
        self.mock_neuron = Neuron(self.inputs)
        # non-zero grads
        for param in self.mock_neuron.parameters():
            param.grad = 5

    def test_initialization(self):
        self.assertEqual(len(self.mock_neuron.w), self.inputs, "Number of weights should match the number of inputs")
        self.assertIsInstance(self.mock_neuron.b, Value, "Bias should be an instance of Value")
        for weight in self.mock_neuron.w:
            self.assertEqual(weight.data, 0.5, "Weights should be initialized to 0.5")
        self.assertEqual(self.mock_neuron.b.data, 0.5, "Bias should be initialized to 0.5")

    def test_call_actual(self):
        result = self.mock_neuron(self.x)
        self.assertIsInstance(result, Value, "The output of Neuron should be an instance of Value")

    @patch('nanograd.neuron.Neuron.__call__', return_value=Value(0.5))
    def test_call(self, mock_call):
        result = self.mock_neuron(self.x)
        self.assertIsInstance(result, Value, "The output of Neuron should be an instance of Value")
        self.assertEqual(result.data, 0.5, msg="The output value is incorrect")
        mock_call.assert_called_once_with(self.x)

    def test_parameters(self):
        params = self.mock_neuron.parameters()
        self.assertEqual(len(params), self.inputs + 1, "Parameters should include all weights and the bias")
        for param in params:
            self.assertIsInstance(param, Value, "Each parameter should be an instance of Value")

    def test_zero_grad(self):
        self.mock_neuron.zero_grad()
        for param in self.mock_neuron.parameters():
            self.assertEqual(param.grad, 0, "grad should be zeroed by zero_grad method")


if __name__ == '__main__':
    unittest.main()
