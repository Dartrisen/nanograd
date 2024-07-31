import unittest
from unittest.mock import patch

from nanograd.neuron import Neuron
from nanograd.value import Value


class TestNeuron(unittest.TestCase):
    """Tests for Neuron class."""

    def setUp(self):
        """Set up Neuron."""
        self.inputs = 3
        with patch('random.uniform', return_value=0.5):
            self.neuron = Neuron(self.inputs)
        self.x = [0.5, -0.3, 0.8]

    def test_initialization(self):
        self.assertEqual(len(self.neuron.w), self.inputs, "Number of weights should match the number of inputs")
        self.assertIsInstance(self.neuron.b, Value, "Bias should be an instance of Value")
        for weight in self.neuron.w:
            self.assertEqual(weight.data, 0.5, "Weights should be initialized to 0.5")
        self.assertEqual(self.neuron.b.data, 0.5, "Bias should be initialized to 0.5")

    def test_call(self):
        with patch('nanograd.neuron.Neuron.__call__', return_value=Value(0.5)):
            result = self.neuron(self.x)
            self.assertIsInstance(result, Value, "The output of Neuron should be an instance of Value")
            self.assertEqual(result.data, 0.5, msg="The output value is incorrect")

    def test_parameters(self):
        params = self.neuron.parameters()
        self.assertEqual(len(params), self.inputs + 1, "Parameters should include all weights and the bias")
        for param in params:
            self.assertIsInstance(param, Value, "Each parameter should be an instance of Value")

    def test_zero_grad(self):
        for param in self.neuron.parameters():
            param.grad = 5  # Set some initial non-zero value for grad

        self.neuron.zero_grad()

        for param in self.neuron.parameters():
            self.assertEqual(param.grad, 0, "grad should be zeroed by zero_grad method")


if __name__ == '__main__':
    unittest.main()