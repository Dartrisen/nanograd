import unittest
from unittest.mock import patch, MagicMock

from nanograd.layer import Layer
from nanograd.neuron import Neuron
from nanograd.value import Value


class TestLayer(unittest.TestCase):
    """Tests for Layer class."""

    def setUp(self):
        """Set up Layer."""
        self.inputs = 3
        self.outputs = 4

        # Mocking Neuron
        with patch('nanograd.neuron.Neuron', autospec=True) as MockNeuron:
            self.mock_neuron_instance = MockNeuron.return_value
            self.mock_neuron_instance.return_value = MagicMock()
            self.mock_neuron_instance.return_value.tanh.return_value = Value(0.5)
            self.layer = Layer(self.inputs, self.outputs)

    def test_initialization(self):
        self.assertEqual(len(self.layer.neurons), self.outputs, "Number of neurons should match the number of outputs")
        for neuron in self.layer.neurons:
            self.assertIsInstance(neuron, Neuron, "Each neuron should be an instance of Neuron")

    def test_call(self):
        with patch('nanograd.neuron.Neuron.__call__', return_value=Value(0.5)):
            x = [0.5, -0.3, 0.8]
            result = self.layer(x)
            self.assertEqual(len(result), self.outputs, "Output should have the same length as the number of neurons")
            for out in result:
                self.assertIsInstance(out, Value, "Each output should be an instance of Value")
                self.assertEqual(out.data, 0.5, "The output value should be 0.5")

    def test_parameters(self):
        with patch('nanograd.neuron.Neuron.parameters', return_value=[Value(0.5), Value(0.5)]) as mock_parameters:
            params = self.layer.parameters()
            expected_length = self.outputs * len(mock_parameters.return_value)
            self.assertEqual(len(params), expected_length, "Parameters should include all weights and biases of all neurons")
            for param in params:
                self.assertIsInstance(param, Value, "Each parameter should be an instance of Value")

    def test_zero_grad(self):
        for neuron in self.layer.neurons:
            for param in neuron.parameters():
                param.grad = 5

        self.layer.zero_grad()

        for neuron in self.layer.neurons:
            for param in neuron.parameters():
                self.assertEqual(param.grad, 0, "grad should be zeroed by zero_grad method")


if __name__ == '__main__':
    unittest.main()
