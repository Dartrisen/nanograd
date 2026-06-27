import unittest

from nanograd import Layer, MLP, Module, Neuron, Tensor, Value, topo_sort_iterative


class TestPackageLayout(unittest.TestCase):
    def test_public_exports_are_available(self):
        self.assertTrue(callable(topo_sort_iterative))
        self.assertTrue(issubclass(Module, object))
        self.assertTrue(issubclass(Neuron, Module))
        self.assertTrue(issubclass(Layer, Module))
        self.assertTrue(issubclass(MLP, Module))
        self.assertIsInstance(Value(1.0), Value)
        self.assertIsInstance(Tensor([[1.0]]), Tensor)

    def test_topological_sort_visits_each_node_once(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        topo = topo_sort_iterative(c)

        self.assertEqual(topo.count(a), 1)
        self.assertEqual(topo.count(b), 1)
        self.assertEqual(topo.count(c), 1)
        self.assertLess(topo.index(a), topo.index(c))
        self.assertLess(topo.index(b), topo.index(c))
