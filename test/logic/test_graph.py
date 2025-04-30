import unittest
from logic.graph import DataFlowGraph
from logic.graph import DataFlowGraph

def add(a, b):
    return a + b

class TestMathFunctions(unittest.TestCase):

    def test_add_positive_numbers(self):
        self.assertEqual(add(2, 3), 4)

    def test_add_negative_numbers(self):
        self.assertEqual(add(-1, -1), -2)

    def test_add_zero(self):
        self.assertEqual(add(0, 0), 0)

if __name__ == '__main__':
    unittest.main()
