import unittest
from logic.graph import DataFlowGraph

def add(a, b):
    return a + b

class TestDataFlowGraph(unittest.TestCase):

    def test_add_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

    def test_dgf_create_simple(self):
        list_node_names = ["R", "C", "F", "G", "I", "J", "K", "B", "E", "A", "H", "D", "L"]
        list_edge_names = [("R", "C"), ("C", "F"), ("C", "G"), ("F", "I"), ("G", "I"), ("G", "J"), ("J", "I"), ("I", "K"), ("K", "I"), ("H", "K"),
        ("R", "B"), ("R", "A"), ("B", "A"), ("B", "D"), ("B", "E"),
        ("E", "H"), ("H", "E"), ("A", "D"), ("D", "L"), ("L", "H")]
        dfg = DataFlowGraph(list_node_names, list_edge_names, "R")
        for node_name in list_node_names:
            can_reach = dfg.can_reach_node_test_api(node_name, None)
            self.assertEqual(dfg.can_reach_node_test_api(node_name, None), True)




if __name__ == '__main__':
    unittest.main()
