"""
Contains unit tests for the data flow graph class.
"""
import logging
import unittest
from logic.graph import DataFlowGraph, DominateNodesSearchAlg

class TestDataFlowGraph(unittest.TestCase):
    """
    Tests the APIs of the class DataFlowGraph
    """
    def setUp(self) -> None:
        self.logger = logging.getLogger('dummy_logger')
        self.logger.setLevel(logging.DEBUG)


    def test_dgf_create(self):
        """
        Tests the creation of the dominate nodes in the data flow graph.
        """
        list_node_names = ["R", "C", "F", "G", "I", "J", "K", "B", "E", "A", "H", "D", "L"]
        list_edge_names = [("R", "C"), ("C", "F"), ("C", "G"), ("F", "I"), ("G", "I"),
                           ("G", "J"), ("J", "I"), ("I", "K"), ("K", "I"), ("H", "K"),
                           ("R", "B"), ("R", "A"), ("B", "A"), ("B", "D"), ("B", "E"),
                          ("E", "H"), ("H", "E"), ("A", "D"), ("D", "L"), ("L", "H")]
        dfg = DataFlowGraph(list_node_names, list_edge_names, "R", self.logger)
        for node_name in list_node_names:
            _ = dfg.can_reach_node_test_api(node_name, None)
            self.assertEqual(dfg.can_reach_node_test_api(node_name, None), True)


    def test_dominate_nodes(self):
        """
        Tests computation of dominate nodes in the data flow graph using reachability algorithm
        """
        list_node_names = ["R", "C", "F", "G", "I", "J", "K", "B", "E", "A", "H", "D", "L"]
        list_edge_names = [("R", "C"), ("C", "F"), ("C", "G"), ("F", "I"), ("G", "I"),
                           ("G", "J"), ("J", "I"), ("I", "K"), ("K", "I"), ("H", "K"),
                            ("R", "B"), ("R", "A"), ("B", "A"), ("B", "D"), ("B", "E"),
                            ("E", "H"), ("H", "E"), ("A", "D"), ("D", "L"), ("L", "H")]
        dfg = DataFlowGraph(list_node_names, list_edge_names, "R", self.logger)
        map_node_name_dominator_names = {
            "R": [], "I": ["R"], "K": ["R"], "C": ["R"], "H": ["R"], "E": ["R"], "A": ["R"],
            "D":["R"], "B": ["R"], "L": ["D", "R"], "F": ["C", "R"], "G": ["C", "R"],
            "J": ["C", "G", "R"]}
        for node_name in list_node_names:
            dominate_node_names = dfg.get_dominate_nodes(node_name,
                                    DominateNodesSearchAlg.REACHABILITY)
            self.assertEqual(set(dominate_node_names),
                             set(map_node_name_dominator_names[node_name]))


    def test_enumerate_nodes(self):
        """
        Tests the enumeration of nodes in the DFS search in the data flow graph.
        """
        list_node_names = ["R", "C", "F", "I", "K", "G",  "J",  "B", "E", "H",  "A",  "D", "L"]
        list_edge_names = [("R", "C"), ("C", "F"), ("C", "G"), ("F", "I"),
                           ("G", "I"), ("G", "J"),  ("I", "K"), ("K", "I"), ("J", "I"),
                           ("R", "B"), ("R", "A"), ("B", "E"), ("B", "A"), ("B", "D"),
                           ("E", "H"), ("H", "E"), ("H", "K"), ("A", "D"), ("D", "L"), ("L", "H")]
        dfg = DataFlowGraph(list_node_names, list_edge_names, "R", self.logger)
        dfg.dfs_enumerate_and_build_tree()
        nodes = dfg.get_nodes()
        nn_to_edg = {node.get_name() : node for node in nodes}

        parent_nodes = {"R": None, "C": nn_to_edg["R"],
                             "F": nn_to_edg["C"], "G": nn_to_edg["C"],
                             "I": nn_to_edg["F"], "J": nn_to_edg["G"],
                             "K": nn_to_edg["I"], "B": nn_to_edg["R"],
                             "E": nn_to_edg["B"], "H": nn_to_edg["E"],
                             "A": nn_to_edg["B"], "D": nn_to_edg["A"],
                             "L": nn_to_edg["D"] }
        # This only holds for this order of the nodes and edges
        # It is important to have this invariant testing dfs ordering
        for node_idx, node_name in enumerate(list_node_names):
            node = nodes[node_idx]
            self.assertEqual(node.get_dfs_node_id(), node_idx)
            if node.get_name() == "R":
                self.assertEqual(node.get_parent_node(), None)
            else:
                self.assertEqual(node.get_parent_node().get_name(),
                                 parent_nodes[node_name].get_name())


    def test_lengauer_tarjan_fast_algorithm(self):
        """
        Tests the Lengauer-Tarjan algorithm for the computation of the dominate nodes
        for one node.
        """
        list_node_names = ["R", "C", "F", "I", "K", "G",  "J",  "B", "E", "H",  "A",  "D", "L"]
        list_edge_names = [("R", "C"), ("C", "F"), ("C", "G"), ("F", "I"), ("G", "I"), ("G", "J"),
                            ("I", "K"), ("K", "I"), ("J", "I"), ("R", "B"), ("R", "A"),
                            ("B", "E"), ("B", "A"), ("B", "D"), ("E", "H"), ("H", "E"),
                            ("H", "K"), ("A", "D"), ("D", "L"), ("L", "H")]

        dfg = DataFlowGraph(list_node_names, list_edge_names, "R", self.logger)
        dom_nodes_j = dfg.get_dominate_nodes("J",
                            DominateNodesSearchAlg.LENGAUER_TARJAN_NON_OPTIMIZED)
        self.assertEqual(set(dom_nodes_j), set(["C","R", "G"]))


    def test_dominate_nodes_lengauer_tarjan(self):
        """
        Tests the Lengauer-Tarjan algorithm for the computation of the dominate nodes
        for each node.
        """
        list_node_names = ["R", "C", "F", "I", "K", "G",  "J",  "B", "E", "H",  "A",  "D", "L"]
        list_edge_names = [("R", "C"), ("C", "F"), ("C", "G"), ("F", "I"), ("G", "I"),
                           ("G", "J"),  ("I", "K"), ("K", "I"), ("J", "I"), ("R", "B"),
                           ("R", "A"), ("B", "E"), ("B", "A"), ("B", "D"), ("E", "H"),
                           ("H", "E"), ("H", "K"), ("A", "D"), ("D", "L"), ("L", "H")]

        dfg = DataFlowGraph(list_node_names, list_edge_names, "R", self.logger)
        map_node_name_dominator_names = {
            "R": [], "I": ["R"], "K": ["R"], "C": ["R"], "H": ["R"], "E": ["R"], "A": ["R"],
            "D":["R"], "B": ["R"], "L": ["D", "R"], "F": ["C", "R"], "G": ["C", "R"],
            "J": ["C", "G", "R"]}

        for node_name in list_node_names:
            dominate_node_names = dfg.get_dominate_nodes(node_name,
                                    DominateNodesSearchAlg.LENGAUER_TARJAN_NON_OPTIMIZED)
            self.assertEqual(set(dominate_node_names),
                             set(map_node_name_dominator_names[node_name]))


if __name__ == '__main__':
    unittest.main()
