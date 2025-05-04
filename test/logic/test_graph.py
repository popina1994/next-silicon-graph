"""
Contains unit tests for the data flow graph class.
"""
import logging
import unittest
import random
import time
from logic.graph import DataFlowGraph, DominateNodesSearchAlg


class TestDataFlowGraph(unittest.TestCase):
    """
    Tests the APIs of the class DataFlowGraph
    """
    def setUp(self) -> None:
        self.logger = logging.getLogger('dummy_logger')
        self.logger.setLevel(logging.DEBUG)

    @staticmethod
    def random_graph(num_nodes: int, edge_prob: float, seed: int, logger)->DataFlowGraph:
        """
        Generate a random data flow graph with num_nodes where the probability
        of an edge between two nodes is edge_prob. For the probability distribution
        we use seed seed
        """
        node_names = [str(i) for i in range(num_nodes - 1)]
        edges = []
        random.seed(seed)
        for node_name_1 in node_names:
            for node_name_2 in node_names:
                if node_name_1 != node_name_2 and random.random() < edge_prob:
                    # print("EDGE", node_name_1, node_name_2)
                    edge = [node_name_1, node_name_2]
                    edges.append(edge)

        start_node_name = str(num_nodes  - 1)
        node_names.append(start_node_name)
        for node_name in node_names:
            if start_node_name != node_name and random.random() < edge_prob:
                edge = [start_node_name, node_name]
                edges.append(edge)


        return DataFlowGraph(node_names=node_names, edges=edges,
                             start_node_name=start_node_name, logger=logger)



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


    def test_dominate_nodes_reachability_all_nodes(self):
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


    def test_dominate_nodes_lengauer_tarjan_one_node(self):
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


    def test_dominate_nodes_lengauer_tarjan_all_nodes(self):
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


    def test_dominate_nodes_specialized_one_node(self):
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
                            DominateNodesSearchAlg.SPECIALIZED)
        self.assertEqual(set(dom_nodes_j), set(["C","R", "G"]))


    def test_dominate_nodes_specialized_one_node_2(self):
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
        dom_nodes_j = dfg.get_dominate_nodes("I",
                            DominateNodesSearchAlg.SPECIALIZED)
        self.assertEqual(set(dom_nodes_j), set(["R"]))


    def test_dominate_nodes_specialized_all_nodes(self):
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
            self.logger.info("Starting for %s", node_name)
            dominate_node_names = dfg.get_dominate_nodes(node_name,
                                    DominateNodesSearchAlg.SPECIALIZED)
            self.assertEqual(set(dominate_node_names),
                             set(map_node_name_dominator_names[node_name]))
            self.logger.info("Successful for %s", node_name)


    @unittest.skip("Skipping this test for now")
    def test_large_data_flow_graphs_lengauer_tarjan(self):
        """
        Tests the Lengauer-Tarjan algorithm for the computation of the dominate nodes
        for each node.
        """
        dfg = TestDataFlowGraph.random_graph(300, 0.1, 0, self.logger)
        # print(dfg)
        for node in dfg.get_nodes():
            node_name = str(node)
            start = time.time()
            dominate_node_names_lt = dfg.get_dominate_nodes(node_name,
                                    DominateNodesSearchAlg.LENGAUER_TARJAN_NON_OPTIMIZED)
            end = time.time()
            print(f"Execution time Lengauer Tarjan: {end - start:.6f} seconds")
            start = time.time()
            dominate_node_names_reach = dfg.get_dominate_nodes(node_name,
                                    DominateNodesSearchAlg.REACHABILITY)
            end = time.time()
            print(f"Execution time Reachability: {end - start:.6f} seconds")
            self.assertEqual(set(dominate_node_names_lt),
                             set(dominate_node_names_reach))

    @unittest.skip("Skipping this test for now")
    def test_large_data_flow_graphs_specialized(self):
        """
        Tests the Lengauer-Tarjan algorithm for the computation of the dominate nodes
        for each node.
        """
        dfg = TestDataFlowGraph.random_graph(300, 0.1, 0, self.logger)
        # print(dfg)
        for node in dfg.get_nodes():
            node_name = str(node)
            start = time.time()
            dominate_node_names_lt = dfg.get_dominate_nodes(node_name,
                                    DominateNodesSearchAlg.SPECIALIZED)
            end = time.time()
            print(f"Execution time Specialized: {end - start:.6f} seconds")
            start = time.time()
            dominate_node_names_reach = dfg.get_dominate_nodes(node_name,
                                    DominateNodesSearchAlg.REACHABILITY)
            end = time.time()
            print(f"Execution time Reachability: {end - start:.6f} seconds")
            self.assertEqual(set(dominate_node_names_lt),
                             set(dominate_node_names_reach))



if __name__ == '__main__':
    unittest.main()
