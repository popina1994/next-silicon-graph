"""
Module providing a support for data flow graph.
"""
from typing import List, Dict, Tuple, Set
from enum import Enum
from collections import deque
import sys
from logic.node import Node

class NodeDoesNotExist(Exception):
    """
    Exception class in the case the node does not exist in a data flow graph.
    """


class VisitState(Enum):
    """
    Enum Class used to track the state of a node in BFS and DFS.
    """
    NOT_VISITED = 1
    VISITING = 2
    VISITED = 3


class DominateNodesSearchAlg(Enum):
    """
    Enum class that specifies algorithms for solving dominate nodes
    in the data flow graph problem.
    """
    REACHABILITY = 1
    LENGAUER_TARJAN_NON_OPTIMIZED = 2
    SPECIALIZED = 4
    #LENGAUER_TARJAN_OPTIMIZED  = 3


class LengauerTarjanAlgorithm:
    """
    Encapsulates the current state information used in
    Lenagauer Tarjan Algorithm.
    """
    preorder_nodes: List[Node]
    semi: Dict[Node, int]
    bucket: Dict[Node, Set[Node]]
    ancestor: Dict[Node, Node | None]
    label: Dict[Node, Node]
    dom: Dict[Node, Node]


    def __init__(self, dfs: 'DataFlowGraph') -> None:
        dfs.logger.info("Starting DFS to enumerate vertices")
        preorder_nodes = dfs.dfs_enumerate_and_build_tree()
        dfs.logger.info("Finished DFS and enumeration of vertices")

        self.preorder_nodes = preorder_nodes
        self.semi = {node: node.get_dfs_node_id()
                     if node is not None else None for node in preorder_nodes}
        self.bucket = {node: set() for node in preorder_nodes}
        self.dom = {node: None for node in preorder_nodes}

        self.ancestor = {node: None for node in preorder_nodes}
        self.label = {node: node for node in preorder_nodes}


    def get_immediate_domminator(self, node: Node)->Node:
        """
        After the algorithm finishes returns imme"""
        return self.dom.get(node, None)


    def link(self, node_v: Node, node_w: Node)->None:
        """
        One to one mapping to the function in the paper
        """
        self.ancestor[node_w] = self.preorder_nodes[node_v.get_dfs_node_id()]


    def compress(self, node_v: Node):
        """
        One to one mapping to the function in the paper
        """
        if self.ancestor[node_v] is not None and self.ancestor[self.ancestor[node_v]] is not None:
            self.compress(self.ancestor[node_v])
            if self.semi[self.label[self.ancestor[node_v]]] < self.semi[self.label[node_v]]:
                self.label[node_v] = self.label[self.ancestor[node_v]]
            self.ancestor[node_v] = self.ancestor[self.ancestor[node_v]]


    def eval(self, node_v: Node | None) -> Node:
        """
        One to one mapping to the function in the paper
        """
        if self.ancestor[node_v] is None: # pylint: disable=no-else-return
            return node_v
        else:
            self.compress(node_v)
            return self.label[node_v]


    def compute_semi_dominators_and_implicit_dominators(self):
        """
        Steps 2 and 3 of Lengauer-Tarjan algorithm
        """
        for node_w in self.preorder_nodes[:0:-1]:
            if node_w is None:
                continue
            for prev_node in node_w.get_in_edges():
                if prev_node.get_dfs_node_id() is None:
                    continue
                # print(prev_node, node_w)
                node_u = self.eval(prev_node)
                if self.semi[node_u] < self.semi[node_w]:
                    self.semi[node_w] = self.semi[node_u]
            self.bucket[self.preorder_nodes[self.semi[node_w]]].add(node_w)
            self.link(node_w.get_parent_node(), node_w)
            bucket_par_w_copy = self.bucket[node_w.get_parent_node()].copy()
            for node_v in bucket_par_w_copy:
                self.bucket[node_w.get_parent_node()].remove(node_v)
                node_u = self.eval(node_v)
                if self.semi[node_u] < self.semi[node_v]:
                    self.dom[node_v] = node_u
                else:
                    self.dom[node_v] = node_w.get_parent_node()


    def explicit_dominator(self, start_node: Node):
        """
        Steps 4 and 3 of Lengauer-Tarjan algorithm
        """
        for node_w in self.preorder_nodes[1:]:
            if node_w is None:
                continue
            if self.dom[node_w] != self.preorder_nodes[self.semi[node_w]]:
                self.dom[node_w] = self.dom[self.dom[node_w]]
        self.dom[start_node] = None


class DominateNodesReachabilityAlgorithm:
    """
    The class that implementes the dominate nodes computation in O(|V| * |E|).
    """
    def __init__(self, start_node: Node,  dfg: 'DataFlowGraph') -> None:
        self.start_node = start_node
        self.dfg = dfg


    def get_dominate_node_names(self, reach_node: Node) -> List[str]:
        """
        The time complexity of finding all come-before nodes for reach node reach_node
        is O(|E| * |V|).
        First we need to check if the vertex reach_node is reachable from the start node.
        If not, there are no come-before nodes.
        For each vertex v in the graph, the algorithm check if the node reach_node is reachable.
        If the reach_node is not reachable, then the vertex v is a come-before-node.
        """
        if not self.dfg.can_reach_node(reach_node, None):
            return []

        dominate_node_names = [str(self.start_node)]
        for skip_node in self.dfg.get_nodes():
            if skip_node == self.start_node or \
                skip_node == reach_node: # pylint: disable=consider-using-in
                continue
            if not self.dfg.can_reach_node(reach_node, skip_node):
                dominate_node_names.append(str(skip_node))

        return dominate_node_names


class SpecializeDominatorNodesAlgorithm:
    """
    Speciallize class that implements the algorithm for the dominator nodes of specific
    code in O(|V| + |E|)
    """
    def __init__(self, start_node: Node, dfg: 'DataFlowGraph', logger) -> None:
        self.logger = logger
        self.logger.info("Starting DFS enumeration for the specialized graph")
        self.start_node = start_node
        self.dfg = dfg
        preorder_nodes = dfg.dfs_enumerate_and_build_tree()
        self.preorder_nodes = preorder_nodes
        self.logger.info("Ending DFS enumeration for the specialized graph")


    def get_dominate_node_names(self, reach_node: Node):
        """
        Returns all the dominate nodes for the node reach_node.
        It computes by enumerating nodes in the DFS tree using DFS search.
        Then extracts the path to the root for reach_node.
        Then for each nodes, checks if there is a bypass towards
        some nodes so other nodes in the path are skipped.
        Then it removes all nodes that are skipped and return the remaining nodes as
        dominating ones. """
        self.logger.info("Computing path to the reach node")
        path_nodes = [reach_node]
        cur_node = reach_node
        while cur_node.get_parent_node() is not None:
            path_nodes.append(cur_node.get_parent_node())
            cur_node = cur_node.get_parent_node()


        path_nodes.reverse()
        # path_nodes.pop()
        self.logger.info(f"Computed path to the reach node {path_nodes}")
        path_nodes_to_delete = [0] * len(path_nodes)
        if path_nodes[0] != self.start_node:
            # Node not reachable
            return []


        self.logger.info("Starting computation of"
                         "the least recent ancesstor in the that to each node")
        map_path_nodes = {node: path_idx for path_idx, node in enumerate(path_nodes)}
        map_common_anc = self.dfg.dfs_enumerate_and_build_tree_compute_common(path_nodes)
        self.logger.info(f"ending computation of the least "
                         f"recent ancesstor in the that to each node {map_common_anc}")

        self.logger.info("Starting markation of parts of the "
                         "path to delete of the specialized algorithm")
        for node in self.preorder_nodes:
            for out_node in node.get_out_edges():
                if out_node in map_path_nodes:
                    out_node_path_idx = map_path_nodes[out_node]
                    path_idx_anc = map_common_anc[node]
                    if path_idx_anc + 1 <= out_node_path_idx:
                        path_nodes_to_delete[path_idx_anc + 1] -= 1
                        path_nodes_to_delete[out_node_path_idx] += 1

        self.logger.info("Ending markation of parts of path to delete of the"
                                  "specialized algorithm")
        dominate_node_names = []

        self.logger.info("Starting computation of the dominate nodes using "
                                " the specialized algorithm")
        cur_sum = 0
        for path_node_idx, path_node_to_delete in enumerate(path_nodes_to_delete):
            cur_sum += path_node_to_delete
            if cur_sum >= 0 and path_nodes[path_node_idx] != reach_node:
                self.logger.info(f"N: {path_nodes[path_node_idx]} RN: {reach_node}")
                dominate_node_names.append(str(path_nodes[path_node_idx]))

        self.logger.info(f"Ending computation of the dominate nodes using "
                                 f"the specialized algorithm: {dominate_node_names}")

        return dominate_node_names
    # 1. DFS in order to enumerate nodes
    # 2. Get the path to the reach node by going from the reach node to the start_node
    # 3. For all other nodes, check if they contain a link towards any element in this path.
    #    Simple preoirder node - nodes on the path and check all links to see
    # if the destination is in the path.
    # if it is mark the range for deletion
    # Finally, just iterated over the path and see if any nodes are remaining.



class DataFlowGraph:
    """
    Class that implements data flow graphs. It supports reachability checks, and
    computation of all the nodes that dominate the specific node.
    """
    nodes: List[Node]
    start_node: Node
    cur_preorder_idx: int
    cur_path_idx: int

    def __init__(self, node_names: List[str], edges: List[Tuple[str, str]], start_node_name: str,
                 logger):
        self.logger = logger
        self.nodes = [None] * len(node_names)
        self.map_name_to_node = {}
        for idx_node_name, node_name in enumerate(node_names):
            new_node = Node(node_name)
            self.nodes[idx_node_name] = new_node
            self.map_name_to_node[node_name] = new_node
        if not start_node_name in self.map_name_to_node:
            raise NodeDoesNotExist(f"Node to reach {start_node_name} does not exist")
        self.start_node = self.map_name_to_node[start_node_name]

        for edge in edges:
            (node_in_name, node_out_name) = edge
            if not node_in_name in self.map_name_to_node:
                raise NodeDoesNotExist(f"Node to reach {node_in_name} does not exist")
            node_in = self.map_name_to_node[node_in_name]
            if not node_out_name in self.map_name_to_node:
                raise NodeDoesNotExist(f"Node to reach {node_out_name} does not exist")
            node_out = self.map_name_to_node[node_out_name]
            node_in.add_out_edges(node_out)
            node_out.add_in_edges(node_in)

        self.cur_preorder_idx = None
        self.cur_path_idx = None


    def __str__(self) -> str:
        out = f"START NODE: {self.start_node}"
        for node in self.nodes:
            out += f"\n NODE: {node}"
            for out_edge in node.get_out_edges():
                out += f"\n EDGE: ({node},{out_edge})"

        return out

    def get_nodes(self)->List[Node]:
        """
        Returns a lit of Nodes that compose the current data flow graph
        """
        return self.nodes.copy()


    def can_reach_node(self, reach_node: Node, skip_node: Node):
        """
    Determines whether the reach_node is reachable from the start_node,
    skipping a specified skip_node, using BFS traversal.

    Args:
        reach_node (Node): The target node to determine reachability.
        skip_node (Node): A node to exclude from the search (treated as if it doesn't exist).

    Returns:
        bool: True if reach_node is reachable from start_node without traversing skip_node,
              False otherwise.

    Notes:
        Complexity: O(E + V) in the worst case.
    """
        map_visit_state= {node: VisitState.NOT_VISITED for node in self.nodes}
        dq_to_visit = deque()

        dq_to_visit.append(self.start_node)
        map_visit_state[self.start_node] = VisitState.VISITING

        while not len(dq_to_visit) == 0:
            cur_node = dq_to_visit.popleft()
            map_visit_state[cur_node] = VisitState.VISITED
            if cur_node == reach_node:
                return True
            for adj_node in cur_node.get_out_edges():
                if map_visit_state[adj_node]  == VisitState.NOT_VISITED and adj_node != skip_node:
                    map_visit_state[adj_node] = VisitState.VISITING
                    dq_to_visit.append(adj_node)

        return False


    def can_reach_node_test_api(self, reach_node_name: str, skip_node_name: str):
        """
        Wrapper around the function can_reach_node that is used for unit testing.
        Instead of passing nodes reach_node and skip_node by instances, the function
        passes the nodes by name
        """
        reach_node = self.map_name_to_node[reach_node_name]
        skip_node = self.map_name_to_node[skip_node_name] if skip_node_name is not None else None
        return self.can_reach_node(reach_node, skip_node)


    def dfs_enumerate_and_build_tree_wrapper(self, cur_node: Node,
                                             preorder_nodes: List[Node],
                                             map_visit_state: Dict[Node, VisitState]):
        """
        It continues DFS search of a data flow graph by visiting current node cur_node,
        updating dfs_node_id of cur_node to the next available id
        updating the state of node in  map_visit_state
        """
        self.cur_preorder_idx += 1
        map_visit_state[cur_node] = VisitState.VISITED
        preorder_nodes[self.cur_preorder_idx] = cur_node
        cur_node.set_dfs_node_id(self.cur_preorder_idx)
        self.logger.info(f"Enumerated node {cur_node} by {self.cur_preorder_idx}")
        for adj_node in cur_node.get_out_edges():
            if map_visit_state[adj_node] == VisitState.NOT_VISITED:
                adj_node.set_parent_node(cur_node)
                self.dfs_enumerate_and_build_tree_wrapper(adj_node, preorder_nodes, map_visit_state)


    def dfs_enumerate_and_build_tree(self):
        """
        Depth first search of data flow graph starting from start_node
        where each node is enumerated in the order of a DFS search.
        """
        sys.setrecursionlimit(2000)  # default is usually 1000
        self.logger.info("Starting to enumerate nodes by DFS")
        map_visit_state = {node: VisitState.NOT_VISITED for node in self.nodes}
        preorder_nodes = [None] * len(self.nodes)
        self.cur_preorder_idx = -1
        self.dfs_enumerate_and_build_tree_wrapper(self.start_node, preorder_nodes, map_visit_state)

        return preorder_nodes


    def dfs_enumerate_and_build_tree_common_wrapper(self, cur_node: Node,
                                             s_path_nodes: Set[Node],
                                             map_common_anc: Dict[Node, int],
                                             cur_path_idx: int):
        """
        It continues DFS search of a data flow graph by visiting current node cur_node,
        updating dfs_node_id of cur_node to the next available id
        updating the state of node in  map_visit_state
        """

        map_common_anc[cur_node] = cur_path_idx
        self.logger.info(f"Current node {cur_node}, {cur_path_idx}")

        # map_visit_state[cur_node] = VisitState.VISITED
        for adj_node in cur_node.get_out_edges():
            if (adj_node not in s_path_nodes) and (map_common_anc[adj_node] is None):
                self.dfs_enumerate_and_build_tree_common_wrapper(adj_node,
                                    s_path_nodes, map_common_anc, cur_path_idx)




    def dfs_enumerate_and_build_tree_compute_common(self, path_nodes: List[Node]):
        """
        For each node e, computes the minimum node id of all nodes
        in the path_nodes that can reach the node e without going through the path.
        """
        sys.setrecursionlimit(2000)  # default is usually 1000
        self.logger.info("Starting to compute most common ancesstor with path nodes by DFS")
        map_common_anc = {node: None for node in self.nodes}
        for path_node_idx, path_node in enumerate(path_nodes):
            map_common_anc[path_node] = path_node_idx

        s_path_nodes = set(path_nodes)
        for path_node_idx, path_node in enumerate(path_nodes):
            self.cur_path_idx = path_node_idx
            self.dfs_enumerate_and_build_tree_common_wrapper(path_node,
                                                s_path_nodes, map_common_anc, path_node_idx)

        return map_common_anc


    def get_dominate_lengauer_tarjan_fast_algorithm(self, reach_node: Node)->List[str]:
        """
        The algorithm is one to one implementation of the following immediate
        dominator set algorithm:
        https://dl.acm.org/doi/pdf/10.1145/357062.357071 know ans Lengauer-Tarjan algorithm.
        The time complexity of the algorithm is O(|E| * log |V|)

        Afterwards we just traverse from the reach node through
        immediate dominator nodes to the entry node in the worst case in
        O(|E|) to get all dominator nodes (come-before-nodes).

        I am listing renames I introduced and in the type setting you have
        type definitions that satisfy the types from the paper
        All enumerations in the implementation start from zero
        pred -> in_edges
        succ -> out_edges
        parent ->parent
        vertex -> preorder_nodes
        semi - semi
        bucket - bucket
        dom -dom
    """
        len_tar = LengauerTarjanAlgorithm(self)

        self.logger.info("Starting compute semi dominators and implicit dominators")
        len_tar.compute_semi_dominators_and_implicit_dominators()
        self.logger.info("Ending compute semi dominators and implicit dominators")

        self.logger.info("Starting building immediate dominators tree")
        len_tar.explicit_dominator(self.start_node)
        self.logger.info("Ending building immediate dominators tree")

        self.logger.info(f"Getting immediate domminator node for {reach_node}")
        imm_dom_node =  len_tar.get_immediate_domminator(reach_node)
        self.logger.info(f"The immediate domminator node for {reach_node} is {imm_dom_node}")
        dom_nodes_a = []
        while imm_dom_node is not None:
            dom_nodes_a.append(imm_dom_node)
            self.logger.info(f"Getting immediate domminator node for {imm_dom_node}")
            imm_dom_node = len_tar.get_immediate_domminator(imm_dom_node)

        dom_node_names = [node.get_name() for node in dom_nodes_a]
        dom_node_names.reverse()

        return dom_node_names


    def get_dominate_nodes_reachability_alg(self, reach_node: Node) -> List[str]:
        """
    The time complexity of finding all come-before nodes for reach node reach_node is O(|E| * |V|).
    First we need to check if the vertex reach_node is reachable from the start node.
    If not, there are no come-before nodes.
    For each vertex v in the graph, the algorithm check if the node reach_node is reachable.
    If the reach_node is not reachable, then the vertex v is a come-before-node.

        """
        reach_alg = DominateNodesReachabilityAlgorithm(self.start_node, self)

        return reach_alg.get_dominate_node_names(reach_node)


    def get_dominate_nodes_specialized_algorithm(self, reach_node: Node) -> List[str]:
        """
    The time complexity of finding all come-before nodes for reach node reach_node is O(|E| + |V|).
        """
        reach_alg = SpecializeDominatorNodesAlgorithm(self.start_node, self, self.logger)

        return reach_alg.get_dominate_node_names(reach_node)


    def get_dominate_nodes(self, reach_node_name: str,
        search_alg: DominateNodesSearchAlg=
        DominateNodesSearchAlg.LENGAUER_TARJAN_NON_OPTIMIZED) ->List[str]:
        """
    Find the dominator nodes for a given target node in a graph.
    A node is considered a dominator of the target node if all paths from the start node
    to the target node lead over the dominator node.
    Args:
        reach_node_name (str): The name of the target node for which dominators
                            need to be identified.
    Returns:
        list of str: A list of node names that dominate the target node. For non-start node,
                    this list will include the start node and any node whose removal makes
                    the target node unreachable. If the reach node is start_node,
                    this list will be empty.
    Raises:
        NodeDoesNotExist: If the reach_node_name does not correspond to an existing node
                           in the graph.
    """
        if reach_node_name == self.start_node.get_name():
            return []
        if not reach_node_name in self.map_name_to_node:
            raise NodeDoesNotExist(f"Node to reach {reach_node_name} does not exist")
        reach_node = self.map_name_to_node[reach_node_name]
        dominate_node_names = []
        self.logger.info("Dominate nodes for the data flow graph {self}"
                         "and the reach node {reach_node_name} has started")
        if search_alg == DominateNodesSearchAlg.REACHABILITY:
            self.logger.info("Computing dominate nodes using reachability algorithm")
            dominate_node_names = self.get_dominate_nodes_reachability_alg(reach_node)
        elif search_alg == DominateNodesSearchAlg.LENGAUER_TARJAN_NON_OPTIMIZED:
            self.logger.info("Computing dominate nodes using lengauer-tarjan algorithm")
            dominate_node_names = self.get_dominate_lengauer_tarjan_fast_algorithm(reach_node)
        elif search_alg == DominateNodesSearchAlg.SPECIALIZED:
            self.logger.info("Computing dominate nodes using specialized algorithm")
            dominate_node_names = self.get_dominate_nodes_specialized_algorithm(reach_node)

        return dominate_node_names
