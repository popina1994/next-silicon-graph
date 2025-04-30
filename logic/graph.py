from typing import List, Dict, Tuple, Set
from enum import Enum
from collections import deque
import logging

class Node:
    name: str
    out_edges: List['Node']
    in_edges: List['Node']
    dfs_node_id: int
    parent_node: 'Node'

    def __init__(self, name: str):
        self.name = name
        self.out_edges = []
        self.in_edges = []
        self.dfs_node_id = -1
        self.parent_node = None

    def add_out_edges(self, out_node: 'Node'):
        self.out_edges.append(out_node)

    def add_in_edges(self, in_node: 'Node'):
        self.in_edges.append(in_node)

    def get_out_edges(self):
        return self.out_edges

    def get_in_edges(self):
        return self.in_edges

    def __str__(self) -> str:
        return self.name

    def set_dfs_node_id(self, node_id: int):
        self.dfs_node_id = node_id

    def get_dfs_node_id(self)-> int:
        return self.dfs_node_id

    def set_parent_node(self, par_node: 'Node'):
        self.parent_node = par_node

    def get_parent_node(self):
        return self.parent_node

    def get_name(self) -> str:
        return self.name


class NodeDoesNotExist(Exception):
    pass

class VisitState(Enum):
    NOT_VISITED = 1
    VISITING = 2
    VISITED = 3


class DominateNodesSearchAlg(Enum):
    REACHABILITY = 1
    LENGAUER_TARJAN_NON_OPTIMIZED = 2
    #LENGAUER_TARJAN_OPTIMIZED  = 3

class DataFlowGraph:
    nodes: List[Node]
    start_node: Node
    preorder_nodes: List[Node]
    semi: Dict[Node, int]
    bucket: Dict[Node, Set[Node]]
    ancestor: Dict[Node, Node]
    label: Dict[Node, Node]
    dom: Dict[Node, Node]

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


    def get_nodes(self)->List[Node]:
        return self.nodes.copy()

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
    def can_reach_node(self, reach_node: Node, skip_node: Node):
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
        reach_node = self.map_name_to_node[reach_node_name]
        skip_node = self.map_name_to_node[skip_node_name] if skip_node_name is not None else None
        return self.can_reach_node(reach_node, skip_node)


    def dfs_enumerate_and_build_tree_wrapper(self, cur_node: Node,
                                             map_visit_state: Dict[Node, VisitState]):
        self.cur_preorder_idx += 1
        map_visit_state[cur_node] = VisitState.VISITED
        self.preorder_nodes[self.cur_preorder_idx] = cur_node
        cur_node.set_dfs_node_id(self.cur_preorder_idx)
        self.logger.info(f"Enumerated node {cur_node} by {self.cur_preorder_idx}")
        for adj_node in cur_node.get_out_edges():
            if map_visit_state[adj_node] == VisitState.NOT_VISITED:
                adj_node.set_parent_node(cur_node)
                self.dfs_enumerate_and_build_tree_wrapper(adj_node, map_visit_state)


    def dfs_enumerate_and_build_tree(self):
        self.logger.info("Starting to enumerate nodes by DFS")
        map_visit_state = {node: VisitState.NOT_VISITED for node in self.nodes}
        self.preorder_nodes = [None] * len(self.nodes)
        self.cur_preorder_idx = -1
        self.dfs_enumerate_and_build_tree_wrapper(self.start_node, map_visit_state)


    def link(self, node_v: Node, node_w: Node)->None:
        self.ancestor[node_w] = self.preorder_nodes[node_v.get_dfs_node_id()]


    def compress(self, node_v: Node):
        if self.ancestor[node_v] is not None and self.ancestor[self.ancestor[node_v]] is not None:
            self.compress(self.ancestor[node_v])
            if self.semi[self.label[self.ancestor[node_v]]] < self.semi[self.label[node_v]]:
                self.label[node_v] = self.label[self.ancestor[node_v]]
            self.ancestor[node_v] = self.ancestor[self.ancestor[node_v]]


    def eval(self, node_v: Node) -> Node:
        if self.ancestor[node_v] is None:
            return node_v
        else:
            self.compress(node_v)
            return self.label[node_v]


    def compute_semi_dominators_and_implicit_dominators(self):
        for node_w in self.preorder_nodes[:0:-1]:
            for prev_node in node_w.get_in_edges():
                node_u = self.eval(prev_node)
                if self.semi[node_u] < self.semi[node_w]:
                    self.semi[node_w] = self.semi[node_u]
            self.bucket[self.preorder_nodes[self.semi[node_w]]].add(node_w)
            self.link(node_w.get_parent_node(), node_w)
            bucket_par_w_copy = self.bucket[node_w.get_parent_node()].copy()
            for node_v in bucket_par_w_copy:
                self.bucket[node_w.get_parent_node()].remove(node_v)
                node_u = self.eval(node_v)
                self.dom[node_v] = node_u if self.semi[node_u] < self.semi[node_v] else node_w.get_parent_node()


    def explicit_dominator(self):
        for node_w in self.preorder_nodes[1:]:
            if self.dom[node_w] != self.preorder_nodes[self.semi[node_w]]:
                self.dom[node_w] = self.dom[self.dom[node_w]]
        self.dom[self.start_node] = None



    """
        The algorithm is one to one implementation of the following immediate dominator set algorithm:
        https://dl.acm.org/doi/pdf/10.1145/357062.357071 know ans Lengauer-Tarjan algorithm.
        The time complexity of the algorithm is O(|E| * log |V|)

        Afterwards we just traverse from the reach node through
        immediate dominator nodes to the entry node in the worst case in
        O(|E|) to get all dominator nodes (come-before-nodes).

        I am listing renames I introduced and in the type setting you have type definitions that satisfy the types from the paper
        All enumerations in the implementation start from zero
        pred -> in_edges
        succ -> out_edges
        parent ->parent
        vertex -> preorder_nodes
        semi - semi
        bucket - bucket
        dom -dom
    """
    def get_dominate_lengauer_tarjan_fast_algorithm(self, reach_node: Node):
        self.logger.info("Starting DFS to enumerate vertices")
        self.dfs_enumerate_and_build_tree()
        self.logger.info("Finished DFS and enumeration of vertices")
        self.semi = {node: node.get_dfs_node_id() for node in self.nodes}
        self.bucket = {node: set() for node in self.nodes}
        self.dom = {node: None for node in self.nodes}

        self.ancestor = {node: None for node in self.nodes}
        self.label = {node: node for node in self.nodes}
        self.logger.info("Starting compute semi dominators and implicit dominators")
        self.compute_semi_dominators_and_implicit_dominators()
        self.logger.info("Ending compute semi dominators and implicit dominators")
        self.logger.info("Starting building immediate dominators tree")
        self.explicit_dominator()
        self.logger.info("Ending building immediate dominators tree")

        imm_dom_node = self.dom[reach_node]
        dom_nodes_a = []
        while imm_dom_node is not None:
            dom_nodes_a.append(imm_dom_node)
            imm_dom_node = self.dom[imm_dom_node]

        dom_node_names = [node.get_name() for node in dom_nodes_a]
        dom_node_names.reverse()

        return dom_node_names

    """
        The time complexity of finding all come-before nodes for reach node reach_node is O(|E| * |V|).
        First we need to check if the vertex reach_node is reachable from the start node.
        If not, there are no come-before nodes.
        For each vertex v in the graph, the algorithm check if the node reach_node is reachable.
        If the reach_node is not reachable, then the vertex v is a come-before-node.

    """
    def get_dominate_nodes_reachability_alg(self, reach_node: Node):
        if not self.can_reach_node(reach_node, None):
            return []
        dominate_nodes = [str(self.start_node)]
        for skip_node in self.nodes:
            if skip_node == self.start_node or skip_node == reach_node:
                continue
            if not self.can_reach_node(reach_node, skip_node):
                dominate_nodes.append(str(skip_node))

        return dominate_nodes

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
                     the target node unreachable. If the reach node is start_node, this list will be empty.

    Raises:
        NodeDoesNotExist: If the reach_node_name does not correspond to an existing node
                           in the graph.
    """
    def get_dominate_nodes(self, reach_node_name: str, search_alg: DominateNodesSearchAlg=DominateNodesSearchAlg.LENGAUER_TARJAN_NON_OPTIMIZED) ->List[str]:
        if reach_node_name == self.start_node.get_name():
            return []
        if not reach_node_name in self.map_name_to_node:
            raise NodeDoesNotExist(f"Node to reach {reach_node_name} does not exist")
        reach_node = self.map_name_to_node[reach_node_name]
        if search_alg == DominateNodesSearchAlg.REACHABILITY:
            self.logger.info("Computing dominate nodes using reachability algorithm")
            dominate_node_names = self.get_dominate_nodes_reachability_alg(reach_node)
        elif search_alg == DominateNodesSearchAlg.LENGAUER_TARJAN_NON_OPTIMIZED:
            self.logger.info("Computing dominate nodes using lengauer-tarjan algorithm")
            dominate_node_names = self.get_dominate_lengauer_tarjan_fast_algorithm(reach_node)

        return dominate_node_names