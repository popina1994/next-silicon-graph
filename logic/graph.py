from typing import List, Dict, Tuple
from enum import Enum
from collections import deque
import logging

class Node:
    def __init__(self, name: str):
        self.name = name
        self.out_edges = set()
        self.in_edges = set()

    def add_out_edges(self, out_node):
        self.out_edges.add(out_node)

    def add_in_edges(self, in_node):
        self.in_edges.add(in_node)

    def get_out_edges(self):
        return self.out_edges

    def get_in_edges(self):
        return self.in_edges

    def __str__(self) -> str:
        return self.name


class NodeDoesNotExist(Exception):
    pass

class VisitState(Enum):
    NOT_VISITED = 1
    VISITING = 2
    VISITED = 3

class DataFlowGraph:
    def __init__(self, node_names: List[str], edges: List[Tuple[str, str]], start_node_name: str):
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
    def reach_node(self, reach_node: Node, skip_node: Node):
        map_visit_state = {}

        for node in self.nodes:
            map_visit_state[node] = VisitState.NOT_VISITED
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


    """
    Find the dominator nodes for a given target node in a graph.

    A node is considered a dominator of the target node if removing it (skipping it)
    makes the target node unreachable from the start node.

    Args:
        reach_node_name (str): The name of the target node for which dominators
                               need to be identified.

    Returns:
        list of str: A list of node names that dominate the target node. This list
                     will include the start node and any node whose removal makes
                     the target node unreachable.

    Raises:
        NodeDoesNotExist: If the reach_node_name does not correspond to an existing node
                           in the graph.
    """
    def get_dominate_nodes(self, reach_node_name: str):
        if not reach_node_name in self.map_name_to_node:
            raise NodeDoesNotExist(f"Node to reach {reach_node_name} does not exist")
        reach_node = self.map_name_to_node[reach_node_name]
        if not self.reach_node(reach_node, None):
            return []
        dominate_nodes = [str(self.start_node)]
        for skip_node in self.nodes:
            if skip_node == self.start_node or skip_node == reach_node:
                continue
            if not self.reach_node(reach_node, skip_node):
                dominate_nodes.append(str(skip_node))

        return dominate_nodes
