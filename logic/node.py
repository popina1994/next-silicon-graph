"""
Module providing a support for nodes in the data flow graph.
"""

from typing import List

class Node:
    """ Class representing a node. """
    name: str
    out_edges: List['Node']
    in_edges: List['Node']
    dfs_node_id: int
    parent_node: 'Node'

    def __init__(self, name: str):
        self.name = name
        self.out_edges = []
        self.in_edges = []
        self.dfs_node_id = None
        self.parent_node = None


    def add_out_edges(self, out_node: 'Node'):
        """
        Adds out edges in the data flow graph.
        """
        self.out_edges.append(out_node)


    def add_in_edges(self, in_node: 'Node'):
        """
        Adds in edges in the data flow graph.
        """
        self.in_edges.append(in_node)


    def get_out_edges(self) -> List['Node']:
        """
        Returns out edges in the data flow graph.
        """
        return self.out_edges


    def get_in_edges(self) -> List['Node']:
        """
        Returns in edges in the data flow graph.
        """
        return self.in_edges

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def set_dfs_node_id(self, node_id: int):
        """
        Sets the id to  be node_id. Used to enumerate nodes in a dfs tree.
        """
        self.dfs_node_id = node_id


    def get_dfs_node_id(self)-> int:
        """
        Gets the id to  be node_id.
        """
        return self.dfs_node_id


    def set_parent_node(self, par_node: 'Node'):
        """
        Sets the parent node in DFS tree to par_node
        """
        self.parent_node = par_node


    def get_parent_node(self) -> 'Node':
        """
        Gets the parent node in DFS tree.
        """
        return self.parent_node


    def get_name(self) -> str:
        """
        Returns the name of the node. It is unique.
        """
        return self.name
