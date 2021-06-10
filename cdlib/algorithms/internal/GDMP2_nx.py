"""
Name :: UnityId
Shruti Kuber :: skuber
Abhishek Lingwal :: aslingwa
Raunaq Saxena :: rsaxena

Chen, Jie, and Yousef Saad.
"Dense subgraph extraction with application to algorithms detection."
IEEE Transactions on Knowledge and Data Engineering 24.7 (2012): 1216-1230.

Reference internal: https://github.com/imabhishekl/CSC591_Community_Detection
"""

import numpy as np
import networkx as nx
import scipy


class __Node:
    def __init__(self, name):
        self.name = name
        self.left = None
        self.right = None
        self.parent = None
        self.num_edges = 0
        self.vertices = set()
        self.density = 0


class __Tree:
    def __init__(self):
        self.root = None

    def findLCA_Node(self, src_node, dest_node):
        while src_node is not None:
            if dest_node.name in src_node.vertices:
                return src_node
            src_node = src_node.parent
        return None

    def count_vertices_and_edges(self, edges_list, nodes_list):
        for edge in edges_list:
            lca_node = None

            src_node = nodes_list[edge[0]] if nodes_list.__contains__(edge[0]) else None
            dst_node = nodes_list[edge[1]] if nodes_list.__contains__(edge[1]) else None
            if src_node is not None and dst_node is not None:
                lca_node = self.findLCA_Node(src_node, dst_node)

            if lca_node is not None:
                lca_node.num_edges = lca_node.num_edges + 1

    def count_vertices_and_edges_wrap(self, root):
        if root.left is not None and root.right is not None:
            self.count_vertices_and_edges_wrap(root.left)
            self.count_vertices_and_edges_wrap(root.right)
        if root.left is not None and root.right is not None:
            root.num_edges = root.left.num_edges + root.right.num_edges + root.num_edges
        # print root.name, root.num_edges

    def compute_density(self, root):
        if root.left is None and root.right is None:
            return
        total_vertices = float(len(root.vertices))
        max_vertices = total_vertices * (total_vertices - 1) / 2
        root.density = root.num_edges / max_vertices
        self.compute_density(root.left)
        self.compute_density(root.right)

    def extract_sub_graph(self, root, min_density, result):

        if root is None:
            return
        if root.density > min_density:
            com = []
            for elem in list(root.vertices):
                com.append(elem)
            result.append(com)
        else:
            self.extract_sub_graph(root.left, min_density, result)
            self.extract_sub_graph(root.right, min_density, result)


def __make_set(r):
    r.parent = None
    r.vertices.add(r.name)


def __set_find(r):
    while r.parent != None:
        r = r.parent
    return r


# Building a new Node as Union of two sets
def __set_union(x, y):
    r = __Node("P" + str(x.name) + str(y.name))
    r.left = x
    r.right = y
    x.parent = r
    y.parent = r
    r.vertices = r.vertices.union(x.vertices, y.vertices)
    return r


def GDMP2(graph, min_threshold=0.75):

    A = nx.adjacency_matrix(graph)
    adj_matrix = A.todense()

    M = np.zeros(adj_matrix.shape)

    row, col = adj_matrix.shape

    # Building similarity function matrix, ie, Cosine Function matrix of all Column Vectors

    for x in range(0, row):
        for y in range(x, col):
            M[x][y] = 1 - scipy.spatial.distance.cosine(
                adj_matrix[:, x], adj_matrix[:, y]
            )

    tuples = []
    # On basis of zero graph
    min_value = 1 if min(graph.nodes()) > 0 else 0

    # Considering only non zero values
    for (x, y), value in np.ndenumerate(M):
        if value != 0 and x != y:
            tuples.append(((x + min_value, y + min_value), value))

    C = sorted(tuples, key=lambda x: x[1])
    t = np.count_nonzero(adj_matrix)
    C = C[-t:]
    ln = len(C)
    ln = ln - 1

    nodes = dict()
    root_nodes = set()
    tree = __Tree()

    for index in range(ln, -1, -1):
        vertices, value = C[index]
        i, j = vertices
        if nodes.__contains__(i) is False:
            a = __Node(i)
            __make_set(a)
            nodes[i] = a
        if nodes.__contains__(j) is False:
            a = __Node(j)
            __make_set(a)
            nodes[j] = a

        i = nodes[i]
        j = nodes[j]
        ri = __set_find(i)
        rj = __set_find(j)
        if ri.vertices != rj.vertices:
            temp_root = __set_union(ri, rj)
            root_nodes.add(temp_root)

    root_nodes = filter(lambda entry: entry.parent is None, list(root_nodes))

    for temp_roots in root_nodes:
        tree.root = temp_roots

        # Counting number of vertices and Edges
        tree.count_vertices_and_edges(graph.edges(), nodes)

        # Summing up number of edges of children to parent
        tree.count_vertices_and_edges_wrap(tree.root)

        # Computing density of Tree Nodes
        tree.compute_density(tree.root)

        # Filtering Nodes as Per Density Threshold
        communities = []
        tree.extract_sub_graph(tree.root, min_threshold, communities)
        return communities
