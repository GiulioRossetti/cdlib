from nclib.community.algorithms import LEMON
import networkx as nx
from nclib import NodeClustering
from nclib.utils import convert_graph_formats
import numpy as np


def lemon(graph, seeds, min_com_size=20, max_com_size=50, expand_step=6, subspace_dim=3, walk_steps=3, biased=False):
    """

    :param graph:
    :param seeds:
    :param min_com_size: the minimum size of a single community in the network
    :param max_com_size: the maximum size of a single community in the network
    :param expand_step: the step of seed set increasement during expansion process
    :param subspace_dim:
    :param walk_steps:
    :param biased:
    :return:
    """

    graph = convert_graph_formats(graph, nx.Graph)
    graph_m = nx.convert_matrix.to_numpy_array(graph)

    node_to_pos = {n: p for p, n in enumerate(graph.nodes())}
    pos_to_node = {p: n for n, p in node_to_pos.items()}

    seeds = np.array([node_to_pos[s] for s in seeds])

    community = LEMON.lemon(graph_m, seeds, min_com_size, max_com_size, expand_step,
                            subspace_dim=subspace_dim, walk_steps=walk_steps, biased=biased)

    return NodeClustering([pos_to_node[n] for n in community], graph, "LEMON", method_parameters={"seeds": seeds, "min_com_size": min_com_size,
                                                                          "max_com_size": max_com_size,
                                                                          "expand_step": expand_step,
                                                                          "subspace_dim": subspace_dim,
                                                                          "walk_steps": walk_steps,
                                                                          "biased": biased})
