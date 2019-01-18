from nclib.community.algorithms import LEMON
import networkx as nx
from nclib import NodeClustering
from nclib.utils import convert_graph_formats
import numpy as np

__all__ = ["lemon"]


def lemon(graph, seeds, min_com_size=20, max_com_size=50, expand_step=6, subspace_dim=3, walk_steps=3, biased=False):
    """
    Lemon is a large scale overlapping community detection method based on local expansion via minimum one norm.

    The algorithm adopts a local expansion method in order to identify the community members from a few exemplary seed members.
    The algorithm finds the community by seeking a sparse vector in the span of the local spectra such that the seeds are in its support. LEMON can achieve the highest detection accuracy among state-of-the-art proposals. The running time depends on the size of the community rather than that of the entire graph.

    :param graph: a networkx/igraph object
    :param seeds: Node list
    :param min_com_size: the minimum size of a single community in the network, default 20
    :param max_com_size: the maximum size of a single community in the network, default 50
    :param expand_step: the step of seed set increasement during expansion process, default 6
    :param subspace_dim: dimension of the subspace; choosing a large dimension is undesirable because it would increase the computation cost of generating local spectra default 3
    :param walk_steps: the number of step for the random walk, default 3
    :param biased: boolean; set if the random walk starting from seed nodes, default False
    :return: a list of overlapping communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> seeds = ["$0$", "$2$", "$3$"]
    >>> coms = community.lemon(G, seeds, min_com_size=2, max_com_size=5)

    :References:

    Yixuan Li, Kun He, David Bindel, John Hopcroft **Uncovering the small community structure in large networks: A local spectral approach.** Proceedings of the 24th international conference on world wide web. International World Wide Web Conferences Steering Committee, 2015.

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
