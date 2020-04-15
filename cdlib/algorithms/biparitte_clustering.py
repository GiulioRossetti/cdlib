from cdlib import BiNodeClustering

import networkx as nx
from cdlib.utils import convert_graph_formats


__all__ = ['bimlpa']


def bimlpa(g_original, theta=0.3, lambd=7):
    """
    BiMLPA is designed to detect the many-to-many correspondence community in bipartite networks using multi-label propagation algorithm.

    :param g_original: a networkx/igraph object
    :param theta: Label weights threshold. Default 0.3.
    :param lambd: The max number of labels. Default 7.
    :return: BiNodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.algorithms.bipartite.generators.random_graph(100, 20, 0.1)
    >>> coms = algorithms.bimlpa(G)

    :References:

    Taguchi, Hibiki, Tsuyoshi Murata, and Xin Liu. "BiMLPA: Community Detection in Bipartite Networks by Multi-Label Propagation." International Conference on Network Science. Springer, Cham, 2020.

    .. note:: Reference implementation: https://github.com/hbkt/BiMLPA
    """
    from BiMLPA import BiMLPA_SqrtDeg, relabeling, output_community

    g = convert_graph_formats(g_original, nx.Graph)

    bimlpa = BiMLPA_SqrtDeg(g, theta, lambd)
    bimlpa.start()
    relabeling(g)
    top_coms, bottom_coms = output_community(g)

    return BiNodeClustering(top_coms, bottom_coms, g_original, "BiMLPA", method_parameters={"theta": theta, "lambd": lambd})
