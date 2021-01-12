from cdlib import BiNodeClustering

import networkx as nx

try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None

try:
    import leidenalg
except ModuleNotFoundError:
    leidenalg = None

from cdlib.utils import convert_graph_formats
from collections import defaultdict

__all__ = ['bimlpa', 'CPM_Bipartite']


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

    return BiNodeClustering(top_coms, bottom_coms, g_original, "BiMLPA",
                            method_parameters={"theta": theta, "lambd": lambd})


def CPM_Bipartite(g_original, resolution_parameter_01,
                  resolution_parameter_0=0, resolution_parameter_1=0, degree_as_node_size=False, seed=0):
    """
    CPM_Bipartite is the extension of CPM to bipartite graphs

    :param g_original: a networkx/igraph object
    :param resolution_parameter_01: Resolution parameter for in between two classes.
    :param resolution_parameter_0: Resolution parameter for class 0.
    :param resolution_parameter_1: Resolution parameter for class 1.
    :param degree_as_node_size: If ``True`` use degree as node size instead of 1, to mimic modularity
    :param seed: the random seed to be used in CPM method to keep results/partitions replicable
    :return: BiNodeClustering object (

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.algorithms.bipartite.generators.random_graph(100, 20, 0.1)
    >>> coms = algorithms.CPM_Bipartite(G, 1)

    :References:

    Barber, M. J. (2007). Modularity and community detection in bipartite networks. Physical Review E, 76(6), 066102. 10.1103/PhysRevE.76.066102

    .. note:: Reference implementation: https://leidenalg.readthedocs.io/en/stable/multiplex.html?highlight=bipartite#bipartite
    """
    if ig is None or leidenalg is None:
        raise ModuleNotFoundError("Optional dependency not satisfied: install igraph and leidenalg to use the "
                                  "selected feature.")

    g = convert_graph_formats(g_original, ig.Graph)

    try:
        g.vs['name']
    except:
        g.vs['name'] = [v.index for v in g.vs]

    optimiser = leidenalg.Optimiser()
    leidenalg.Optimiser.set_rng_seed(self=optimiser, value=seed)

    p_01, p_0, p_1 = leidenalg.CPMVertexPartition.Bipartite(g, resolution_parameter_01=resolution_parameter_01,
                                                            resolution_parameter_0=resolution_parameter_0,
                                                            resolution_parameter_1=resolution_parameter_1,
                                                            degree_as_node_size=degree_as_node_size)
    optimiser.optimise_partition_multiplex([p_01, p_0, p_1], layer_weights=[1, -1, -1])

    coms = defaultdict(list)
    for n in g.vs:
        coms[p_01.membership[n.index]].append(n.index)

    return BiNodeClustering(list(coms.values()), [], g_original, "CPM_Bipartite",
                            method_parameters={"resolution_parameter_0": resolution_parameter_01,
                                               "resolution_parameter_0": resolution_parameter_0,
                                               "resolution_parameter_1": resolution_parameter_1,
                                               "degree_as_node_size": degree_as_node_size, "seed": seed})
