import community as louvain_modularity
import leidenalg
from collections import defaultdict
import networkx as nx
import igraph as ig
from nclib import NodeClustering
from nclib.utils import convert_graph_formats


def louvain(g, weight='weight', resolution=1., randomize=False):
    """
    Louvain  maximizes a modularity score for each community.
    The algorithm optimises the modularity in two elementary phases:
    (1) local moving of nodes;
    (2) aggregation of the network.
    In the local moving phase, individual nodes are moved to the community that yields the largest increase in the quality function.
    In the aggregation phase, an aggregate network is created based on the partition obtained in the local moving phase.
    Each community in this partition becomes a node in the aggregate network. The two phases are repeated until the quality function cannot be increased further.

    :param g: a networkx/igraph object
    :param weight: str, optional the key in graph to use as weight. Default to 'weight'
    :param resolution: double, optional  Will change the size of the communities, default to 1.
    :param randomize:  boolean, optional  Will randomize the node evaluation order and the community evaluation  order to get different partitions at each call
    :return: a list of communities


    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.louvain(G,'weight', 1.,False)

    :References:

    Blondel, Vincent D., et al. "Fast unfolding of communities in large networks." Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = louvain_modularity.best_partition(g, weight=weight, resolution=resolution, randomize=randomize)

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in coms.items():
        coms_to_node[c].append(n)

    coms_louvain = [tuple(c) for c in coms_to_node.values()]
    return NodeClustering(coms_louvain, g, "Louvain", method_parameters={"weight": weight, "resolution": resolution,
                                                                         "randomize": randomize})


def leiden(g, initial_membership=None, weights=None):
    """
    The Leiden algorithm is an improvement of the Louvain algorithm.
    The Leiden algorithm consists of three phases:
    (1) local moving of nodes,
    (2) refinement of the partition
    (3) aggregation of the network based on the refined partition, using the non-refined partition to create an initial partition for the aggregate network.

    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition.
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute.
    :return: a list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.leiden(G)

    :References:

    Traag, Vincent, Ludo Waltman, and Nees Jan van Eck. "From Louvain to Leiden: guaranteeing well-connected communities." arXiv preprint arXiv:1810.08473 (2018).
    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                    initial_membership=initial_membership, weights=weights
                                    )
    coms = [g.vs[x]['name'] for x in part]
    return NodeClustering(coms, g, "Leiden", method_parameters={"initial_membership": initial_membership,
                                                                "weights": weights})


def rb_pots(g, initial_membership=None, weights=None, resolution_parameter=1):
    """
    Rb_pots is a Leiden model where the quality function to optimize is:

    .. math:: Q = \\sum_{ij} \\left(A_{ij} - \\gamma \\frac{k_i k_j}{2m} \\right)\\delta(\\sigma_i, \\sigma_j)

    where :math:`A` is the adjacency matrix, :math:`k_i` is the (weighted) degree of node :math:`i`, :math:`m` is the total number of edges (or total edge weight), :math:`\\sigma_i` denotes the community of node :math:`i` and :math:`\\delta(\\sigma_i, \\sigma_j) = 1` if :math:`\\sigma_i = \\sigma_j` and `0` otherwise.
    For directed graphs a slightly different formulation is used, as proposed by Leicht and Newman :

    .. math:: Q = \\sum_{ij} \\left(A_{ij} - \\gamma \\frac{k_i^\mathrm{out} k_j^\mathrm{in}}{m} \\right)\\delta(\\sigma_i, \\sigma_j),

    where :math:`k_i^\\mathrm{out}` and :math:`k_i^\\mathrm{in}` refers to respectively the outdegree and indegree of node :math:`i` , and :math:`A_{ij}` refers to an edge from :math:`i` to :math:`j`.
    Note that this is the same of Leiden algorithm when setting :math:`\\gamma=1` and normalising by :math:`2m`, or :math:`m` for directed graphs.


    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition.
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute.
    :param resolution_parameter: double >0 A parameter value controlling the coarseness of the clustering. Higher resolutions lead to more communities, while lower resolutions lead to fewer communities.
    :return: a list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.rb_pots(G)

    :References:

    Reichardt, J., & Bornholdt, S. (2006).  Statistical mechanics of community detection.  Physical Review E, 74(1), 016110. 10.1103/PhysRevE.74.016110

    Leicht, E. A., & Newman, M. E. J. (2008).  Community Structure in Directed Networks.  Physical Review Letters, 100(11), 118703. 10.1103/PhysRevLett.100.118703

    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution_parameter,
                                    initial_membership=initial_membership, weights=weights)
    coms = [g.vs[x]['name'] for x in part]
    return NodeClustering(coms, g, "RB Pots", method_parameters={"initial_membership": initial_membership,
                                                                 "weights": weights,
                                                                 "resolution_parameter": resolution_parameter})


def rber_pots(g, initial_membership=None, weights=None, node_sizes=None, resolution_parameter=1):
    """
    rber_pots is a Leiden model where the quality function to optimize is:

    .. math:: Q = \\sum_{ij} \\left(A_{ij} - \\gamma p \\right)\\delta(\\sigma_i, \\sigma_j)

    where :math:`A` is the adjacency matrix,

    .. math:: p = \\frac{m}{\\binom{n}{2}}

    is the overall density of the graph, :math:`\\sigma_i` denotes the community of node :math:`i`, :math:`\\delta(\\sigma_i, \\sigma_j) = 1` if

    :math:`\\sigma_i = \\sigma_j` and `0` otherwise, and, finally :math:`\\gamma` is a resolution parameter.


    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition.
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute.
    :param node_sizes: list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed.
    :param resolution_parameter: double >0 A parameter value controlling the coarseness of the clustering. Higher resolutions lead to more communities, while lower resolutions lead to fewer communities.
    :return: a list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.rber_pots(G)

    :References:

    Reichardt, J., & Bornholdt, S. (2006).  Statistical mechanics of community detection.  Physical Review E, 74(1), 016110. 10.1103/PhysRevE.74.016110

    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.RBERVertexPartition,
                                    resolution_parameter=resolution_parameter,
                                    initial_membership=initial_membership, weights=weights,
                                    node_sizes=node_sizes,
                                    )
    coms = [g.vs[x]['name'] for x in part]
    return NodeClustering(coms, g, "RBER Pots", method_parameters={"initial_membership": initial_membership,
                                                                   "weights": weights, "node_sizes": node_sizes,
                                                                   "resolution_parameter": resolution_parameter})


def cpm(g, initial_membership=None, weights=None, node_sizes=None, resolution_parameter=1):
    """
    CPM is a Leiden model where the quality function to optimize is:

    .. math:: Q = \\sum_{ij} \\left(A_{ij} - \\gamma \\right)\\delta(\\sigma_i, \\sigma_j)

    where :math:`A` is the adjacency matrix, :math:`\\sigma_i` denotes the community of node :math:`i`, :math:`\\delta(\\sigma_i, \\sigma_j) = 1` if

    :math:`\\sigma_i = \\sigma_j` and `0` otherwise, and, finally :math:`\\gamma` is a resolution parameter.

     The internal density of communities

    .. math:: p_c = \\frac{m_c}{\\binom{n_c}{2}} \\geq \\gamma

    is higher than :math:`\\gamma`, while the external density

    .. math:: p_{cd} = \\frac{m_{cd}}{n_c n_d} \\leq \\gamma

    is lower than :math:`\\gamma`. In other words, choosing a particular
    :math:`\\gamma` corresponds to choosing to find communities of a particular
    density, and as such defines communities. Finally, the definition of a community is in a sense independent of the actual graph, which is not the case for any of the other methods.


    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition.
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute.
    :param node_sizes: list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed.
    :param resolution_parameter: double >0 A parameter value controlling the coarseness of the clustering. Higher resolutions lead to more communities, while lower resolutions lead to fewer communities.
    :return: a list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.cpm(G)

    :References:

    Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011).  Narrow scope for resolution-limit-free community detection. Physical Review E, 84(1), 016114. 10.1103/PhysRevE.84.016114

    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.CPMVertexPartition,
                                    resolution_parameter=resolution_parameter, initial_membership=initial_membership,
                                    weights=weights, node_sizes=node_sizes, )
    coms = [g.vs[x]['name'] for x in part]
    return NodeClustering(coms, g, "CPM", method_parameters={"initial_membership": initial_membership,
                                                             "weights": weights, "node_sizes": node_sizes,
                                                             "resolution_parameter": resolution_parameter})


def significance_communities(g, initial_membership=None, node_sizes=None):
    """
    Significance_communities is a Leiden model where the quality function to optimize is:

    .. math:: Q = \\sum_c \\binom{n_c}{2} D(p_c \\parallel p)

    where :math:`n_c` is the number of nodes in community :math:`c`,

    .. math:: p_c = \\frac{m_c}{\\binom{n_c}{2}}, is the density of community :math:`c`,  .. math:: p = \\frac{m}{\\binom{n}{2}}

    is the overall density of the graph, and finally  .. math:: D(x \\parallel y) = x \\ln \\frac{x}{y} + (1 - x) \\ln \\frac{1 - x}{1 - y} is the binary Kullback-Leibler divergence.
    For directed graphs simply multiply the binomials by 2. The expected Significance in Erdos-Renyi graphs behaves roughly as :math:`\\frac{1}{2} n \\ln n` for both directed and undirected graphs in this formulation.

    .. warning:: This method is not suitable for weighted graphs.


    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition.
    :param node_sizes: list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed.
    :return: a list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.significance_communities(G)

    :References:

    Traag, V. A., Krings, G., & Van Dooren, P. (2013). Significant scales in community structure.Scientific Reports, 3, 2930. `10.1038/srep02930 <http://doi.org/10.1038/srep02930>`
    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.SignificanceVertexPartition, initial_membership=initial_membership,
                                    node_sizes=node_sizes)
    coms = [g.vs[x]['name'] for x in part]
    return NodeClustering(coms, g, "Significance", method_parameters={"initial_membership": initial_membership,
                                                                      "node_sizes": node_sizes})


def surprise_communities(g, initial_membership=None, weights=None, node_sizes=None):
    """

    Surprise_communities is a Leiden model where the quality function to optimize is:

    .. math:: Q = m D(q \\parallel \\langle q \\rangle)

    where :math:`m` is the number of edges,  .. math:: q = \\frac{\\sum_c m_c}{m},  is the fraction of internal edges,

    .. math:: \\langle q \\rangle = \\frac{\\sum_c \\binom{n_c}{2}}{\\binom{n}{2}}

    is the expected fraction of internal edges, and finally

    .. math:: D(x \\parallel y) = x \\ln \\frac{x}{y} + (1 - x) \\ln \\frac{1 - x}{1 - y}

    is the binary Kullback-Leibler divergence.
    For directed graphs we can multiplying the binomials by 2, and this leaves :math:`\\langle q \\rangle` unchanged, so that we can simply use the same
    formulation.  For weighted graphs we can simply count the total internal weight instead of the total number of edges for :math:`q`, while :math:`\\langle q \\rangle` remains unchanged.

    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition.
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute.
    :param node_sizes: list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed.
    :return: a list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.surprise_communities(G)

    :References:

    Traag, V. A., Aldecoa, R., & Delvenne, J.-C. (2015).  Detecting communities using asymptotical surprise. Physical Review E, 92(2), 022816. 10.1103/PhysRevE.92.022816
    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.SurpriseVertexPartition, initial_membership=initial_membership,
                                    weights=weights, node_sizes=node_sizes)
    coms = [g.vs[x]['name'] for x in part]
    return NodeClustering(coms, g, "Surprise", method_parameters={"initial_membership": initial_membership,
                                                                  "weights": weights, "node_sizes": node_sizes})


def greedy_modularity(g, weight=None):
    """
    The CNM algorithm uses the modularity to find the communities strcutures.
    At every step of the algorithm two communities that contribute maximum positive value to global modularity are merged.

    :param g: a networkx/igraph object
    :param weight:
    :return: a list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.greedy_modularity(G)

    :References:

    Clauset, A., Newman, M. E., & Moore, C. “Finding community structure in very large networks.” Physical Review E 70(6), 2004
    """
    g = convert_graph_formats(g, nx.Graph)

    gc = nx.algorithms.community.greedy_modularity_communities(g, weight)
    coms = [tuple(x) for x in gc]
    return NodeClustering(coms, g, "Greedy Modularity", method_parameters={"weight": weight})
