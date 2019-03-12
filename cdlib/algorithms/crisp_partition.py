import infomap as imp
from wurlitzer import pipes
from cdlib.algorithms.internal import DER
import community as louvain_modularity
import leidenalg
from collections import defaultdict
from cdlib import NodeClustering, FuzzyNodeClustering
from cdlib.algorithms.internal.em import EM_nx
from cdlib.algorithms.internal.scan import SCAN_nx
from cdlib.algorithms.internal.GDMP2_nx import GDMP2
from cdlib.algorithms.internal.AGDL import Agdl
from cdlib.algorithms.internal.FuzzyCom import fuzzy_comm
import networkx as nx
import igraph as ig
from cdlib.utils import convert_graph_formats, nx_node_integer_mapping

__all__ = ["louvain", "leiden", "rb_pots", "rber_pots", "cpm", "significance_communities", "surprise_communities",
           "greedy_modularity", "der", "label_propagation", "async_fluid", "infomap", "walktrap", "girvan_newman", "em",
           "scan", "gdmp2", "spinglass", "eigenvector", "agdl", "frc_fgsn"]


def girvan_newman(g, level):
    """
    The Girvan–Newman algorithm detects communities by progressively removing edges from the original graph.
    The algorithm removes the "most valuable" edge, traditionally the edge with the highest betweenness centrality, at each step. As the graph breaks down into pieces, the tightly knit community structure is exposed and the result can be depicted as a dendrogram.

    :param g: a networkx/igraph object
    :param level: the level where to cut the dendrogram
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.girvan_newman(G, level=3)

    :References:

    Girvan, Michelle, and Mark EJ Newman. `Community structure in social and biological networks. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC122977/>`_ Proceedings of the national academy of sciences 99.12 (2002): 7821-7826.
    """

    g = convert_graph_formats(g, nx.Graph)

    gn_hierarchy = nx.algorithms.community.girvan_newman(g)
    coms = []
    for _ in range(level):
        coms = next(gn_hierarchy)

    communities = []

    for c in coms:
        communities.append(list(c))

    return NodeClustering(communities, g, "Girvan Newman", method_parameters={"level": level})


def em(g, k):
    """
    EM is based on based on a mixture model.
    The algorithm uses the expectation–maximization algorithm to detect structure in networks.

    :param g: a networkx/igraph object
    :param k: the number of desired communities
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.em(G, k=3)

    :References:

    Newman, Mark EJ, and Elizabeth A. Leicht. `Mixture community and exploratory analysis in networks.  <https://www.pnas.org/content/104/23/9564/>`_  Proceedings of the National Academy of Sciences 104.23 (2007): 9564-9569.
    """

    g = convert_graph_formats(g, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    algorithm = EM_nx(g, k)
    coms = algorithm.execute()

    if maps is not None:
        communities = []
        for c in coms:
            communities.append([maps[n] for n in c])
        nx.relabel_nodes(g, maps, False)
    else:
        communities = [list(c) for c in coms]

    return NodeClustering(communities, g, "EM", method_parameters={"k": k})


def scan(g, epsilon, mu):
    """
    SCAN (Structural Clustering Algorithm for Networks) is an algorithm which detects clusters, hubs and outliers in networks.
    It clusters vertices based on a structural similarity measure.
    The method uses the neighborhood of the vertices as clustering criteria instead of only their direct connections.
    Vertices are grouped into the clusters by how they share neighbors.

    :param g: a networkx/igraph object
    :param epsilon: the minimum threshold to assigning cluster membership
    :param mu: minimum number of neineighbors with a structural similarity that exceeds the threshold epsilon
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.scan(G, epsilon=0.7, mu=3)

    :References:

    Xu, X., Yuruk, N., Feng, Z., & Schweiger, T. A. (2007, August). `Scan: a structural clustering algorithm for networks. <http://www1.se.cuhk.edu.hk/~hcheng/seg5010/slides/p824-xu.pdf/>`_ In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 824-833)
    """

    g = convert_graph_formats(g, nx.Graph)

    algorithm = SCAN_nx(g, epsilon, mu)
    coms = algorithm.execute()
    return NodeClustering(coms, g, "SCAN", method_parameters={"epsilon": epsilon,
                                                              "mu": mu})


def gdmp2(g, min_threshold=0.75):
    """
    Gdmp2 is a method for identifying a set of dense subgraphs of a given sparse graph.
    It is inspired by an effective technique designed for a similar problem—matrix blocking, from a different discipline (solving linear systems).

    :param g: a networkx/igraph object
    :param min_threshold:  the minimum density threshold parameter to control the density of the output subgraphs, default 0.75
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.gdmp2(G)

    :References:

    Chen, Jie, and Yousef Saad. `Dense subgraph extraction with application to community detection. <https://ieeexplore.ieee.org/document/5677532/>`_ IEEE Transactions on Knowledge and Data Engineering 24.7 (2012): 1216-1230.

    .. note:: Reference implementation: https://github.com/imabhishekl/CSC591_Community_Detection
    """

    g = convert_graph_formats(g, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    coms = GDMP2(g, min_threshold)

    if maps is not None:
        communities = []
        for c in coms:
            communities.append([maps[n] for n in c])
        nx.relabel_nodes(g, maps, False)
    else:
        communities = coms

    return NodeClustering(communities, g, "GDMP2", method_parameters={"min_threshold": min_threshold})


def spinglass(g):
    """
    Spinglass relies on an analogy between a very popular statistical mechanic model called Potts spin glass, and the community structure.
    It applies the simulated annealing optimization technique on this model to optimize the modularity.

    :param g: a networkx/igraph object
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.spinglass(G)

    :References:

    Reichardt, Jörg, and Stefan Bornholdt. `Statistical mechanics of community detection. <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.74.016110/>`_ Physical Review E 74.1 (2006): 016110.
    """
    g = convert_graph_formats(g, ig.Graph)
    coms = g.community_spinglass()
    communities = []

    for c in coms:
        communities.append([g.vs[x]['name'] for x in c])

    return NodeClustering(communities, g, "Spinglass")


def eigenvector(g):
    """
    Newman's leading eigenvector method for detecting community structure based on modularity.
    This is the proper internal of the recursive, divisive algorithm: each split is done by maximizing the modularity regarding the original network.

    :param g: a networkx/igraph object
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.eigenvector(G)

    :References:

    Newman, Mark EJ. `Finding community structure in networks using the eigenvectors of matrices. <https://journals.aps.org/pre/pdf/10.1103/PhysRevE.74.036104/>`_ Physical review E 74.3 (2006): 036104.
    """

    g = convert_graph_formats(g, ig.Graph)
    coms = g.community_leading_eigenvector()

    communities = [g.vs[x]['name'] for x in coms]

    return NodeClustering(communities, g, "Eigenvector")


def agdl(g, number_communities, number_neighbors, kc, a):
    """
    AGDL is a graph-based agglomerative algorithm, for clustering high-dimensional data.
    The algorithm uses  the indegree and outdegree to characterize the affinity between two clusters.

    :param g: a networkx/igraph object
    :param number_communities: number of communities
    :param number_neighbors: Number of neighbors to use for KNN
    :param kc: size of the neighbor set for each cluster
    :param a: range(-infinity;+infinty). From the authors: a=np.arange(-2,2.1,0.5)
    :return: NodeClustering object

     :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.agdl(g, number_communities=3, number_neighbors=3, kc=4, a=1)

    :References:

    Zhang, W., Wang, X., Zhao, D., & Tang, X. (2012, October). `Graph degree linkage: Agglomerative clustering on a directed graph. <https://link.springer.com/chapter/10.1007/978-3-642-33718-5_31/>`_ In European Conference on Computer Vision (pp. 428-441). Springer, Berlin, Heidelberg.

    .. note:: Reference implementation: https://github.com/myungjoon/GDL
    """

    g = convert_graph_formats(g, nx.Graph)

    communities = Agdl(g, number_communities, number_neighbors, kc, a)
    nodes = {k: v for k, v in enumerate(g.nodes())}
    coms = []
    for com in communities:
        coms.append([nodes[n] for n in com])

    return NodeClustering(coms, g, "AGDL", method_parameters={"number_communities": number_communities,
                                                              "number_neighbors": number_neighbors,
                                                              "kc": kc, "a": a})


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
    :param randomize:  boolean, optional  Will randomize the node evaluation order and the community evaluation  order to get different partitions at each call, default False
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.louvain(G, weight='weight', resolution=1., randomize=False)

    :References:

    Blondel, Vincent D., et al. `Fast unfolding of communities in large networks. <https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008/meta/>`_ Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.

    .. note:: Reference implementation: https://github.com/taynaud/python-louvain
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = louvain_modularity.best_partition(g, weight=weight, resolution=resolution, randomize=randomize)

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in coms.items():
        coms_to_node[c].append(n)

    coms_louvain = [list(c) for c in coms_to_node.values()]
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
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.leiden(G)

    :References:

    Traag, Vincent, Ludo Waltman, and Nees Jan van Eck. `From Louvain to Leiden: guaranteeing well-connected communities. <https://arxiv.org/abs/1810.08473/>`_ arXiv preprint arXiv:1810.08473 (2018).

    .. note:: Reference implementation: https://github.com/vtraag/leidenalg
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
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
    :param resolution_parameter: double >0 A parameter value controlling the coarseness of the clustering. Higher resolutions lead to more communities, while lower resolutions lead to fewer communities. Default 1
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.rb_pots(G)

    :References:

    Reichardt, J., & Bornholdt, S. (2006).  `Statistical mechanics of community detection. <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.74.016110/>`_  Physical Review E, 74(1), 016110. 10.1103/PhysRevE.74.016110

    Leicht, E. A., & Newman, M. E. J. (2008).  `Community Structure in Directed Networks. <https://www.ncbi.nlm.nih.gov/pubmed/18517839/>`_  Physical Review Letters, 100(11), 118703. 10.1103/PhysRevLett.100.118703

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

    where :math:`A` is the adjacency matrix,  :math:`p = \\frac{m}{\\binom{n}{2}}` is the overall density of the graph, :math:`\\sigma_i` denotes the community of node :math:`i`, :math:`\\delta(\\sigma_i, \\sigma_j) = 1` if  :math:`\\sigma_i = \\sigma_j` and `0` otherwise, and, finally :math:`\\gamma` is a resolution parameter.


    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
    :param node_sizes: list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed. Deafault None
    :param resolution_parameter: double >0 A parameter value controlling the coarseness of the clustering. Higher resolutions lead to more communities, while lower resolutions lead to fewer communities. Deafault 1
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.rber_pots(G)

    :References:

    Reichardt, J., & Bornholdt, S. (2006).  `Statistical mechanics of community detection. <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.74.016110/>`_  Physical Review E, 74(1), 016110. 10.1103/PhysRevE.74.016110


    .. note:: Reference implementation: https://github.com/vtraag/leidenalg

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

    where :math:`A` is the adjacency matrix, :math:`\\sigma_i` denotes the community of node :math:`i`, :math:`\\delta(\\sigma_i, \\sigma_j) = 1` if  :math:`\\sigma_i = \\sigma_j` and `0` otherwise, and, finally :math:`\\gamma` is a resolution parameter.

    The internal density of communities

    .. math:: p_c = \\frac{m_c}{\\binom{n_c}{2}} \\geq \\gamma

    is higher than :math:`\\gamma`, while the external density

    :math:`p_{cd} = \\frac{m_{cd}}{n_c n_d} \\leq \\gamma`    is lower than :math:`\\gamma`. In other words, choosing a particular
    :math:`\\gamma` corresponds to choosing to find communities of a particular
    density, and as such defines communities. Finally, the definition of a community is in a sense independent of the actual graph, which is not the case for any of the other methods.


    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
    :param node_sizes: list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed. Deafault None
    :param resolution_parameter: double >0 A parameter value controlling the coarseness of the clustering. Higher resolutions lead to more communities, while lower resolutions lead to fewer communities. Deafault 1
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.cpm(G)

    :References:

    Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011).  `Narrow scope for resolution-limit-free community detection. <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.016114/>`_ Physical Review E, 84(1), 016114. 10.1103/PhysRevE.84.016114


    .. note:: Reference implementation: https://github.com/vtraag/leidenalg

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

    where :math:`n_c` is the number of nodes in community :math:`c`, :math:`p_c = \\frac{m_c}{\\binom{n_c}{2}}`, is the density of community :math:`c`,  :math:`p = \\frac{m}{\\binom{n}{2}}`  is the overall density of the graph, and finally  :math:`D(x \\parallel y) = x \\ln \\frac{x}{y} + (1 - x) \\ln \\frac{1 - x}{1 - y}` is the binary Kullback-Leibler divergence.
    For directed graphs simply multiply the binomials by 2. The expected Significance in Erdos-Renyi graphs behaves roughly as :math:`\\frac{1}{2} n \\ln n` for both directed and undirected graphs in this formulation.

    .. warning:: This method is not suitable for weighted graphs.


    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
    :param node_sizes: list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed. Deafault None
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.significance_communities(G)

    :References:

    Traag, V. A., Krings, G., & Van Dooren, P. (2013). `Significant scales in community structure. <https://www.nature.com/articles/srep02930/>`_  Scientific Reports, 3, 2930. `10.1038/srep02930 <http://doi.org/10.1038/srep02930>`

    .. note:: Reference implementation: https://github.com/vtraag/leidenalg

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

    where :math:`m` is the number of edges,  :math:`q = \\frac{\\sum_c m_c}{m}`,  is the fraction of internal edges, :math:`\\langle q \\rangle = \\frac{\\sum_c \\binom{n_c}{2}}{\\binom{n}{2}}` is the expected fraction of internal edges, and finally

    :math:`D(x \\parallel y) = x \\ln \\frac{x}{y} + (1 - x) \\ln \\frac{1 - x}{1 - y}`  is the binary Kullback-Leibler divergence.

    For directed graphs we can multiplying the binomials by 2, and this leaves :math:`\\langle q \\rangle` unchanged, so that we can simply use the same
    formulation.  For weighted graphs we can simply count the total internal weight instead of the total number of edges for :math:`q` , while :math:`\\langle q \\rangle` remains unchanged.

    :param g: a networkx/igraph object
    :param initial_membership:  list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
    :param weights: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
    :param node_sizes: list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed. Deafault None
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.surprise_communities(G)

    :References:

    Traag, V. A., Aldecoa, R., & Delvenne, J.-C. (2015).  `Detecting communities using asymptotical surprise. <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.022816/>`_ Physical Review E, 92(2), 022816. 10.1103/PhysRevE.92.022816

    .. note:: Reference implementation: https://github.com/vtraag/leidenalg

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
    :param weight: list of double, or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.greedy_modularity(G)

    :References:

    Clauset, A., Newman, M. E., & Moore, C. `Finding community structure in very large networks. <http://ece-research.unm.edu/ifis/papers/community-moore.pdf/>`_ Physical Review E 70(6), 2004
    """
    g = convert_graph_formats(g, nx.Graph)

    coms = nx.algorithms.community.greedy_modularity_communities(g, weight)
    coms = [list(x) for x in coms]
    return NodeClustering(coms, g, "Greedy Modularity", method_parameters={"weight": weight})


def infomap(g):
    """
    Infomap is based on ideas of information theory.
    The algorithm uses the probability flow of random walks on a network as a proxy for information flows in the real system and it decomposes the network into modules by compressing a description of the probability flow.

    :param g: a networkx/igraph object
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.infomap(G)

    :References:

    Rosvall M, Bergstrom CT (2008) `Maps of random walks on complex networks reveal community structure. <https://www.pnas.org/content/105/4/1118/>`_ Proc Natl Acad SciUSA 105(4):1118–1123

    .. note:: Reference implementation: https://pypi.org/project/infomap/
    """
    g = convert_graph_formats(g, nx.Graph)

    g1 = nx.convert_node_labels_to_integers(g, label_attribute="name")
    name_map = nx.get_node_attributes(g1, 'name')
    coms_to_node = defaultdict(list)

    with pipes():
        im = imp.Infomap()
        network = im.network()
        for e in g1.edges():
            network.addLink(e[0], e[1])
        im.run()

        for node in im.iterTree():
            if node.isLeaf():
                nid = node.physicalId
                module = node.moduleIndex()
                nm = name_map[nid]
                coms_to_node[module].append(nm)

    coms_infomap = [list(c) for c in coms_to_node.values()]
    return NodeClustering(coms_infomap, g, "Infomap")


def walktrap(g):
    """
    walktrap is an approach based on random walks.
    The general idea is that if you perform random walks on the graph, then the walks are more likely to stay within the same community because there are only a few edges that lead outside a given community. Walktrap runs short random walks and uses the results of these random walks to merge separate communities in a bottom-up manner.

    :param g: a networkx/igraph object
    :return: NodeClusterint object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.walktrap(G)

    :References:

    Pons, Pascal, and Matthieu Latapy. `Computing communities in large networks using random walks. <http://jgaa.info/accepted/2006/PonsLatapy2006.10.2.pdf/>`_ J. Graph Algorithms Appl. 10.2 (2006): 191-218.
    """
    g = convert_graph_formats(g, ig.Graph)
    coms = g.community_walktrap().as_clustering()
    communities = []

    for c in coms:
        communities.append([g.vs[x]['name'] for x in c])

    return NodeClustering(communities, g, "Walktrap")


def label_propagation(g):
    """
    The Label Propagation algorithm (LPA) detects communities using network structure alone.
    The algorithm doesn’t require a pre-defined objective function or prior information about the communities.
    It works as follows:
    -Every node is initialized with a unique label (an identifier)
    -These labels propagate through the network
    -At every iteration of propagation, each node updates its label to the one that the maximum numbers of its neighbours belongs to. Ties are broken uniformly and randomly.
    -LPA reaches convergence when each node has the majority label of its neighbours.

    :param g: a networkx/igraph object
    :return: EdgeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.label_propagation(G)

    :References:

    Raghavan, U. N., Albert, R., & Kumara, S. (2007). `Near linear time algorithm to detect community structures in large-scale networks. <http://www.leonidzhukov.net/hse/2017/networks/papers/raghavan2007.pdf/>`_ Physical review E, 76(3), 036106.
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = list(nx.algorithms.community.label_propagation_communities(g))
    coms = [list(x) for x in coms]

    return NodeClustering(coms, g, "Label Propagation")


def async_fluid(g, k):
    """
    Fluid Communities (FluidC) is based on the simple idea of fluids (i.e., communities) interacting in an environment (i.e., a non-complete graph), expanding and contracting.
    It is propagation-based algorithm and it allows to specify the number of desired communities (k) and it is asynchronous, where each vertex update is computed using the latest partial state of the graph.


    :param g: a networkx/igraph object
    :param k: Number of communities to search
    :return: EdgeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.async_fluid(G,k=2)


    :References:

    Ferran Parés, Dario Garcia-Gasulla, Armand Vilalta, Jonatan Moreno, Eduard Ayguadé, Jesús Labarta, Ulises Cortés, Toyotaro Suzumura T. `Fluid Communities: A Competitive and Highly Scalable Community Detection Algorithm. <https://link.springer.com/chapter/10.1007/978-3-319-72150-7_19/>`_
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = nx.algorithms.community.asyn_fluidc(g, k)
    coms = [list(x) for x in coms]
    return NodeClustering(coms, g, "Fluid")


def der(graph, walk_len=3, threshold=.00001, iter_bound=50):
    """
    DER is a Diffusion Entropy Reducer graph clustering algorithm.
    The algorithm uses random walks to embed the graph in a space of measures, after which a modification of k-means in that space is applied. It creates the walks, creates an initialization, runs the algorithm,
    and finally extracts the communities.

    :param graph: an undirected networkx graph object
    :param walk_len: length of the random walk, default 3
    :param threshold: threshold for stop criteria; if the likelihood_diff is less than threshold tha algorithm stops, default 0.00001
    :param iter_bound: maximum number of iteration, default 50
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.der(G, 3, .00001, 50)


    :References:

    M. Kozdoba and S. Mannor, `Community Detection via Measure Space Embedding <https://papers.nips.cc/paper/5808-community-detection-via-measure-space-embedding/>`_, NIPS 2015

    .. note:: Reference implementation: https://github.com/komarkdev/der_graph_clustering
    """

    graph = convert_graph_formats(graph, nx.Graph)

    communities, _ = DER.der_graph_clustering(graph, walk_len=walk_len,
                                              alg_threshold=threshold, alg_iterbound=iter_bound)

    maps = {k: v for k, v in enumerate(graph.nodes())}
    coms = []
    for c in communities:
        coms.append([maps[n] for n in c])

    return NodeClustering(coms, graph, "DER", method_parameters={"walk_len": walk_len, "threshold": threshold,
                                                                 "iter_bound": iter_bound})


def frc_fgsn(graph, theta, eps, r):
    """Fuzzy-Rough Community Detection on Fuzzy Granular model of Social Network.

    FRC-FGSN assigns nodes to communities specifying the probability of each association.
    The flattened partition ensure that each node is associated to the community that maximize such association probability.
    FRC-FGSN may generate orphan nodes (i.e., nodes not assigned to any community).

    :param graph: networkx/igraph object
    :param theta: community density coefficient
    :param eps: coupling coefficient of the community. Ranges in [0, 1], small values ensure that only strongly connected node granules are merged togheter.
    :param r: radius of the granule (int)
    :return: FuzzyNodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = frc_fgsn(G, theta=1, eps=0.5, r=3)


    :References:

    Kundu, S., & Pal, S. K. (2015). `Fuzzy-rough community in social networks. <https://www.sciencedirect.com/science/article/pii/S0167865515000537/>`_ Pattern Recognition Letters, 67, 145-152.

    .. note:: Reference implementation: https://github.com/nidhisridhar/Fuzzy-Community-Detection
    """

    graph = convert_graph_formats(graph, nx.Graph)
    g, maps = nx_node_integer_mapping(graph)

    communities, fuzz_assoc = fuzzy_comm(graph, theta, eps, r)

    if maps is not None:
        coms = []
        for c in communities:
            coms.append([tuple(maps[n]) for n in c])

        nx.relabel_nodes(g, maps, False)
        fuzz_assoc = {maps[nid]: v for nid, v in fuzz_assoc.items()}
    else:
        coms = [list(c) for c in communities]

    return FuzzyNodeClustering(coms, fuzz_assoc, graph, "FuzzyComm", method_parameters={"theta": theta,
                                                                                        "eps": eps, "r": r})
