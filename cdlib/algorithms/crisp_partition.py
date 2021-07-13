try:
    import infomap as imp
except ModuleNotFoundError:
    imp = None

try:
    from wurlitzer import pipes
except ModuleNotFoundError:
    pipes = None

try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None

try:
    import leidenalg
except ModuleNotFoundError:
    leidenalg = None

try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    gt = None

import warnings
import numpy as np
from typing import Callable
from copy import deepcopy
from cdlib.algorithms.internal import DER
import community as louvain_modularity
import warnings
from collections import defaultdict
from cdlib import NodeClustering, FuzzyNodeClustering
from cdlib.algorithms.internal.belief_prop import detect_belief_communities
from cdlib.algorithms.internal.em import EM_nx
from cdlib.algorithms.internal.scan import SCAN_nx
from cdlib.algorithms.internal.GDMP2_nx import GDMP2
from cdlib.algorithms.internal.AGDL import Agdl
from cdlib.algorithms.internal.FuzzyCom import fuzzy_comm
from cdlib.algorithms.internal.Markov import markov
from cdlib.algorithms.internal.ga import ga_community_detection
from cdlib.algorithms.internal.SiblinarityAntichain import (
    matrix_node_recursive_antichain_partition,
)
from cdlib.algorithms.internal.LSWL import (
    LSWLCommunityDiscovery_offline,
    LSWLPlusCommunityDetection,
    LSWLCommunityDiscovery,
)
from cdlib.algorithms.internal.spectralCD import spectral_communities
from cdlib.algorithms.internal.modularity_m import ModularityMCommunityDiscovery
from cdlib.algorithms.internal.modularity_r import ModularityRCommunityDiscovery
from cdlib.algorithms.internal.headtail import HeadTail
from cdlib.algorithms.internal.Kcut import kcut_exec
from cdlib.algorithms.internal.paris import paris as paris_alg, paris_best_clustering
from cdlib.algorithms.internal.principled import principled

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
except ModuleNotFoundError:
    OllivierRicci = None

try:
    import pycombo as pycombo_part
except ModuleNotFoundError:
    pycombo_part = None

from karateclub import EdMot, GEMSEC, SCD
import markov_clustering as mc
from chinese_whispers import chinese_whispers as cw
from chinese_whispers import aggregate_clusters
from thresholdclustering.thresholdclustering import best_partition as th_best_partition

import networkx as nx

from cdlib.utils import (
    convert_graph_formats,
    __from_nx_to_graph_tool,
    affiliations2nodesets,
    nx_node_integer_mapping,
)

__all__ = [
    "louvain",
    "leiden",
    "rb_pots",
    "rber_pots",
    "cpm",
    "significance_communities",
    "surprise_communities",
    "greedy_modularity",
    "der",
    "label_propagation",
    "async_fluid",
    "infomap",
    "walktrap",
    "girvan_newman",
    "em",
    "scan",
    "gdmp2",
    "spinglass",
    "eigenvector",
    "agdl",
    "frc_fgsn",
    "sbm_dl",
    "sbm_dl_nested",
    "markov_clustering",
    "edmot",
    "chinesewhispers",
    "siblinarity_antichain",
    "ga",
    "belief",
    "threshold_clustering",
    "lswl_plus",
    "lswl",
    "mod_m",
    "mod_r",
    "head_tail",
    "kcut",
    "gemsec",
    "scd",
    "pycombo",
    "paris",
    "principled_clustering",
    "ricci_community",
    "spectral",
]


def girvan_newman(g_original: object, level: int) -> NodeClustering:
    """
    The Girvan–Newman algorithm detects communities by progressively removing edges from the original graph.
    The algorithm removes the "most valuable" edge, traditionally the edge with the highest betweenness centrality, at each step. As the graph breaks down into pieces, the tightly knit community structure is exposed and the result can be depicted as a dendrogram.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
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

    g = convert_graph_formats(g_original, nx.Graph)

    gn_hierarchy = nx.algorithms.community.girvan_newman(g)
    coms = []
    for _ in range(level):
        coms = next(gn_hierarchy)

    communities = []

    for c in coms:
        communities.append(list(c))

    return NodeClustering(
        communities, g_original, "Girvan Newman", method_parameters={"level": level}
    )


def em(g_original: object, k: int) -> NodeClustering:
    """
    EM is based on based on a mixture model.
    The algorithm uses the expectation–maximization algorithm to detect structure in networks.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        Yes      No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param k: the number of desired communities
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.em(G, k=3)

    :References:

    Newman, Mark EJ, and Elizabeth A. Leicht. `Mixture community and exploratory analysis in networks.  <https://www.pnas.org/content/104/23/9564/>`_  Proceedings of the National Academy of Sciences 104.23 (2007): 9564-9569.

    .. note:: Reference implementation: https://github.com/duckneo/CommunityDetection
    """

    g = convert_graph_formats(g_original, nx.Graph)
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

    return NodeClustering(communities, g_original, "EM", method_parameters={"k": k})


def scan(g_original: object, epsilon: float, mu: int) -> NodeClustering:
    """
    SCAN (Structural Clustering Algorithm for Networks) is an algorithm which detects clusters, hubs and outliers in networks.
    It clusters vertices based on a structural similarity measure.
    The method uses the neighborhood of the vertices as clustering criteria instead of only their direct connections.
    Vertices are grouped into the clusters by how they share neighbors.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
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

    g = convert_graph_formats(g_original, nx.Graph)

    algorithm = SCAN_nx(g, epsilon, mu)
    coms = algorithm.execute()
    return NodeClustering(
        coms, g_original, "SCAN", method_parameters={"epsilon": epsilon, "mu": mu}
    )


def gdmp2(g_original: object, min_threshold: float = 0.75) -> NodeClustering:
    """
    Gdmp2 is a method for identifying a set of dense subgraphs of a given sparse graph.
    It is inspired by an effective technique designed for a similar problem—matrix blocking, from a different discipline (solving linear systems).


    **Supported Graph Types**

    ========== ======== ======== =========
    Undirected Directed Weighted Bipartite
    ========== ======== ======== =========
    Yes        Yes      No       Yes
    ========== ======== ======== =========

    :param g_original: a networkx/igraph object
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

    g = convert_graph_formats(g_original, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    coms = GDMP2(g, min_threshold)

    if maps is not None:
        communities = []
        for c in coms:
            communities.append([maps[n] for n in c])
        nx.relabel_nodes(g, maps, False)
    else:
        communities = coms

    return NodeClustering(
        communities,
        g_original,
        "GDMP2",
        method_parameters={"min_threshold": min_threshold},
    )


def spinglass(g_original: object) -> NodeClustering:
    """
    Spinglass relies on an analogy between a very popular statistical mechanic model called Potts spin glass, and the community structure.
    It applies the simulated annealing optimization technique on this model to optimize the modularity.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.spinglass(G)

    :References:

    Reichardt, Jörg, and Stefan Bornholdt. `Statistical mechanics of community detection. <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.74.016110/>`_ Physical Review E 74.1 (2006): 016110.
    """
    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)
    coms = g.community_spinglass()
    communities = []

    for c in coms:
        communities.append([g.vs[x]["name"] for x in c])

    return NodeClustering(
        communities, g_original, "Spinglass", method_parameters={"": ""}
    )


def eigenvector(g_original: object) -> NodeClustering:
    """
    Newman's leading eigenvector method for detecting community structure based on modularity.
    This is the proper internal of the recursive, divisive algorithm: each split is done by maximizing the modularity regarding the original network.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.eigenvector(G)

    :References:

    Newman, Mark EJ. `Finding community structure in networks using the eigenvectors of matrices. <https://journals.aps.org/pre/pdf/10.1103/PhysRevE.74.036104/>`_ Physical review E 74.3 (2006): 036104.
    """

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)
    coms = g.community_leading_eigenvector()

    communities = [g.vs[x]["name"] for x in coms]

    return NodeClustering(
        communities, g_original, "Eigenvector", method_parameters={"": ""}
    )


def agdl(g_original: object, number_communities: int, kc: int) -> NodeClustering:
    """
    AGDL is a graph-based agglomerative algorithm, for clustering high-dimensional data.
    The algorithm uses  the indegree and outdegree to characterize the affinity between two clusters.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        Yes      Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param number_communities: number of communities
    :param kc: size of the neighbor set for each cluster
    :return: NodeClustering object

     :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.agdl(g, number_communities=3, kc=4)

    :References:

    Zhang, W., Wang, X., Zhao, D., & Tang, X. (2012, October). `Graph degree linkage: Agglomerative clustering on a directed graph. <https://link.springer.com/chapter/10.1007/978-3-642-33718-5_31/>`_ In European Conference on Computer Vision (pp. 428-441). Springer, Berlin, Heidelberg.

    .. note:: Reference implementation: https://github.com/myungjoon/GDL
    """

    g = convert_graph_formats(g_original, nx.Graph)

    communities = Agdl(g, number_communities, kc)
    nodes = {k: v for k, v in enumerate(g.nodes())}
    coms = []
    for com in communities:
        coms.append([nodes[n] for n in com])

    return NodeClustering(
        coms,
        g_original,
        "AGDL",
        method_parameters={"number_communities": number_communities, "kc": kc},
    )


def louvain(
    g_original: object,
    weight: str = "weight",
    resolution: float = 1.0,
    randomize: int = None,
) -> NodeClustering:
    """
    Louvain  maximizes a modularity score for each community.
    The algorithm optimises the modularity in two elementary phases:
    (1) local moving of nodes;
    (2) aggregation of the network.
    In the local moving phase, individual nodes are moved to the community that yields the largest increase in the quality function.
    In the aggregation phase, an aggregate network is created based on the partition obtained in the local moving phase.
    Each community in this partition becomes a node in the aggregate network. The two phases are repeated until the quality function cannot be increased further.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param weight: str, optional the key in graph to use as weight. Default to 'weight'
    :param resolution: double, optional  Will change the size of the communities, default to 1.
    :param randomize: int, RandomState instance or None, optional (default=None). If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.louvain(G, weight='weight', resolution=1.)

    :References:

    Blondel, Vincent D., et al. `Fast unfolding of communities in large networks. <https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008/meta/>`_ Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.

    .. note:: Reference implementation: https://github.com/taynaud/python-louvain
    """

    g = convert_graph_formats(g_original, nx.Graph)

    coms = louvain_modularity.best_partition(
        g, weight=weight, resolution=resolution, randomize=randomize
    )

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in coms.items():
        coms_to_node[c].append(n)

    coms_louvain = [list(c) for c in coms_to_node.values()]
    return NodeClustering(
        coms_louvain,
        g_original,
        "Louvain",
        method_parameters={
            "weight": weight,
            "resolution": resolution,
            "randomize": randomize,
        },
    )


def leiden(
    g_original: object, initial_membership: list = None, weights: list = None
) -> NodeClustering:
    """
    The Leiden algorithm is an improvement of the Louvain algorithm.
    The Leiden algorithm consists of three phases:
    (1) local moving of nodes,
    (2) refinement of the partition
    (3) aggregation of the network based on the refined partition, using the non-refined partition to create an initial partition for the aggregate network.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
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

    if ig is None or leidenalg is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph and leidenalg to use the "
            "selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)

    part = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        initial_membership=initial_membership,
        weights=weights,
    )
    coms = [g.vs[x]["name"] for x in part]
    return NodeClustering(
        coms,
        g_original,
        "Leiden",
        method_parameters={
            "initial_membership": initial_membership,
            "weights": weights,
        },
    )


def rb_pots(
    g_original: object,
    initial_membership: list = None,
    weights: list = None,
    resolution_parameter: float = 1,
) -> NodeClustering:
    """
    Rb_pots is a model where the quality function to optimize is:

    .. math:: Q = \\sum_{ij} \\left(A_{ij} - \\gamma \\frac{k_i k_j}{2m} \\right)\\delta(\\sigma_i, \\sigma_j)

    where :math:`A` is the adjacency matrix, :math:`k_i` is the (weighted) degree of node :math:`i`, :math:`m` is the total number of edges (or total edge weight), :math:`\\sigma_i` denotes the community of node :math:`i` and :math:`\\delta(\\sigma_i, \\sigma_j) = 1` if :math:`\\sigma_i = \\sigma_j` and `0` otherwise.
    For directed graphs a slightly different formulation is used, as proposed by Leicht and Newman :

    .. math:: Q = \\sum_{ij} \\left(A_{ij} - \\gamma \\frac{k_i^\mathrm{out} k_j^\mathrm{in}}{m} \\right)\\delta(\\sigma_i, \\sigma_j),

    where :math:`k_i^\\mathrm{out}` and :math:`k_i^\\mathrm{in}` refers to respectively the outdegree and indegree of node :math:`i` , and :math:`A_{ij}` refers to an edge from :math:`i` to :math:`j`.
    Note that this is the same of Leiden algorithm when setting :math:`\\gamma=1` and normalising by :math:`2m`, or :math:`m` for directed graphs.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        Yes      Yes
    ========== ======== ========


    :param g_original: a networkx/igraph object
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

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution_parameter,
        initial_membership=initial_membership,
        weights=weights,
    )
    coms = [g.vs[x]["name"] for x in part]
    return NodeClustering(
        coms,
        g_original,
        "RB Pots",
        method_parameters={
            "initial_membership": initial_membership,
            "weights": weights,
            "resolution_parameter": resolution_parameter,
        },
    )


def rber_pots(
    g_original: object,
    initial_membership: list = None,
    weights: list = None,
    node_sizes: list = None,
    resolution_parameter: float = 1,
) -> NodeClustering:
    """
    rber_pots is a  model where the quality function to optimize is:

    .. math:: Q = \\sum_{ij} \\left(A_{ij} - \\gamma p \\right)\\delta(\\sigma_i, \\sigma_j)

    where :math:`A` is the adjacency matrix,  :math:`p = \\frac{m}{\\binom{n}{2}}` is the overall density of the graph, :math:`\\sigma_i` denotes the community of node :math:`i`, :math:`\\delta(\\sigma_i, \\sigma_j) = 1` if  :math:`\\sigma_i = \\sigma_j` and `0` otherwise, and, finally :math:`\\gamma` is a resolution parameter.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========


    :param g_original: a networkx/igraph object
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

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)

    part = leidenalg.find_partition(
        g,
        leidenalg.RBERVertexPartition,
        resolution_parameter=resolution_parameter,
        initial_membership=initial_membership,
        weights=weights,
        node_sizes=node_sizes,
    )
    coms = [g.vs[x]["name"] for x in part]
    return NodeClustering(
        coms,
        g_original,
        "RBER Pots",
        method_parameters={
            "initial_membership": initial_membership,
            "weights": weights,
            "node_sizes": node_sizes,
            "resolution_parameter": resolution_parameter,
        },
    )


def cpm(
    g_original: object,
    initial_membership: list = None,
    weights: list = None,
    node_sizes: list = None,
    resolution_parameter: float = 1,
) -> NodeClustering:
    """
    CPM is a model where the quality function to optimize is:

    .. math:: Q = \\sum_{ij} \\left(A_{ij} - \\gamma \\right)\\delta(\\sigma_i, \\sigma_j)

    where :math:`A` is the adjacency matrix, :math:`\\sigma_i` denotes the community of node :math:`i`, :math:`\\delta(\\sigma_i, \\sigma_j) = 1` if  :math:`\\sigma_i = \\sigma_j` and `0` otherwise, and, finally :math:`\\gamma` is a resolution parameter.

    The internal density of communities

    .. math:: p_c = \\frac{m_c}{\\binom{n_c}{2}} \\geq \\gamma

    is higher than :math:`\\gamma`, while the external density

    :math:`p_{cd} = \\frac{m_{cd}}{n_c n_d} \\leq \\gamma`    is lower than :math:`\\gamma`. In other words, choosing a particular
    :math:`\\gamma` corresponds to choosing to find communities of a particular
    density, and as such defines communities. Finally, the definition of a community is in a sense independent of the actual graph, which is not the case for any of the other methods.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========


    :param g_original: a networkx/igraph object
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

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)

    part = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        resolution_parameter=resolution_parameter,
        initial_membership=initial_membership,
        weights=weights,
        node_sizes=node_sizes,
    )
    coms = [g.vs[x]["name"] for x in part]
    return NodeClustering(
        coms,
        g_original,
        "CPM",
        method_parameters={
            "initial_membership": initial_membership,
            "weights": weights,
            "node_sizes": node_sizes,
            "resolution_parameter": resolution_parameter,
        },
    )


def significance_communities(
    g_original: object, initial_membership: list = None, node_sizes: list = None
) -> NodeClustering:
    """
    Significance_communities is a model where the quality function to optimize is:

    .. math:: Q = \\sum_c \\binom{n_c}{2} D(p_c \\parallel p)

    where :math:`n_c` is the number of nodes in community :math:`c`, :math:`p_c = \\frac{m_c}{\\binom{n_c}{2}}`, is the density of community :math:`c`,  :math:`p = \\frac{m}{\\binom{n}{2}}`  is the overall density of the graph, and finally  :math:`D(x \\parallel y) = x \\ln \\frac{x}{y} + (1 - x) \\ln \\frac{1 - x}{1 - y}` is the binary Kullback-Leibler divergence.
    For directed graphs simply multiply the binomials by 2. The expected Significance in Erdos-Renyi graphs behaves roughly as :math:`\\frac{1}{2} n \\ln n` for both directed and undirected graphs in this formulation.

    .. warning:: This method is not suitable for weighted graphs.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========


    :param g_original: a networkx/igraph object
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

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)

    part = leidenalg.find_partition(
        g,
        leidenalg.SignificanceVertexPartition,
        initial_membership=initial_membership,
        node_sizes=node_sizes,
    )
    coms = [g.vs[x]["name"] for x in part]
    return NodeClustering(
        coms,
        g_original,
        "Significance",
        method_parameters={
            "initial_membership": initial_membership,
            "node_sizes": node_sizes,
        },
    )


def surprise_communities(
    g_original: object,
    initial_membership: list = None,
    weights: list = None,
    node_sizes: list = None,
) -> NodeClustering:
    """

    Surprise_communities is a model where the quality function to optimize is:

    .. math:: Q = m D(q \\parallel \\langle q \\rangle)

    where :math:`m` is the number of edges,  :math:`q = \\frac{\\sum_c m_c}{m}`,  is the fraction of internal edges, :math:`\\langle q \\rangle = \\frac{\\sum_c \\binom{n_c}{2}}{\\binom{n}{2}}` is the expected fraction of internal edges, and finally

    :math:`D(x \\parallel y) = x \\ln \\frac{x}{y} + (1 - x) \\ln \\frac{1 - x}{1 - y}`  is the binary Kullback-Leibler divergence.

    For directed graphs we can multiplying the binomials by 2, and this leaves :math:`\\langle q \\rangle` unchanged, so that we can simply use the same
    formulation.  For weighted graphs we can simply count the total internal weight instead of the total number of edges for :math:`q` , while :math:`\\langle q \\rangle` remains unchanged.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        Yes      Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
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

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)

    part = leidenalg.find_partition(
        g,
        leidenalg.SurpriseVertexPartition,
        initial_membership=initial_membership,
        weights=weights,
        node_sizes=node_sizes,
    )
    coms = [g.vs[x]["name"] for x in part]
    return NodeClustering(
        coms,
        g_original,
        "Surprise",
        method_parameters={
            "initial_membership": initial_membership,
            "weights": weights,
            "node_sizes": node_sizes,
        },
    )


def greedy_modularity(g_original: object, weight: list = None) -> NodeClustering:
    """
    The CNM algorithm uses the modularity to find the communities strcutures.
    At every step of the algorithm two communities that contribute maximum positive value to global modularity are merged.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
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
    g = convert_graph_formats(g_original, nx.Graph)

    coms = nx.algorithms.community.greedy_modularity_communities(g, weight)
    coms = [list(x) for x in coms]
    return NodeClustering(
        coms, g_original, "Greedy Modularity", method_parameters={"weight": weight}
    )


def infomap(g_original: object, flags: str = "") -> NodeClustering:
    """
    Infomap is based on ideas of information theory.
    The algorithm uses the probability flow of random walks on a network as a proxy for information flows in the real system and it decomposes the network into modules by compressing a description of the probability flow.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        Yes      Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param flags: str flags for Infomap
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.infomap(G)

    :References:

    Rosvall M, Bergstrom CT (2008) `Maps of random walks on complex networks reveal community structure. <https://www.pnas.org/content/105/4/1118/>`_ Proc Natl Acad SciUSA 105(4):1118–1123

    .. note:: Reference implementation: https://pypi.org/project/infomap/

    .. note:: Infomap Python API documentation: https://mapequation.github.io/infomap/python/
    """

    if imp is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install infomap to use the selected feature."
        )
    if pipes is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install package wurlitzer to use infomap."
        )

    g = convert_graph_formats(g_original, nx.Graph, directed=g_original.is_directed())

    if g_original.is_directed() and "-d" not in flags and "--directed" not in flags:
        flags += " -d"

    g1 = nx.convert_node_labels_to_integers(g, label_attribute="name")
    name_map = nx.get_node_attributes(g1, "name")
    coms_to_node = defaultdict(list)

    with pipes():
        im = imp.Infomap(flags)

        im.add_nodes(g1.nodes)

        for source, target, data in g1.edges(data=True):
            if "weight" in data:
                im.add_link(source, target, data["weight"])
            else:
                im.add_link(source, target)
        im.run()

        for node_id, module_id in im.modules:
            node_name = name_map[node_id]
            coms_to_node[module_id].append(node_name)

    coms_infomap = [list(c) for c in coms_to_node.values()]
    return NodeClustering(
        coms_infomap, g_original, "Infomap", method_parameters={"flags": flags}
    )


def walktrap(g_original: object) -> NodeClustering:
    """
    walktrap is an approach based on random walks.
    The general idea is that if you perform random walks on the graph, then the walks are more likely to stay within the same community because there are only a few edges that lead outside a given community. Walktrap runs short random walks and uses the results of these random walks to merge separate communities in a bottom-up manner.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No      No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClusterint object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.walktrap(G)

    :References:

    Pons, Pascal, and Matthieu Latapy. `Computing communities in large networks using random walks. <http://jgaa.info/accepted/2006/PonsLatapy2006.10.2.pdf/>`_ J. Graph Algorithms Appl. 10.2 (2006): 191-218.
    """

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)
    coms = g.community_walktrap().as_clustering()
    communities = []

    for c in coms:
        communities.append([g.vs[x]["name"] for x in c])

    return NodeClustering(
        communities, g_original, "Walktrap", method_parameters={"": ""}
    )


def label_propagation(g_original: object) -> NodeClustering:
    """
    The Label Propagation algorithm (LPA) detects communities using network structure alone.
    The algorithm doesn’t require a pre-defined objective function or prior information about the communities.
    It works as follows:
    -Every node is initialized with a unique label (an identifier)
    -These labels propagate through the network
    -At every iteration of propagation, each node updates its label to the one that the maximum numbers of its neighbours belongs to. Ties are broken uniformly and randomly.
    -LPA reaches convergence when each node has the majority label of its neighbours.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: EdgeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.label_propagation(G)

    :References:

    Raghavan, U. N., Albert, R., & Kumara, S. (2007). `Near linear time algorithm to detect community structures in large-scale networks. <http://www.leonidzhukov.net/hse/2017/networks/papers/raghavan2007.pdf/>`_ Physical review E, 76(3), 036106.
    """

    g = convert_graph_formats(g_original, nx.Graph)

    coms = list(nx.algorithms.community.label_propagation_communities(g))
    coms = [list(x) for x in coms]

    return NodeClustering(
        coms, g_original, "Label Propagation", method_parameters={"": ""}
    )


def async_fluid(g_original: object, k: int) -> NodeClustering:
    """
    Fluid Communities (FluidC) is based on the simple idea of fluids (i.e., communities) interacting in an environment (i.e., a non-complete graph), expanding and contracting.
    It is propagation-based algorithm and it allows to specify the number of desired communities (k) and it is asynchronous, where each vertex update is computed using the latest partial state of the graph.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========


    :param g_original: a networkx/igraph object
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

    g = convert_graph_formats(g_original, nx.Graph)

    coms = nx.algorithms.community.asyn_fluidc(g, k)
    coms = [list(x) for x in coms]
    return NodeClustering(coms, g_original, "Fluid", method_parameters={"k": k})


def der(
    g_original: object,
    walk_len: int = 3,
    threshold: float = 0.00001,
    iter_bound: int = 50,
) -> NodeClustering:
    """
    DER is a Diffusion Entropy Reducer graph clustering algorithm.
    The algorithm uses random walks to embed the graph in a space of measures, after which a modification of k-means in that space is applied. It creates the walks, creates an initialization, runs the algorithm,
    and finally extracts the communities.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: an undirected networkx graph object
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

    graph = convert_graph_formats(g_original, nx.Graph)

    communities, _ = DER.der_graph_clustering(
        graph, walk_len=walk_len, alg_threshold=threshold, alg_iterbound=iter_bound
    )

    maps = {k: v for k, v in enumerate(graph.nodes())}
    coms = []
    for c in communities:
        coms.append([maps[n] for n in c])

    return NodeClustering(
        coms,
        g_original,
        "DER",
        method_parameters={
            "walk_len": walk_len,
            "threshold": threshold,
            "iter_bound": iter_bound,
        },
    )


def frc_fgsn(
    g_original: object, theta: float, eps: float, r: int
) -> FuzzyNodeClustering:
    """Fuzzy-Rough Community Detection on Fuzzy Granular model of Social Network.

    FRC-FGSN assigns nodes to communities specifying the probability of each association.
    The flattened partition ensure that each node is associated to the community that maximize such association probability.
    FRC-FGSN may generate orphan nodes (i.e., nodes not assigned to any community).


    **Supported Graph Types**

    ========== ======== ======== =========
    Undirected Directed Weighted BiPartite
    ========== ======== ======== =========
    Yes        No       No       Yes
    ========== ======== ======== =========

    :param g_original: networkx/igraph object
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

    graph = convert_graph_formats(g_original, nx.Graph)
    g, maps = nx_node_integer_mapping(graph)

    communities, fuzz_assoc = fuzzy_comm(graph, theta, eps, r)

    if maps is not None:
        coms = []
        for c in communities:
            coms.append([maps[n] for n in c])

        nx.relabel_nodes(g, maps, False)
        fuzz_assoc = {maps[nid]: v for nid, v in fuzz_assoc.items()}
    else:
        coms = [list(c) for c in communities]

    return FuzzyNodeClustering(
        coms,
        fuzz_assoc,
        g_original,
        "FuzzyComm",
        method_parameters={"theta": theta, "eps": eps, "r": r},
    )


def principled_clustering(
    g_original: object, cluster_count: int
) -> FuzzyNodeClustering:
    """
    An efficient and principled method for detecting communities in networks


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: networkx/igraph object
    :param cluster_count: number of desired communities
    :return: FuzzyNodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = principled_clustering(G, 3)


    :References:

    B Ball, B., & E JNewman, M. (2011). An efficient and principled method for detecting communities in networks. Physical ReviewE, 84, 036103.

    .. note:: Reference implementation: https://github.com/Zabot/principled_clustering
    """

    graph = convert_graph_formats(g_original, nx.Graph)
    g, maps = nx_node_integer_mapping(graph)

    communities, fuzz_assoc = principled(graph, cluster_count)

    if maps is not None:
        coms = []
        for c in communities:
            coms.append([maps[n] for n in c])

        nx.relabel_nodes(g, maps, False)
        fuzz_assoc = {maps[nid]: v for nid, v in fuzz_assoc.items()}
    else:
        coms = [list(c) for c in communities]

    return FuzzyNodeClustering(
        coms,
        fuzz_assoc,
        g_original,
        "Principled Clustering",
        method_parameters={"cluster_count": cluster_count},
    )


def sbm_dl(
    g_original: object,
) -> NodeClustering:
    """Efficient Monte Carlo and greedy heuristic for the inference of stochastic block models.

    Fit a non-overlapping stochastic block model (SBM) by minimizing its description length using an agglomerative heuristic.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: network/igraph object
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = sbm_dl(G)


    :References:

    Tiago P. Peixoto, “Efficient Monte Carlo and greedy heuristic for the inference of stochastic block models”, Phys. Rev. E 89, 012804 (2014), DOI: 10.1103/PhysRevE.89.012804 [sci-hub, @tor], arXiv: 1310.4378.

    .. note:: Implementation from graph-tool library, please report to https://graph-tool.skewed.de for details
    """
    if gt is None:
        raise Exception(
            "===================================================== \n"
            "The graph-tool library seems not to be installed (or incorrectly installed). \n"
            "Please check installation procedure there https://git.skewed.de/count0/graph-tool/wikis/installation-instructions#native-installation \n"
            "on linux/mac, you can use package managers to do so(apt-get install python3-graph-tool, brew install graph-tool, etc.)"
        )
    gt_g = convert_graph_formats(g_original, nx.Graph)
    gt_g, label_map = __from_nx_to_graph_tool(gt_g)
    state = gt.minimize_blockmodel_dl(gt_g)

    affiliations = state.get_blocks().get_array()
    affiliations = {label_map[i]: affiliations[i] for i in range(len(affiliations))}
    coms = affiliations2nodesets(affiliations)
    coms = [list(v) for k, v in coms.items()]
    return NodeClustering(coms, g_original, "SBM", method_parameters={})


def sbm_dl_nested(
    g_original: object,
) -> NodeClustering:
    """Efficient Monte Carlo and greedy heuristic for the inference of stochastic block models. (nested)

    Fit a nested non-overlapping stochastic block model (SBM) by minimizing its description length using an agglomerative heuristic.
    Return the lowest level found. Currently cdlib do not support hierarchical clustering.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: igraph/networkx object
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = sbm_dl(G)


    :References:

    Tiago P. Peixoto, “Hierarchical block structures and high-resolution model selection in large networks”, Physical Review X 4.1 (2014): 011047

    .. note:: Implementation from graph-tool library, please report to https://graph-tool.skewed.de for details
    """
    if gt is None:
        raise Exception(
            "===================================================== \n"
            "The graph-tool library seems not to be installed (or incorrectly installed). \n"
            "Please check installation procedure there https://git.skewed.de/count0/graph-tool/wikis/installation-instructions#native-installation \n"
            "on linux/mac, you can use package managers to do so(apt-get install python3-graph-tool, brew install graph-tool, etc.)"
        )
    gt_g = convert_graph_formats(g_original, nx.Graph)
    gt_g, label_map = __from_nx_to_graph_tool(gt_g)
    state = gt.minimize_nested_blockmodel_dl(gt_g)

    level0 = state.get_levels()[0]

    affiliations = level0.get_blocks().get_array()
    affiliations = {label_map[i]: affiliations[i] for i in range(len(affiliations))}
    coms = affiliations2nodesets(affiliations)
    coms = [list(v) for k, v in coms.items()]
    return NodeClustering(
        coms,
        g_original,
        "SBM_nested",
        method_parameters={},
    )


def markov_clustering(
    g_original: object,
    expansion: int = 2,
    inflation: int = 2,
    loop_value: int = 1,
    iterations: int = 100,
    pruning_threshold: float = 0.001,
    pruning_frequency: int = 1,
    convergence_check_frequency: int = 1,
) -> NodeClustering:
    """
    The Markov clustering algorithm (MCL) is based on simulation of (stochastic) flow in graphs.
    The MCL algorithm finds cluster structure in graphs by a mathematical bootstrapping procedure. The process deterministically computes (the probabilities of) random walks through the graph, and uses two operators transforming one set of probabilities into another. It does so using the language of stochastic matrices (also called Markov matrices) which capture the mathematical concept of random walks on a graph.
    The MCL algorithm simulates random walks within a graph by alternation of two operators called expansion and inflation.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param expansion: The cluster expansion factor
    :param inflation: The cluster inflation factor
    :param loop_value: Initialization value for self-loops
    :param iterations: Maximum number of iterations
           (actual number of iterations will be less if convergence is reached)
    :param pruning_threshold: Threshold below which matrix elements will be set set to 0
    :param pruning_frequency: Perform pruning every 'pruning_frequency'
           iterations.
    :param convergence_check_frequency: Perform the check for convergence
           every convergence_check_frequency iterations
    :return:  NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.markov_clustering(G)

    :References:

    Enright, Anton J., Stijn Van Dongen, and Christos A. Ouzounis. `An efficient algorithm for large-scale detection of protein families. <https://www.ncbi.nlm.nih.gov/pubmed/11917018/>`_ Nucleic acids research 30.7 (2002): 1575-1584.

    .. note:: Reference implementation: https://github.com/GuyAllard/markov_clustering
    """

    g = convert_graph_formats(g_original, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    if maps is not None:
        matrix = nx.to_scipy_sparse_matrix(g, nodelist=range(len(maps)))
    else:
        matrix = nx.to_scipy_sparse_matrix(g)

    result = mc.run_mcl(
        matrix,
        expansion=expansion,
        inflation=inflation,
        loop_value=loop_value,
        iterations=iterations,
        pruning_threshold=pruning_threshold,
        pruning_frequency=pruning_frequency,
        convergence_check_frequency=convergence_check_frequency,
    )  # run MCL with default parameters
    clusters = mc.get_clusters(result)

    coms = []
    if maps is not None:
        for c in clusters:
            coms.append([maps[n] for n in c])

        nx.relabel_nodes(g, maps, False)
    else:
        coms = [list(c) for c in clusters]

    return NodeClustering(
        coms,
        g_original,
        "Markov Clustering",
        method_parameters={
            "expansion": expansion,
            "inflation": inflation,
            "loop_value": loop_value,
            "iterations": iterations,
            "pruning_threshold": pruning_threshold,
            "pruning_frequency": pruning_frequency,
            "convergence_check_frequency": convergence_check_frequency,
        },
    )


def chinesewhispers(
    g_original: object, weighting: str = "top", iterations: int = 20, seed: int = None
) -> NodeClustering:
    """

    Fuzzy graph clustering that (i) creates an intermediate representation of the input graph, which reflects the “ambiguity” of its nodes,
    and (ii) uses hard clustering to discover crisp clusters in such “disambiguated” intermediate graph.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original:
    :param weighting: edge weighing schemas. Available modalities: ['top', 'lin', 'log']
    :param iterations: number of iterations
    :param seed: random seed
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.chinesewhispers(G)

    :References:

    Biemann, Chris. 2006. Chinese Whispers: An Efficient Graph Clustering Algorithm and Its Application to Natural Language Processing Problems. In Proceedings of the First Workshop on Graph Based Methods for Natural Language Processing, TextGraphs-1, pages 73–80, Association for Computational Linguistics, New York, NY, USA.

    .. note:: Reference implementation: https://github.com/nlpub/chinese-whispers-python
    """

    g = convert_graph_formats(g_original, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    cw(g, weighting=weighting, iterations=iterations, seed=seed)

    coms = []
    if maps is not None:
        for _, cluster in sorted(
            aggregate_clusters(g).items(), key=lambda e: len(e[1]), reverse=True
        ):
            coms.append([maps[n] for n in cluster])

        nx.relabel_nodes(g, maps, False)
    else:
        for _, cluster in sorted(
            aggregate_clusters(g).items(), key=lambda e: len(e[1]), reverse=True
        ):
            coms.append(list(cluster))

    return NodeClustering(
        coms,
        g_original,
        "Chinese Whispers",
        method_parameters={"weighting": weighting, "iterations": iterations},
    )


def edmot(
    g_original: object, component_count: int = 2, cutoff: int = 10
) -> NodeClustering:
    """
    The algorithm first creates the graph of higher order motifs. This graph is clustered by the Louvain method.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param component_count: Number of extracted motif hypergraph components. Default is 2.
    :param cutoff: Motif edge cut-off value. Default is 10.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.edmot(G, max_loop=1000)

    :References:

    Li, Pei-Zhen, et al. "EdMot: An Edge Enhancement Approach for Motif-aware Community Detection." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """

    g = convert_graph_formats(g_original, nx.Graph)
    model = EdMot(component_count=2, cutoff=10)

    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "EdMot",
        method_parameters={"component_count": component_count, "cutoff": cutoff},
    )


def siblinarity_antichain(
    g_original: object,
    forwards_backwards_on: bool = True,
    backwards_forwards_on: bool = False,
    Lambda: int = 1,
    with_replacement: bool = False,
) -> NodeClustering:
    """
    The algorithm extract communities from a DAG that (i) respects its intrinsic order and (ii) are composed of similar nodes.
    The approach takes inspiration from classic similarity measures of bibliometrics, used to assess how similar two publications are, based on their relative citation patterns.


    **Supported Graph Types**

    ========== ========= ========
    Undirected Directed  Weighted
    ========== ========= ========
    No         Yes (DAG) No
    ========== ========= ========

    :param g_original: a networkx/igraph object representing a DAG (directed acyclic graph)
    :param forwards_backwards_on: checks successors' similarity. Boolean, default True
    :param backwards_forwards_on: checks predecessors' similarity. Boolean, default True
    :param Lambda: desired resolution of the partition. Default 1
    :param with_replacement: If True he similarity of a node to itself is equal to the number of its neighbours based on which the similarity is defined. Boolean, default True.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.siblinarity_antichain(G, Lambda=1)

    :References:

    Vasiliauskaite, V., Evans, T.S. Making communities show respect for order. Appl Netw Sci 5, 15 (2020). https://doi.org/10.1007/s41109-020-00255-5

    .. note:: Reference implementation: https://github.com/vv2246/siblinarity_antichains
    """

    g = convert_graph_formats(g_original, nx.Graph)

    if not nx.is_directed_acyclic_graph(g):
        raise Exception(
            "The Siblinarity Antichain algorithm require as input a Directed Acyclic Graph (DAG)."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_list = matrix_node_recursive_antichain_partition(
            g,
            forwards_backwards_on=forwards_backwards_on,
            backwards_forwards_on=backwards_forwards_on,
            Q_check_on=True,
            Lambda=Lambda,
            with_replacement=with_replacement,
            space_label=None,
            time_label=None,
        )

    node_partition = {}
    for n in g.nodes():
        p_at_level = result_list[0]["n_to_p"][n]
        for i in range(1, len(result_list) - 1):
            p_at_level = result_list[i]["n_to_p"][p_at_level]
        node_partition[n] = p_at_level

    partition = defaultdict(list)
    for key, val in node_partition.items():
        partition[val].append(key)

    coms = [list(c) for c in partition.values()]

    return NodeClustering(
        coms,
        g_original,
        "Siblinarity Antichain",
        method_parameters={
            "forwards_backwards_on": forwards_backwards_on,
            "backwards_forwards_on": backwards_forwards_on,
            "Lambda": Lambda,
            "with_replacement": with_replacement,
        },
    )


def ga(
    g_original: object, population: int = 300, generation: int = 30, r: float = 1.5
) -> NodeClustering:
    """
    Genetic based approach to discover communities in social networks.
    GA optimizes a simple but efficacious fitness function able to identify densely connected groups of nodes with sparse connections between groups.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param population:
    :param generation:
    :param r:
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.ga(G)

    :References:

     Pizzuti, C. (2008). Ga-net: A genetic algorithm for community detection in social networks. In Inter conf on parallel problem solving from nature, pages 1081–1090.Springer.

    .. note:: Reference implementation: https://github.com/hariswb/ga-community-detection
    """

    g = convert_graph_formats(g_original, nx.Graph)
    flag = False
    for _, _, d in g.edges(data=True):
        if len(d) > 0:
            flag = True
        d.clear()

    if flag:
        warnings.warn(
            "GA only works on unweighted graphs: edge attributes have been removed from the input network"
        )

    coms = ga_community_detection(g, population, generation, r)

    return NodeClustering(
        coms,
        g_original,
        "ga",
        method_parameters={"population": population, "generation": generation, "r": r},
    )


def belief(
    g_original: object,
    max_it: int = 100,
    eps: float = 0.0001,
    reruns_if_not_conv: int = 5,
    threshold: float = 0.005,
    q_max: int = 7,
) -> NodeClustering:
    """
    Belief community seeks the consensus of many high-modularity partitions.
    It does this with a scalable message-passing algorithm, derived by treating the modularity as a Hamiltonian and applying the cavity method.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param max_it:
    :param eps:
    :param reruns_if_not_conv:
    :param threshold:
    :param q_max:
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.belief(G)

    :References:

    Zhang, Pan, and Cristopher Moore. "Scalable detection of statistically significant communities and hierarchies, using message passing for modularity." Proceedings of the National Academy of Sciences 111.51 (2014): 18144-18149.

    .. note:: Reference implementation: https://github.com/weberfm/belief_propagation_community_detection
    """

    g = convert_graph_formats(g_original, nx.Graph)

    mapping = {n: i for i, n in enumerate(g.nodes())}
    inv_map = {v: k for k, v in mapping.items()}
    g = nx.relabel_nodes(g, mapping)

    coms = detect_belief_communities(
        g,
        max_it=max_it,
        eps=eps,
        reruns_if_not_conv=reruns_if_not_conv,
        threshold=threshold,
        q_max=q_max,
    )

    res = []
    for com in coms:
        com = [inv_map[c] for c in com]
        res.append(com)

    return NodeClustering(
        res,
        g_original,
        "Belief",
        method_parameters={
            "max_it": max_it,
            "eps": eps,
            "reruns_if_not_conv": reruns_if_not_conv,
            "threshold": threshold,
            "q_max": q_max,
        },
    )


def threshold_clustering(
    g_original: object, threshold_function: Callable[[list], float] = np.mean
) -> NodeClustering:
    """
    Developed for semantic similarity networks, this algorithm specifically targets **weighted** and **directed** graphs.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        Yes      Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param threshold_function: callable, optional
        Ties smaller than threshold_function(out_ties) are deleted. Example: np.mean, np.median. Default is np.mean.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.threshold_clustering(G)

    :References:

    Guzzi, Pietro Hiram, Pierangelo Veltri, and Mario Cannataro. "Thresholding of semantic similarity networks using a spectral graph-based technique." International Workshop on New Frontiers in Mining Complex Patterns. Springer, Cham, 2013.

    """

    g = convert_graph_formats(g_original, nx.Graph)

    if not nx.is_directed(g):
        warnings.warn(
            "Threshold Clustering is defined for directed graphs: the undirected graph in input will be treated as directed."
        )

    if not nx.is_weighted(g):
        raise ValueError("Threshold Clustering is defined only for weighted graphs.")

    coms, _ = th_best_partition(g, threshold_function=threshold_function)

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in coms.items():
        coms_to_node[c].append(n)

    coms_louvain = [list(c) for c in coms_to_node.values()]
    return NodeClustering(
        coms_louvain, g_original, "Threshold Clustering", method_parameters={}
    )


def lswl(
    g_original: object,
    query_node: object,
    strength_type: int = 2,
    timeout: float = 1.0,
    online: bool = True,
) -> NodeClustering:
    """

    LSWL locally discovers networks' the communities precisely, deterministically, and quickly.
    This method works in a one-node-expansion model based on a notion of strong and weak links in a graph.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param timeout: The maximum time in which LSWL should retrieve the community. Default is 1 second.
    :param strength_type: 1 strengths between [-1,+1] or, 2 strengths between [0,1]. Default, 2.
    :param query_node: Id of the network node whose local community is queried.
    :param online: wehter the computation should happen in memory or not. Default, True.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.lswl(G, 1)

    :References:

    Fast Local Community Discovery: Relying on the Strength of Links (submitted for KDD 2021).

    .. note:: Reference implementation: https://github.com/mahdi-zafarmand/LSWL

    """

    g = convert_graph_formats(g_original, nx.Graph)
    if online:
        community_searcher = LSWLCommunityDiscovery(g, strength_type, timeout)
    else:
        community_searcher = LSWLCommunityDiscovery_offline(g, strength_type, timeout)

    community = community_searcher.community_search(start_node=query_node)
    community_searcher.reset()

    return NodeClustering(
        [community],
        g_original,
        "LSWL",
        method_parameters={
            "query_node": query_node,
            "strength_type": strength_type,
            "timeout": timeout,
            "online": online,
        },
    )


def lswl_plus(
    g_original: object,
    strength_type: int = 1,
    merge_outliers: bool = True,
    detect_overlap: bool = False,
) -> NodeClustering:
    """
    LSWL+ is capable of finding a partition with overlapping communities or without them, based on user preferences.
    This method can also find outliers (peripheral nodes of the graph that are marginally connected to communities) and hubs (nodes that bridge the communities)


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param strength_type: 1 strengths between [-1,+1] or, 2 strengths between [0,1]. Default, 2.
    :param merge_outliers: If outliers need to merge into communities. Default, True.
    :param detect_overlap: If overlapping communities need to be detected. Default, False
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.lswl_plus(G)

    :References:

    Fast Local Community Discovery: Relying on the Strength of Links (submitted for KDD 2021)

    .. note:: Reference implementation: https://github.com/mahdi-zafarmand/LSWL

    """

    g = convert_graph_formats(g_original, nx.Graph)

    community_detector = LSWLPlusCommunityDetection(
        deepcopy(g), strength_type, merge_outliers, detect_overlap
    )
    partition = community_detector.community_detection()

    return NodeClustering(
        partition,
        g_original,
        "LSWL+",
        method_parameters={
            "strength_type": strength_type,
            "merge_outliers": merge_outliers,
            "detect_overlap": detect_overlap,
        },
    )


def mod_r(g_original: object, query_node: object) -> NodeClustering:
    """
    Community Discovery algorithm that infers the hierarchy of communities that enclose a given vertex by exploring the graph one vertex at a time.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========


    :param g_original: a networkx/igraph object
    :param query_node: Id of the network node whose local community is queried.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.mod_r(G, 1)

    :References:

    Clauset, Aaron. "Finding local community structure in networks." Physical review E 72.2 (2005): 026132.

    .. note:: Reference implementation: https://github.com/mahdi-zafarmand/LSWL

    """

    g = convert_graph_formats(g_original, nx.Graph)
    community_searcher = ModularityRCommunityDiscovery(g)
    community = community_searcher.community_search(start_node=query_node)
    community_searcher.reset()

    return NodeClustering(
        [community], g_original, "mod_r", method_parameters={"query_node": query_node}
    )


def mod_m(g_original: object, query_node: object) -> NodeClustering:
    """
    Community Discovery algorithm designed to find local optimal community structures in large networks starting from a given source vertex.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param query_node: Id of the network node whose local community is queried.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.mod_m(G, 1)

    :References:

    Luo, Feng, James Z. Wang, and Eric Promislow. "Exploring local community structures in large networks." Web Intelligence and Agent Systems: An International Journal 6.4 (2008): 387-400.

    .. note:: Reference implementation: https://github.com/mahdi-zafarmand/LSWL

    """

    g = convert_graph_formats(g_original, nx.Graph)
    community_searcher = ModularityMCommunityDiscovery(g)
    community = community_searcher.community_search(start_node=query_node)
    community_searcher.reset()

    return NodeClustering(
        [community], g_original, "mod_m", method_parameters={"query_node": query_node}
    )


def head_tail(g_original: object, head_tail_ratio: float = 0.4) -> NodeClustering:
    """
    Identifying homogeneous communities in complex networks by applying head/tail breaks on edge betweenness given its heavy-tailed distribution.

    Note: this implementation is suited for small-medium sized graphs, and it may take couple of minutes or longer for a bigger graph.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param head_tail_ratio: head/tail division rule. Float in [0,1], dafault 0.4.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.head_tail()
    >>> coms = algorithms.head_tail(G, head_tail_ratio=0.8)

    :References:

    Jiang B. and Ding M. (2015), Defining least community as a homogeneous group in complex networks, Physica A, 428, 154-160.

    .. note:: Reference implementation: https://github.com/dingmartin/HeadTailCommunityDetection

    """

    g = convert_graph_formats(g_original, nx.Graph)
    coms = HeadTail(g)

    return NodeClustering(
        coms,
        g_original,
        "head_tail",
        method_parameters={"head_tail_ratio": head_tail_ratio},
    )


def kcut(g_original: object, kmax: int = 4) -> NodeClustering:
    """
    An Efficient Spectral Algorithm for Network Community Discovery.
    Kcut is designed to provide a unique combination of recursive partitioning and direct k-way methods, able to guarantee the efficiency of a recursive approach, while also having the same accuracy as a direct k-way method.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param kmax: maximum value of k, dafault 4.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.head_tail()
    >>> coms = algorithms.kcut(G, head_tail_ratio=0.8)

    :References:

    Ruan, Jianhua, and Weixiong Zhang. "An efficient spectral algorithm for network community discovery and its applications to biological and social networks." Seventh IEEE International Conference on Data Mining (ICDM 2007). IEEE, 2007.

    .. note:: Reference implementation: https://github.com/hmliangliang/kcut-algorithm

    """

    g = convert_graph_formats(g_original, nx.Graph)
    coms = kcut_exec(g, kmax)

    return NodeClustering(coms, g_original, "Kcut", method_parameters={"kmax": kmax})


def gemsec(
    g_original: object,
    walk_number: int = 5,
    walk_length: int = 80,
    dimensions: int = 32,
    negative_samples: int = 5,
    window_size: int = 5,
    learning_rate: float = 0.1,
    clusters: int = 10,
    gamma: float = 0.1,
    seed: int = 42,
) -> NodeClustering:
    """
    The procedure uses random walks to approximate the pointwise mutual information matrix obtained by pooling normalized adjacency matrix powers.
    This matrix is decomposed by an approximate factorization technique which is combined with a k-means like clustering cost.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        Yes      No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param walk_number: Number of random walks. Default is 5.
    :param walk_length: Length of random walks. Default is 80.
    :param dimensions: Dimensionality of embedding. Default is 32.
    :param negative_samples: Number of negative samples. Default is 5.
    :param window_size: Matrix power order. Default is 5.
    :param learning_rate: Gradient descent learning rate. Default is 0.1.
    :param clusters: Number of cluster centers. Default is 10.
    :param gamma: Clustering cost weight coefficient. Default is 0.1.
    :param seed: Random seed value. Default is 42.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.gemsec(G)

    :References:

    Rozemberczki, B., Davies, R., Sarkar, R., & Sutton, C. (2019, August). Gemsec: Graph embedding with self clustering. In Proceedings of the 2019 IEEE/ACM international conference on advances in social networks analysis and mining (pp. 65-72).

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """
    g = convert_graph_formats(g_original, nx.Graph)
    model = GEMSEC(
        walk_number=walk_number,
        walk_length=walk_length,
        dimensions=dimensions,
        negative_samples=negative_samples,
        window_size=window_size,
        learning_rate=learning_rate,
        clusters=clusters,
        gamma=gamma,
        seed=seed,
    )
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "GEMSEC",
        method_parameters={
            "walk_number": walk_number,
            "walk_length": walk_length,
            "dimensions": dimensions,
            "negative_samples": negative_samples,
            "window_size": window_size,
            "learning_rate": learning_rate,
            "clusters": clusters,
            "gamma": gamma,
            "seed": seed,
        },
        overlap=False,
    )


def scd(
    g_original: object, iterations: int = 25, eps: float = 1e-06, seed: int = 42
) -> NodeClustering:
    """
    The procedure greedily optimizes the approximate weighted community clustering metric.
    First, clusters are built around highly clustered nodes. Second, we refine the initial partition by using the approximate WCC.
    These refinements happen for the whole vertex set.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param iterations: Refinemeent iterations. Default is 25.
    :param eps: Epsilon score for zero division correction. Default is 10**-6.
    :param seed: Random seed value. Default is 42.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.scd(G)

    :References:

    Prat-Pérez, A., Dominguez-Sal, D., & Larriba-Pey, J. L. (2014, April). High quality, scalable and parallel community detection for large real graphs. In Proceedings of the 23rd international conference on World wide web (pp. 225-236).

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """
    g = convert_graph_formats(g_original, nx.Graph)
    model = SCD(iterations=iterations, eps=eps, seed=seed)
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "SCD",
        method_parameters={"iterations": iterations, "eps": eps, "seed": seed},
        overlap=False,
    )


def pycombo(
    g_original: object,
    weight: str = "weight",
    max_communities: int = None,
    modularity_resolution: float = 1.0,
    num_split_attempts: int = 0,
    start_separate: bool = False,
    treat_as_modularity: bool = False,
    random_seed: int = 42,
) -> NodeClustering:
    """
    This is an implementation (for Modularity maximization) of the community detection algorithm called "Combo".


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param weight: Optional, defaults to weight. Graph edges property to use as weights. If None, graph assumed to be unweighted. Ignored if graph is passed as string (path to the file), or such property does not exist.
    :param max_communities: Optional, defaults to None. Maximum number of communities. If <= 0 or None, assume to be infinite.
    :param modularity_resolution: float, defaults to 1.0. Modularity resolution parameter.
    :param num_split_attempts: int, defaults to 0. Number of split attempts. If 0, autoadjust this number automatically.
    :param start_separate: bool, default False. Indicates if Combo should start from assigning each node into its own separate community. This could help to achieve higher modularity, but it makes execution much slower.
    :param treat_as_modularity:  bool, default False. Indicates if edge weights should be treated as modularity scores. If True, the algorithm solves clique partitioning problem over the given graph, treated as modularity graph (matrix). For example, this allows users to provide their own custom 'modularity' matrix. modularity_resolution is ignored in this case.
    :param random_seed: int, defaults to 42. Random seed to use.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.pycombo(G)

    :References:

    Sobolevsky, S., Campari, R., Belyi, A. and Ratti, C., 2014. General optimization technique for high-quality community detection in complex networks. Physical Review E, 90(1), p.012811.

    .. note:: Reference implementation: https://github.com/Casyfill/pyCombo
    """

    if pycombo_part is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install pycombo to use the selected feature."
        )

    g = convert_graph_formats(g_original, nx.Graph)
    partition = pycombo_part.execute(
        g,
        weight=weight,
        max_communities=max_communities,
        modularity_resolution=modularity_resolution,
        return_modularity=False,
        num_split_attempts=num_split_attempts,
        start_separate=start_separate,
        treat_as_modularity=treat_as_modularity,
        random_seed=random_seed,
    )

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in partition.items():
        coms_to_node[c].append(n)
    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms, g_original, "pyCombo", method_parameters={}, overlap=False
    )


def paris(g_original: object) -> NodeClustering:
    """
    Paris is a hierarchical graph clustering algorithm inspired by modularity-based clustering techniques.
    The algorithm is agglomerative and based on a simple distance between clusters induced by the probability of sampling node pairs.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.paris(G)

    :References:

    Bonald, T., Charpentier, B., Galland, A., & Hollocou, A. (2018). Hierarchical graph clustering using node pair sampling. arXiv preprint arXiv:1806.01664.

    .. note:: Reference implementation: https://github.com/tbonald/paris
    """
    g = convert_graph_formats(g_original, nx.Graph)
    D = paris_alg(g)
    clustering = paris_best_clustering(D)

    return NodeClustering(
        clustering, g_original, "Paris", method_parameters={}, overlap=False
    )


def ricci_community(
    g_original: object, alpha: float = 0.5, method: str = "Sinkhorn"
) -> NodeClustering:
    """
    Curvature is a geometric property to describe the local shape of an object. If we draw two parallel paths on a surface with positive curvature like a sphere, these two paths move closer to each other while for a negatively curved surface like a saddle, these two paths tend to be apart.
    Currently there are multiple ways to discretize curvature on graph, in this algorithm, we include two of the most frequently used discrete Ricci curvature: Ollivier-Ricci curvature which is based on optimal transportation theory and Forman-Ricci curvature which is base on CW complexes.
    Edge Ricci curvature is observed to play an important role in the graph structure.
    An edge with positive curvature represents an edge within a cluster, while a negatively curved edge tent to be a bridge within clusters.
    Also, negatively curved edges are highly related to graph connectivity, with negatively curved edges removed from a connected graph, the graph soon become disconnected.
    Ricci flow is a process to uniformized the edge Ricci curvature of the graph.
    For a given graph, the Ricci flow gives a "Ricci flow metric" on each edge as edge weights, such that under these edge weights, the Ricci curvature of the graph is mostly equal everywhere. In [Ni3], this "Ricci flow metric" is shown to be able to detect communities.
    Both Ricci curvature and Ricci flow metric can act as a graph fingerprint for graph classification.
    The different graph gives different edge Ricci curvature distributions and different Ricci flow metric.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param alpha: The parameter for the probability distribution, range from [0 ~ 1]. It means the share of mass to leave on the original node. Default, 0.5.
    :param method: Transportation method. [“OTD”, “ATD”, “Sinkhorn”]. Default: Sinkhorn
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.ricci_community(G)

    :References:

    Ni, C. C., Lin, Y. Y., Luo, F., & Gao, J. (2019). Community detection on networks with ricci flow. Scientific reports, 9(1), 1-12.

    .. note:: Reference implementation: https://github.com/saibalmars/GraphRicciCurvature
    """
    if OllivierRicci is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install GraphRicciCurvature to use the selected feature."
        )

    g = convert_graph_formats(g_original, nx.Graph)

    cricci = OllivierRicci(g, alpha=alpha, method=method)
    _, clustering = cricci.ricci_community()
    coms = defaultdict(list)

    for k, v in clustering.items():
        coms[v].append(k)

    return NodeClustering(
        list(coms.values()),
        g_original,
        "Ricci",
        method_parameters={"alpha": alpha, "method": method},
        overlap=False,
    )


def spectral(
    g_original: object,
    kmax: int,
    projection_on_smaller_class: bool = True,
    scaler: Callable = None,
) -> NodeClustering:
    """
    SCD implements a Spectral Clustering algorithm for Communities Discovery.
    It is based on Fielder’s vector (obtained from the eigenvector related to the second eigenvalue of the normalized Laplacian) that are leveraged to extract the communities using Kmeans clustering.
    SCD a hierarchical graph clustering algorithm inspired by modularity-based clustering techniques.
    The algorithm is agglomerative and based on a simple distance between clusters induced by the probability of sampling node pairs.

    :param g_original: a networkx/igraph object
    :param kmax: maximum number of desired communities
    :param projection_on_smaller_class: a boolean value that if True then it project a bipartite network in the smallest class of node. (default is True)
    :param scaler: the function to scale the fielder’s vector to apply KMeans
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.spectral(G)

    :References:

    Higham, Desmond J., Gabriela Kalna, and Milla Kibble. "Spectral clustering and its use in bioinformatics." Journal of computational and applied mathematics 204.1 (2007): 25-37.

    .. note:: Implementation provided by Gianmarco Pepi <g.pepi2@unipi.it>,  Monia Bennici <m.bennici4@studenti.unipi.it>,  Khashayar Abtin <k.abtin@studenti.unipi.it> and Kamran Mehravar <k.mehravar@studenti.unipi.it> (Computer Science Dept., University of Pisa, Italy)
    """

    from sklearn.preprocessing import StandardScaler

    if scaler is None:
        scaler = StandardScaler()

    g = convert_graph_formats(g_original, nx.Graph)
    clustering = spectral_communities(
        g,
        kmax=kmax,
        projection_on_smaller_class=projection_on_smaller_class,
        scaler=scaler,
    )

    return NodeClustering(
        clustering,
        g_original,
        "spectral",
        method_parameters={
            "kmax": kmax,
            "projection_on_smaller_class": projection_on_smaller_class,
        },
        overlap=False,
    )
