import pquality as pq
import networkx as nx
from cdlib.utils import convert_graph_formats
from collections import namedtuple
import numpy as np
import scipy
from cdlib.evaluation.internal.link_modularity import cal_modularity
import Eva

__all__ = ["FitnessResult", "link_modularity", "normalized_cut", "internal_edge_density", "average_internal_degree",
           "fraction_over_median_degree", "expansion", "cut_ratio", "edges_inside", "flake_odf", "avg_odf", "max_odf",
           "triangle_participation_ratio", "modularity_density", "z_modularity", "erdos_renyi_modularity",
           "newman_girvan_modularity", "significance", "surprise", "conductance", "size", "avg_embeddedness",
           "scaled_density", "avg_distance", "hub_dominance", "avg_transitivity", "purity"]

# FitnessResult = namedtuple('FitnessResult', ['min', 'max', 'mean', 'std'])
FitnessResult = namedtuple('FitnessResult', 'min max score std')
FitnessResult.__new__.__defaults__ = (None,) * len(FitnessResult._fields)


def __quality_indexes(graph, communities, scoring_function, summary=True):
    """

    :param graph: NetworkX/igraph graph
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-communitys ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in communities.communities:
        community = nx.subgraph(graph, com)
        if scoring_function in [pq.PartitionQuality.average_internal_degree, pq.PartitionQuality.internal_edge_density,
                                pq.PartitionQuality.triangle_participation_ratio, pq.PartitionQuality.edges_inside,
                                pq.PartitionQuality.fraction_over_median_degree]:
            values.append(scoring_function(community))
        else:
            values.append(scoring_function(graph, community))

    if summary:
        return FitnessResult(min=min(values), max=max(values), score=np.mean(values), std=np.std(values))
    return values


def size(graph, communities, **kwargs):
    """Size is the number of nodes in the community

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> sz = evaluation.size(g,communities)
    """

    return __quality_indexes(graph, communities, lambda g, com: len(com), **kwargs)


def scaled_density(graph, communities, **kwargs):
    """Scaled density.

    The scaled density of a community is defined as the ratio of the community density w.r.t. the complete graph density.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> scd = evaluation.scaled_density(g,communities)
    """

    return __quality_indexes(graph, communities,
                             lambda graph, coms: nx.density(nx.subgraph(graph, coms)) / nx.density(graph), **kwargs)


def avg_distance(graph, communities, **kwargs):
    """Average distance.

    The average distance of a community is defined average path length across all possible pair of nodes composing it.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> scd = evaluation.avg_distance(g,communities)
    """

    return __quality_indexes(graph, communities,
                             lambda graph, coms: nx.average_shortest_path_length(nx.subgraph(graph, coms)), **kwargs)


def hub_dominance(graph, communities, **kwargs):
    """Hub dominance.

    The hub dominance of a community is defined as the ratio of the degree of its most connected node w.r.t. the theoretically maximal degree within the community.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> scd = evaluation.hub_dominance(g,communities)
    """

    return __quality_indexes(graph, communities,
                             lambda graph, coms: max([x[1] for x in
                                                      list(nx.degree(nx.subgraph(graph, coms)))]) / (len(coms) - 1),
                             **kwargs)


def avg_transitivity(graph, communities, **kwargs):
    """Average transitivity.

    The average transitivity of a community is defined the as the average clustering coefficient of its nodes w.r.t. their connection within the community itself.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> scd = evaluation.avg_transitivity(g,communities)
    """

    return __quality_indexes(graph, communities,
                             lambda graph, coms: nx.average_clustering(nx.subgraph(graph, coms)),
                             **kwargs)


def avg_embeddedness(graph, communities, **kwargs):
    """Average embeddedness of nodes within the community.

    The embeddedness of a node n w.r.t. a community C is the ratio of its degree within the community and its overall degree.

    .. math:: emb(n,C) = \\frac{k_n^C}{k_n}

    The average embeddedness of a community C is:

    .. math:: avg_embd(c) = \\frac{1}{|C|} \sum_{i \in C} \\frac{k_n^C}{k_n}

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> ave = evaluation.avg_embeddedness(g,communities)

    :References:


    """

    return __quality_indexes(graph, communities,
                             lambda g, com:
                             np.mean([float(nx.degree(nx.subgraph(g, com))[n]) / nx.degree(g)[n] for n in com]),
                             **kwargs)


def normalized_cut(graph, community, **kwargs):
    """Normalized variant of the Cut-Ratio

    .. math:: f(S) = \\frac{c_S}{2m_S+c_S} + \\frac{c_S}{2(m−m_S )+c_S}

    where :math:`m` is the number of graph edges, :math:`m_S` is the number of community internal edges and :math:`c_S` is the number of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.


    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.normalized_cut(g,communities)

    :References:

    1.Shi, J., Malik, J.: Normalized cuts and image segmentation. Departmental Papers (CIS), 107 (2000)
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.normalized_cut, **kwargs)


def internal_edge_density(graph, community, **kwargs):
    """The internal density of the community set.

     .. math:: f(S) = \\frac{m_S}{n_S(n_S−1)/2}

     where :math:`m_S` is the number of community internal edges and :math:`n_S` is the number of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.internal_edge_density(g,communities)


    :References:

    1. Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.internal_edge_density, **kwargs)


def average_internal_degree(graph, community, **kwargs):
    """The average internal degree of the community set.

    .. math:: f(S) = \\frac{2m_S}{n_S}

     where :math:`m_S` is the number of community internal edges and :math:`n_S` is the number of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.average_internal_degree(g,communities)

    :References:

    1. Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.average_internal_degree, **kwargs)


def fraction_over_median_degree(graph, community, **kwargs):
    """Fraction of community nodes of having internal degree higher than the median degree value.

    .. math:: f(S) = \\frac{|\{u: u \\in S,| \{(u,v): v \\in S\}| > d_m\}| }{n_S}


    where :math:`d_m` is the internal degree median value

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.fraction_over_median_degree(g,communities)

    :References:

    1. Yang, J., Leskovec, J.: Defining and evaluating network communities based on ground-truth. Knowledge and Information Systems 42(1), 181–213 (2015)
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.fraction_over_median_degree, **kwargs)


def expansion(graph, community, **kwargs):
    """Number of edges per community node that point outside the cluster.

    .. math:: f(S) = \\frac{c_S}{n_S}

    where :math:`n_S` is the number of edges on the community boundary, :math:`c_S` is the number of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.expansion(g,communities)

    :References:

    1. Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.expansion, **kwargs)


def cut_ratio(graph, community, **kwargs):
    """Fraction of existing edges (out of all possible edges) leaving the community.

    ..math:: f(S) = \\frac{c_S}{n_S (n − n_S)}

    where :math:`c_S` is the number of community nodes and, :math:`n_S` is the number of edges on the community boundary

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.cut_ratio(g,communities)

    :References:

    1. Fortunato, S.: Community detection in graphs. Physics reports 486(3-5), 75–174 (2010)
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.cut_ratio, **kwargs)


def edges_inside(graph, community, **kwargs):
    """Number of edges internal to the community.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.edges_inside(g,communities)

    :References:

    1. Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.edges_inside, **kwargs)


def conductance(graph, community, **kwargs):
    """ Fraction of total edge volume that points outside the community.

    .. math:: f(S) = \\frac{c_S}{2 m_S+c_S}

    where :math:`c_S` is the number of community nodes and, :math:`m_S` is the number of community edges

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.conductance(g,communities)

    :References:

    1.Shi, J., Malik, J.: Normalized cuts and image segmentation. Departmental Papers (CIS), 107 (2000)
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.conductance, **kwargs)


def max_odf(graph, community, **kwargs):
    """Maximum fraction of edges of a node of a community that point outside the community itself.

    .. math:: max_{u \\in S} \\frac{|\{(u,v)\\in E: v \\not\\in S\}|}{d(u)}

    where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S` and :math:`d(u)` is the degree of :math:`u`

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.max_odf(g,communities)

    :References:

    1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.max_odf, **kwargs)


def avg_odf(graph, community, **kwargs):
    """Average fraction of edges of a node of a community that point outside the community itself.

    .. math:: \\frac{1}{n_S} \\sum_{u \\in S} \\frac{|\{(u,v)\\in E: v \\not\\in S\}|}{d(u)}

    where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S`, :math:`d(u)` is the degree of :math:`u` and :math:`n_S` is the set of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.avg_odf(g,communities)

    :References:

    1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.avg_odf, **kwargs)


def flake_odf(graph, community, **kwargs):
    """Fraction of nodes in S that have fewer edges pointing inside than to the outside of the community.

    .. math:: f(S) = \\frac{| \{ u:u \in S,| \{(u,v) \in E: v \in S \}| < d(u)/2 \}|}{n_S}

    where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S`, :math:`d(u)` is the degree of :math:`u` and :math:`n_S` is the set of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.flake_odf(g,communities)

    :References:

    1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.flake_odf, **kwargs)


def triangle_participation_ratio(graph, community, **kwargs):
    """Fraction of community nodes that belong to a triad.

    .. math:: f(S) = \\frac{ | \{ u: u \in S,\{(v,w):v, w \in S,(u,v) \in E,(u,w) \in E,(v,w) \in E \} \\not = \\emptyset \} |}{n_S}

    where :math:`n_S` is the set of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.triangle_participation_ratio(g,communities)

    :References:

    1. Yang, J., Leskovec, J.: Defining and evaluating network communities based on ground-truth. Knowledge and Information Systems 42(1), 181–213 (2015)
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.triangle_participation_ratio, **kwargs)


def link_modularity(graph, communities, **kwargs):
    """
    Quality function designed for directed graphs with overlapping communities.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.link_modularity(g,communities)

    :References:

    1. Nicosia, V., Mangioni, G., Carchiolo, V., Malgeri, M.: Extending the definition of modularity to directed graphs with overlapping communities. Journal of Statistical Mechanics: Theory and Experiment 2009(03), 03024 (2009)

    """

    graph = convert_graph_formats(graph, nx.Graph)

    return FitnessResult(score=cal_modularity(graph, communities.communities))


def newman_girvan_modularity(graph, communities, **kwargs):
    """Difference the fraction of intra community edges of a partition with the expected number of such edges if distributed according to a null model.

    In the standard version of modularity, the null model preserves the expected degree sequence of the graph under consideration. In other words, the modularity compares the real network structure with a corresponding one where nodes are connected without any preference about their neighbors.

    .. math:: Q(S) = \\frac{1}{m}\\sum_{c \\in S}(m_S - \\frac{(2 m_S + l_S)^2}{4m})

    where :math:`m` is the number of graph edges, :math:`m_S` is the number of community edges, :math:`l_S` is the number of edges from nodes in S to nodes outside S.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object


    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.newman_girvan_modularity(g,communities)

    :References:

    1. Newman, M.E.J. & Girvan, M. `Finding and evaluating community structure in networks. <https://www.ncbi.nlm.nih.gov/pubmed/14995526/>`_ Physical Review E 69, 26113(2004).
    """

    graph = convert_graph_formats(graph, nx.Graph)
    partition = {}
    for cid, com in enumerate(communities.communities):
        for node in com:
            partition[node] = cid

    return FitnessResult(score=pq.PartitionQuality.community_modularity(partition, graph))


def erdos_renyi_modularity(graph, communities, **kwargs):
    """Erdos-Renyi modularity is a variation of the Newman-Girvan one.
    It assumes that vertices in a network are connected randomly with a constant probability :math:`p`.

    .. math:: Q(S) = \\frac{1}{m}\\sum_{c \\in S} (m_S − \\frac{mn_S(n_S −1)}{n(n−1)})

    where :math:`m` is the number of graph edges, :math:`m_S` is the number of community edges, :math:`l_S` is the number of edges from nodes in S to nodes outside S.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.erdos_renyi_modularity(g,communities)

    :References:

    1. Erdos, P., & Renyi, A. (1959). `On random graphs I. <https://gnunet.org/sites/default/files/Erd%C5%91s%20%26%20R%C3%A9nyi%20-%20On%20Random%20Graphs.pdf/>`_ Publ. Math. Debrecen, 6, 290-297.
    """
    graph = convert_graph_formats(graph, nx.Graph)
    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    q = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        nc = c.number_of_nodes()
        q += mc - (m * nc * (nc - 1)) / (n * (n - 1))

    return FitnessResult(score=(1 / m) * q)


def modularity_density(graph, communities, **kwargs):
    """The modularity density is one of several propositions that envisioned to palliate the resolution limit issue of modularity based measures.
    The idea of this metric is to include the information about community size into the expected density of community to avoid the negligence of small and dense communities.
    For each community :math:`C` in partition :math:`S`, it uses the average modularity degree calculated by :math:`d(C) = d^{int(C)} − d^{ext(C)}` where :math:`d^{int(C)}` and :math:`d^{ext(C)}` are the average internal and external degrees of :math:`C` respectively to evaluate the fitness of :math:`C` in its network.
    Finally, the modularity density can be calculated as follows:

    .. math:: Q(S) = \\sum_{C \\in S} \\frac{1}{n_C} ( \\sum_{i \\in C} k^{int}_{iC} - \\sum_{i \\in C} k^{out}_{iC})

    where :math:`n_C` is the number of nodes in C, :math:`k^{int}_{iC}` is the degree of node i within :math:`C` and :math:`k^{out}_{iC}` is the deree of node i outside :math:`C`.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object


    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.modularity_density(g,communities)

    :References:

    1. Li, Z., Zhang, S., Wang, R. S., Zhang, X. S., & Chen, L. (2008). `Quantitative function for community detection. <https://www.sciencedirect.com/science/article/pii/S0020025516305059/>`_ Physical review E, 77(3), 036109.
    """

    q = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)

        nc = c.number_of_nodes()
        dint = []
        dext = []
        for node in c:
            dint.append(c.degree(node))
            dext.append(graph.degree(node) - c.degree(node))

        try:
            q += (1 / nc) * (np.mean(dint) - np.mean(dext))
        except ZeroDivisionError:
            pass

    return FitnessResult(score=q)


def z_modularity(graph, communities, **kwargs):
    """Z-modularity is another variant of the standard modularity proposed to avoid the resolution limit.
    The concept of this version is based on an observation that the difference between the fraction of edges inside communities and the expected number of such edges in a null model should not be considered as the only contribution to the final quality of community structure.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.z_modularity(g,communities)


    :References:

    1. Miyauchi, Atsushi, and Yasushi Kawase. `Z-score-based modularity for community detection in networks. <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147805/>`_ PloS one 11.1 (2016): e0147805.
    """

    m = graph.number_of_edges()

    mmc = 0
    dc2m = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        dc = 0

        for node in c:
            dc += graph.degree(node)

        mmc += (mc / m)
        dc2m += (dc / (2 * m)) ** 2

    res = 0
    try:
        res = (mmc - dc2m) / np.sqrt(dc2m * (1 - dc2m))
    except ZeroDivisionError:
        pass

    return FitnessResult(score=res)


def surprise(graph, communities, **kwargs):
    """Surprise is statistical approach proposes a quality metric assuming that edges between vertices emerge randomly according to a hyper-geometric distribution.

    According to the Surprise metric, the higher the score of a partition, the less likely it is resulted from a random realization, the better the quality of the community structure.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.surprise(g,communities)

    :References:

    1. Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015). `Detecting communities using asymptotical surprise. <https://link.aps.org/doi/10.1103/PhysRevE.92.022816/>`_ Physical Review E, 92(2), 022816.
    """

    m = graph.number_of_edges()
    n = graph.number_of_nodes()

    q = 0
    qa = 0
    sp = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        nc = c.number_of_nodes()

        q += mc
        qa += scipy.special.comb(nc, 2, exact=True)
    try:
        q = q / m
        qa = qa / scipy.special.comb(n, 2, exact=True)

        sp = m * (q * np.log(q / qa) + (1 - q) * np.log2((1 - q) / (1 - qa)))
    except ZeroDivisionError:
        pass

    return FitnessResult(score=sp)


def significance(graph, communities, **kwargs):
    """Significance estimates how likely a partition of dense communities appear in a random graph.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.significance(g,communities)

    :References:

    1. Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015). `Detecting communities using asymptotical surprise. <https://link.aps.org/doi/10.1103/PhysRevE.92.022816/>`_ Physical Review E, 92(2), 022816.
    """

    m = graph.number_of_edges()

    binom = scipy.special.comb(m, 2, exact=True)
    p = m / binom

    q = 0

    for community in communities.communities:
        try:
            c = nx.subgraph(graph, community)
            nc = c.number_of_nodes()
            mc = c.number_of_edges()

            binom_c = scipy.special.comb(nc, 2, exact=True)
            pc = mc / binom_c

            q += binom_c * (pc * np.log(pc / p) + (1 - pc) * np.log((1 - pc) / (1 - p)))
        except ZeroDivisionError:
            pass
    return FitnessResult(score=q)


def purity(communities):
    """Purity is the product of the frequencies of the most frequent labels carried by the nodes within the communities

        :param communities: AttrNodeClustering object
        :return: FitnessResult object

        Example:

        >>> from cdlib.algorithms import eva
        >>> from cdlib import evaluation
        >>> import random
        >>> l1 = ['A', 'B', 'C', 'D']
        >>> l2 = ["E", "F", "G"]
        >>> g = nx.barabasi_albert_graph(100, 5)
        >>> labels=dict()
        >>> for node in g.nodes():
        >>>    labels[node]={"l1":random.choice(l1), "l2":random.choice(l2)}
        >>> communities = eva(g_attr, labels, alpha=0.5)
        >>> pur = evaluation.purity(communities)

        :References:

        1. Citraro, Salvatore, and Giulio Rossetti. "Eva: Attribute-Aware Network Segmentation." International Conference on Complex Networks and Their Applications. Springer, Cham, 2019.
        """

    pur = Eva.purity(communities.coms_labels)
    return FitnessResult(score=pur)
