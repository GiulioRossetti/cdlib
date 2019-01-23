import pquality as pq
import networkx as nx
from nclib.utils import convert_graph_formats
from collections import namedtuple
import numpy as np
import scipy
from nclib.evaluation.internal.link_modularity import cal_modularity

__all__ = ["FitnessResult", "link_modularity", "normalized_cut", "internal_edge_density", "average_internal_degree",
           "fraction_over_median_degree", "expansion", "cut_ratio", "edges_inside", "flake_odf", "avg_odf", "max_odf",
           "triangle_participation_ratio", "modularity_density", "z_modularity", "erdos_renyi_modularity",
           "newman_girvan_modularity", "significance", "surprise", "conductance", "size"]


FitnessResult = namedtuple('FitnessResult', ['min', 'max', 'mean', 'std'])


def __quality_indexes(graph, communities, scoring_function, summary=True):
    """

    :param graph:
    :param communities:
    :param scoring_function:
    :return:
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
        return FitnessResult(min=min(values), max=max(values), mean=np.mean(values), std=np.std(values))
    return values


def normalized_cut(graph, community, **kwargs):
    """Normalized variant of the Cut-Ratio

    .. math:: : f(S) = \\frac{c_S}{2m_S+c_S} + \\frac{c_S}{2(m−m_S )+c_S}

    where :math:`m` is the number of graph edges, :math:`m_S` is the number of algorithms internal edges and :math:`c_S` is the number of algorithms nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.normalized_cut, **kwargs)


def internal_edge_density(graph, community, **kwargs):
    """The internal density of the algorithms set.

     .. math:: f(S) = \\frac{m_S}{n_S(n_S−1)/2}

     where :math:`m_S` is the number of algorithms internal edges and :math:`n_S` is the number of algorithms nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.internal_edge_density, **kwargs)


def average_internal_degree(graph, community, **kwargs):
    """The average internal degree of the algorithms set.

    .. math:: f(S) = \\frac{2m_S}{n_S}

     where :math:`m_S` is the number of algorithms internal edges and :math:`n_S` is the number of algorithms nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.average_internal_degree, **kwargs)


def fraction_over_median_degree(graph, community, **kwargs):
    """Fraction of algorithms nodes of having internal degree higher than the median degree value.

    .. math:: f(S) = \\frac{|\{u: u \\in S,| \{(u,v): v \\in S\}| > d_m\}| }{n_S}


    where :math:`d_m` is the internal degree median value

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.fraction_over_median_degree, **kwargs)


def expansion(graph, community, **kwargs):
    """Number of edges per algorithms node that point outside the cluster.

    .. math:: f(S) = \\frac{c_S}{n_S}

    where :math:`n_S` is the number of edges on the algorithms boundary, :math:`c_S` is the number of algorithms nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.expansion, **kwargs)


def cut_ratio(graph, community, **kwargs):
    """Fraction of existing edges (out of all possible edges) leaving the algorithms.

    ..math:: f(S) = \\frac{c_S}{n_S (n − n_S)}

    where :math:`c_S` is the number of algorithms nodes and, :math:`n_S` is the number of edges on the algorithms boundary

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.cut_ratio, **kwargs)


def edges_inside(graph, community, **kwargs):
    """Number of edges internal to the algorithms.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.edges_inside, **kwargs)


def conductance(graph, community, **kwargs):
    """ Fraction of total edge volume that points outside the algorithms.

    .. math:: f(S) = \\frac{c_S}{2 m_S+c_S}

    where :math:`c_S` is the number of algorithms nodes and, :math:`m_S` is the number of algorithms edges

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.conductance, **kwargs)


def max_odf(graph, community, **kwargs):
    """Maximum fraction of edges of a node of a algorithms that point outside the algorithms itself.

    .. math:: max_{u \\in S} \\frac{|\{(u,v)\\in E: v \\not\\in S\}|}{d(u)}

    where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S` and :math:`d(u)` is the degree of :math:`u`

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.max_odf, **kwargs)


def avg_odf(graph, community, **kwargs):
    """Average fraction of edges of a node of a algorithms that point outside the algorithms itself.

    .. math:: \\frac{1}{n_S} \\sum_{u \\in S} \\frac{|\{(u,v)\\in E: v \\not\\in S\}|}{d(u)}

    where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S`, :math:`d(u)` is the degree of :math:`u` and :math:`n_S` is the set of algorithms nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.avg_odf, **kwargs)


def flake_odf(graph, community, **kwargs):
    """Fraction of nodes in S that have fewer edges pointing inside than to the outside of the algorithms.

    .. math:: f(S) = \\frac{| \{ u:u \in S,| \{(u,v) \in E: v \in S \}| < d(u)/2 \}|}{n_S}

    where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S`, :math:`d(u)` is the degree of :math:`u` and :math:`n_S` is the set of algorithms nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.flake_odf, **kwargs)


def triangle_participation_ratio(graph, community, **kwargs):
    """Fraction of algorithms nodes that belong to a triad.

    .. math:: f(S) = \\frac{ | \{ u: u \in S,\{(v,w):v, w \in S,(u,v) \in E,(u,w) \in E,(v,w) \in E \} \\not = \\emptyset \} |}{n_S}

    where :math:`n_S` is the set of algorithms nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-algorithms ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.
    """

    return __quality_indexes(graph, community, pq.PartitionQuality.triangle_participation_ratio, **kwargs)


def link_modularity(graph, communities):
    """Quality function designed for directed graphs with overlapping communities.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: the link modularity score


    :References:

    1.
    """

    graph = convert_graph_formats(graph, nx.Graph)

    return cal_modularity(graph, communities.communities)


def newman_girvan_modularity(graph, communities):
    """Difference the fraction of intra algorithms edges of a partition with the expected number of such edges if distributed according to a null model.

    In the standard version of modularity, the null model preserves the expected degree sequence of the graph under consideration. In other words, the modularity compares the real network structure with a corresponding one where nodes are connected without any preference about their neighbors.

    .. math:: Q(S) = \\frac{1}{m}\\sum_{c \\in S}(m_S - \\frac{(2 m_S + l_S)^2}{4m})

    where :math:`m` is the number of graph edges, :math:`m_S` is the number of algorithms edges, :math:`l_S` is the number of edges from nodes in S to nodes outside S.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: the Newman-Girvan modularity score

    :References:

    1. Newman, M.E.J. & Girvan, M. **Finding and evaluating algorithms structure in networks.** Physical Review E 69, 26113(2004).
    """

    graph = convert_graph_formats(graph, nx.Graph)
    partition = {}
    for cid, com in enumerate(communities.communities):
        for node in com:
            partition[node] = cid

    return pq.PartitionQuality.community_modularity(partition, graph)


def erdos_renyi_modularity(graph, communities):
    """Erdos-Renyi modularity is a variation of the Newman-Girvan one.
    It assumes that vertices in a network are connected randomly with a constant probability :math:`p`.

    .. math:: Q(S) = \\frac{1}{m}\\sum_{c \\in S} (m_S − \\frac{mn_S(n_S −1)}{n(n−1)})

    where :math:`m` is the number of graph edges, :math:`m_S` is the number of algorithms edges, :math:`l_S` is the number of edges from nodes in S to nodes outside S.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: the Erdos-Renyi modularity score

    :References:

    1. Erdos, P., & Renyi, A. (1959). **On random graphs I.** Publ. Math. Debrecen, 6, 290-297.
    """

    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    q = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        nc = c.number_of_nodes()
        q += mc - (m*nc*(nc - 1)) / (n*(n-1))

    return (1 / m) * q


def modularity_density(graph, communities):
    """The modularity density is one of several propositions that envisioned to palliate the resolution limit issue of modularity based measures.
    The idea of this metric is to include the information about algorithms size into the expected density of algorithms to avoid the negligence of small and dense communities.
    For each algorithms :math:`C` in partition :math:`S`, it uses the average modularity degree calculated by :math:`d(C) = d^{int(C)} − d^{ext(C)}` where :math:`d^{int(C)}` and :math:`d^{ext(C)}` are the average internal and external degrees of :math:`C` respectively to evaluate the fitness of :math:`C` in its network.
    Finally, the modularity density can be calculated as follows:

    .. math:: Q(S) = \\sum_{C \\in S} \\frac{1}{n_C} ( \\sum_{i \\in C} k^{int}_{iC} - \\sum_{i \\in C} k^{out}_{iC})

    where :math:`n_C` is the number of nodes in C, :math:`k^{int}_{iC}` is the degree of node i within :math:`C` and :math:`k^{out}_{iC}` is the deree of node i outside :math:`C`.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: the modularity density score

    :References:

    1. Li, Z., Zhang, S., Wang, R. S., Zhang, X. S., & Chen, L. (2008). **Quantitative function for algorithms detection.** Physical review E, 77(3), 036109.
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

        q += (1 / nc) * (np.mean(dint) - np.mean(dext))

    return q


def z_modularity(graph, communities):
    """Z-modularity is another variant of the standard modularity proposed to avoid the resolution limit.
    The concept of this version is based on an observation that the difference between the fraction of edges inside communities and the expected number of such edges in a null model should not be considered as the only contribution to the final quality of algorithms structure.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: the z-modularity score


    :References:

    1. Miyauchi, Atsushi, and Yasushi Kawase. **Z-score-based modularity for algorithms detection in networks.** PloS one 11.1 (2016): e0147805.
    """

    m = graph.number_of_edges()

    mmc = 0
    dc2m = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        dc = 0

        for node in c:
            dc += c.degree(node)

        mmc += (mc/m)
        dc2m += (dc/(2*m))**2

    return (mmc - dc2m) / np.sqrt(dc2m * (1 - dc2m))


def surprise(graph, communities):
    """Surprise is statistical approach proposes a quality metric assuming that edges between vertices emerge randomly according to a hyper-geometric distribution.S
    According to the Surprise metric, the higher the score of a partition, the less likely it is resulted from a random realization, the better the quality of the algorithms structure.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: the surprise score

    :References:

    1. Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015). **Detecting communities using asymptotical surprise.** Physical Review E, 92(2), 022816.
    """

    m = graph.number_of_edges()
    n = graph.number_of_nodes()

    q = 0
    qa = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        nc = c.number_of_nodes()

        q += mc
        qa += scipy.special.comb(nc, 2, exact=True)

    q = q/m
    qa = qa/scipy.special.comb(n, 2, exact=True)

    sp = m*(q*np.log(q/qa) + (1-q)*np.log2((1-q)/(1-qa)))
    return sp


def significance(graph, communities):
    """Significance estimates how likely a partition of dense communities appear in a random graph.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: the significance score

    :References:

    1. Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015). **Detecting communities using asymptotical surprise.** Physical Review E, 92(2), 022816.
    """

    m = graph.number_of_edges()

    binom = scipy.special.comb(m, 2, exact=True)
    p = m/binom

    q = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        nc = c.number_of_nodes()
        mc = c.number_of_edges()

        binom_c = scipy.special.comb(nc, 2, exact=True)
        pc = mc / binom_c

        q += binom_c * (pc * np.log(pc/p) + (1-pc)*np.log((1-pc)/(1-p)))
    return q

def size(graph, communities, **kwargs):
    """Size is the number of nodes in the community

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :return: the size
    """


    return __quality_indexes(graph, communities, lambda graph, com: len(com), **kwargs)