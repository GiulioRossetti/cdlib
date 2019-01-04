import pquality.PartitionQuality as pq
import networkx as nx
from nclib.utils import convert_graph_formats
from collections import namedtuple
import numpy as np
import scipy
from nclib.evaluation.scoring_functions.link_modularity import cal_modularity


Result = namedtuple('Result', ['min', 'max', 'mean', 'std'])


def link_modularity(graph, communities):
    """

    :param graph:
    :param communities:
    :return:
    """

    graph = convert_graph_formats(graph, nx.Graph)

    return cal_modularity(graph, communities)


def quality_indexes(graph, communities, scoring_function):
    """

    :param graph:
    :param communities:
    :param scoring_function:
    :return:
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in communities:
        community = nx.subgraph(graph, com)
        if scoring_function in [pq.average_internal_degree, pq.internal_edge_density,
                                pq.triangle_participation_ratio, pq.edges_inside, pq.fraction_over_median_degree]:
            values.append(scoring_function(community))
        else:
            values.append(scoring_function(graph, community))

    return Result(min=min(values), max=max(values), mean=np.mean(values), std=np.std(values))


def normalized_cut(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.normalized_cut)


def internal_edge_density(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.internal_edge_density)


def average_internal_degree(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.average_internal_degree)


def fraction_over_median_degree(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.fraction_over_median_degree)


def expansion(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.expansion)


def cut_ratio(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.cut_ratio)


def edges_inside(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.edges_inside)


def conductance(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.conductance)


def max_odf(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.max_odf)


def avg_odf(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.avg_odf)


def flake_odf(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.flake_odf)


def triangle_participation_ratio(graph, community):
    """

    :param graph:
    :param community:
    :return:
    """

    return quality_indexes(graph, community, pq.triangle_participation_ratio)


def newman_girvan_modularity(graph, communities):
    """

    Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).

    :param graph:
    :param communities:
    :return:
    """

    graph = convert_graph_formats(graph, nx.Graph)
    partition = {}
    for cid, com in enumerate(communities):
        for node in com:
            partition[node] = cid

    return pq.community_modularity(partition, graph)


def erdos_renyi_modularity(graph, communities):
    """

    Erdos, P., & Renyi, A. (1959).
    On random graphs I. Publ. Math. Debrecen, 6, 290-297.

    :param graph:
    :param communities:
    :return:
    """

    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    q = 0

    for community in communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        nc = c.number_of_nodes()
        q += mc - (m*nc*(nc - 1)) / (n*(n-1))

    return (1 / m) * q


def modularity_density(graph, communities):
    """

    Li, Z., Zhang, S., Wang, R. S., Zhang, X. S., & Chen, L. (2008).
    Quantitative function for community detection.
    Physical review E, 77(3), 036109.

    :param graph:
    :param communities:
    :return:
    """

    q = 0

    for community in communities:
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
    """

    Miyauchi, Atsushi, and Yasushi Kawase.
    "Z-score-based modularity for community detection in networks."
    PloS one 11.1 (2016): e0147805.

    :param graph:
    :param communities:
    :return:
    """

    m = graph.number_of_edges()

    mmc = 0
    dc2m = 0

    for community in communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        dc = 0

        for node in c:
            dc += c.degree(node)

        mmc += (mc/m)
        dc2m += (dc/(2*m))**2

    return (mmc - dc2m) / np.sqrt(dc2m * (1 - dc2m))


def surprise(graph, communities):
    """

    Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015).
    Detecting communities using asymptotical surprise.
    Physical Review E, 92(2), 022816.

    :param graph:
    :param communities:
    :return:
    """

    m = graph.number_of_edges()
    n = graph.number_of_nodes()

    q = 0
    qa = 0

    for community in communities:
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
    """

    Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015).
    Detecting communities using asymptotical surprise.
    Physical Review E, 92(2), 022816.

    :param graph:
    :param communities:
    :return:
    """

    m = graph.number_of_edges()

    binom = scipy.special.comb(m, 2, exact=True)
    p = m/binom

    q = 0

    for community in communities:
        c = nx.subgraph(graph, community)
        nc = c.number_of_nodes()
        mc = c.number_of_edges()

        binom_c = scipy.special.comb(nc, 2, exact=True)
        pc = mc / binom_c

        q += binom_c * (pc * np.log(pc/p) + (1-pc)*np.log((1-pc)/(1-p)))
    return q
