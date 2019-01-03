import pquality.PartitionQuality as pq
import networkx as nx
from nclib.utils import convert_graph_formats
from collections import namedtuple
import numpy as np
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


def modularity(graph, communities):
    """

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



