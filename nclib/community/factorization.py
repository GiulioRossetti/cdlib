from nclib.community.algorithms import DER
from nclib.community.algorithms import BIGCLAM
from nclib.utils import convert_graph_formats
from nclib import NodeClustering
import networkx as nx


def der(graph, walk_len=3, threshold=.00001, iter_bound=50):
    """

    :param graph:
    :param walk_len:
    :param threshold:
    :param iter_bound:
    :return:
    """

    graph = convert_graph_formats(graph, nx.Graph)

    communities, _ = DER.der_graph_clustering(graph, walk_len=walk_len,
                                              alg_threshold=threshold, alg_iterbound=iter_bound)

    return NodeClustering(communities, graph, "DER", method_parameters={"walk_len": walk_len, "threshold": threshold,
                                                                        "iter_bound": iter_bound})


def big_clam(g, number_communities=5):
    """

    :param g: the graph
    :param number_communities: number communities desired
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    communities = BIGCLAM.big_Clam(g, number_communities)

    return NodeClustering(communities, g, "BigClam", method_parameters={"number_communities": number_communities})
