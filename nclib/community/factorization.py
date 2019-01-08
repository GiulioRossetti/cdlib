from nclib.community.algorithms import DER
from nclib.utils import convert_graph_formats
import networkx as nx


def der(graph, walck_len=3, threshold=.00001, iterbound=50):
    """

    :param graph:
    :param walck_len:
    :param threshold:
    :param iterbound:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    communities, _ = DER.der_graph_clustering(g, walk_len=walck_len,
                                              alg_threshold=threshold, alg_iterbound=iterbound)

    return communities
