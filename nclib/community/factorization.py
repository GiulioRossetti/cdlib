from nclib.community.algorithms import DER
from nclib.utils import convert_graph_formats
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

    return communities



def bigClam(g, number_communities=5):
    """

    :param graph:
    :param walck_len:
    :param threshold:
    :param iterbound:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    communities, _ = bigClam.big_Clam(g,number_communities)


    return communities
