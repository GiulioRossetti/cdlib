from nclib.community.algorithms import DER


def der(graph, walck_len=3, threshold=.00001, iterbound=50):
    """

    :param graph:
    :param walck_len:
    :param threshold:
    :param iterbound:
    :return:
    """

    communities, _ = DER.der_graph_clustering(graph, walk_len=walck_len,
                                              alg_threshold=threshold, alg_iterbound=iterbound)

    return communities
