from networkx.algorithms import community
from networkx import edge_betweenness_centrality as betweenness


def kclique(g, k):
    """

    :param g:
    :param k:
    :return:
    """
    kc = list(community.k_clique_communities(g, k))
    coms = [tuple(x) for x in kc]
    return coms


def girvan_newman(g, level):
    """

    :param g:
    :param level:
    :return:
    """

    gn_hierarchy = community.girvan_newman(g)
    coms = []
    for _ in range(level):
        coms = next(gn_hierarchy)

    return list(coms)
