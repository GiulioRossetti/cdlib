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


def girvan_newman(g, level, most_valuable_edge=None):
    """

    :param g:
    :param level:
    :param most_valuable_edge:
    :return:
    """
    if most_valuable_edge is None:
        most_valuable_edge = betweenness

    gn_hierarchy = community.girvan_newman(g, most_valuable_edge)
    coms = []
    for _ in range(level):
        coms = next(gn_hierarchy)
    return coms
