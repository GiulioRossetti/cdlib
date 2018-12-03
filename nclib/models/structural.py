from networkx.algorithms import community
from nclib.models.algorithms.em import EM_nx
from nclib.models.algorithms.lfm import LFM_nx
from nclib.models.algorithms.scan import SCAN_nx


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


def EM(g, k):
    algorithm = EM_nx(g, k)
    coms = algorithm.execute()
    return coms


def LFM(g, alpha):
    algorithm = LFM_nx(g, alpha)
    coms = algorithm.execute()
    return coms


def SCAN(g, epsilon, mu):
    algorithm = SCAN_nx(g, epsilon, mu)
    coms = algorithm.execute()
    return coms