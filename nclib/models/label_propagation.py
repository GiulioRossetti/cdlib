from networkx.algorithms import community
from nclib.models.algorithms.SLPA_nx import slpa_nx


def label_propagation(g):
    """

    :param g:
    :return:
    """
    lp = list(community.label_propagation_communities(g))
    coms = [tuple(x) for x in lp]
    return coms


def async_fluid(g, k):
    """

    :param g:
    :param k:
    :return:
    """

    fluid = community.asyn_fluidc(g, k)
    coms = [tuple(x) for x in fluid]
    return coms


def SLPA(g, t=21, r=0.1):
    """

    :param g:
    :param t:
    :param r:
    :return:
    """
    coms = slpa_nx(g, T=t, r=r)
    return coms