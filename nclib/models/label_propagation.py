from networkx.algorithms import community


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
    return  coms