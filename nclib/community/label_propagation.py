from networkx.algorithms import community
from nclib.community.algorithms.SLPA_nx import slpa_nx
from nclib.community.algorithms.multicom import MultiCom
import networkx as nx
from nclib.utils import convert_graph_formats


def label_propagation(g):
    """

    :param g:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    lp = list(community.label_propagation_communities(g))
    coms = [tuple(x) for x in lp]
    return coms


def async_fluid(g, k):
    """

    :param g:
    :param k:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

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

    g = convert_graph_formats(g, nx.Graph)

    coms = slpa_nx(g, T=t, r=r)
    return coms


def multicom(g, seed_node):
    """

    :param g:
    :param seed_node:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    mc = MultiCom(g)
    coms = mc.execute(seed_node)
    return coms