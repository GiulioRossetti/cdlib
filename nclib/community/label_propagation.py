from nclib.community.algorithms.SLPA_nx import slpa_nx
from nclib.community.algorithms.multicom import MultiCom
from nclib.community.algorithms.Markov import markov
import networkx as nx
from nclib.utils import convert_graph_formats
from nclib import NodeClustering


def label_propagation(g):
    """

    :param g:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    lp = list(nx.algorithms.community.label_propagation_communities(g))
    coms = [tuple(x) for x in lp]

    return NodeClustering(coms, g, "Label Propagation")


def async_fluid(g, k):
    """

    :param g:
    :param k:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    fluid = nx.algorithms.community.asyn_fluidc(g, k)
    coms = [tuple(x) for x in fluid]
    return NodeClustering(coms, g, "Fluid")


def slpa(g, t=21, r=0.1):
    """

    :param g:
    :param t:
    :param r:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = slpa_nx(g, T=t, r=r)
    return NodeClustering(coms, g, "SLPA", method_parameters={"T": t, "r": r})


def multicom(g, seed_node):
    """

    :param g:
    :param seed_node:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    mc = MultiCom(g)
    coms = mc.execute(seed_node)
    return NodeClustering(coms, g, "Multicom", method_parameters={"seeds": seed_node})


def markov_clustering(g,  max_loop=1000):
    """

    :param g:
    :param max_loop:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = markov(g, max_loop)
    return NodeClustering(coms, g, "Markov Clustering", method_parameters={"max_loop": max_loop})