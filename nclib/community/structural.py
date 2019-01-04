from networkx.algorithms import community
from nclib.community.algorithms.em import EM_nx
from nclib.community.algorithms.lfm import LFM_nx
from nclib.community.algorithms.scan import SCAN_nx
from nclib.community.algorithms.HLC import *
import networkx as nx
from nclib.utils import convert_graph_formats


def kclique(g, k):
    """

    :param g:
    :param k:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    kc = list(community.k_clique_communities(g, k))
    coms = [tuple(x) for x in kc]
    return coms


def girvan_newman(g, level):
    """

    :param g:
    :param level:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    gn_hierarchy = community.girvan_newman(g)
    coms = []
    for _ in range(level):
        coms = next(gn_hierarchy)

    return list(coms)


def EM(g, k):
    """

    :param g:
    :param k:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    algorithm = EM_nx(g, k)
    coms = algorithm.execute()
    return coms


def LFM(g, alpha):
    """

    :param g:
    :param alpha:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    algorithm = LFM_nx(g, alpha)
    coms = algorithm.execute()
    return coms


def SCAN(g, epsilon, mu):
    """

    :param g:
    :param epsilon:
    :param mu:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    algorithm = SCAN_nx(g, epsilon, mu)
    coms = algorithm.execute()
    return coms


def HierarchicalLinkCommunity(g, threshold=None, weighted=False):
    """

    :param g:
    :param threshold:
    :param weighted:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    ij2wij = None

    if weighted:
        adj, edges, ij2wij = read_edge_list_weighted(g)
    else:
        adj, edges = read_edge_list_unweighted(g)

    if threshold is not None:
        if weighted:
            edge2cid, _ = HLC(adj, edges).single_linkage(threshold, w=ij2wij)
        else:
            edge2cid, _ = HLC(adj, edges).single_linkage(threshold)
    else:
        if weighted:
            edge2cid, _, _, _ = HLC(adj, edges).single_linkage(w=ij2wij)
        else:
            edge2cid, _, _, _ = HLC(adj, edges).single_linkage()

    coms = defaultdict(list)
    for e, com in edge2cid.items():
        coms[com].append(e)

    coms = [c for c in coms.values()]
    return coms

