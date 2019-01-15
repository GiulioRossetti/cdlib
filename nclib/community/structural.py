from nclib.community.algorithms.em import EM_nx
from nclib.community.algorithms.lfm import LFM_nx
from nclib.community.algorithms.scan import SCAN_nx
from nclib.community.algorithms.LAIS2_nx import LAIS2
from nclib.community.algorithms.GDMP2_nx import GDMP2
from nclib.community.algorithms.HLC import HLC, HLC_read_edge_list_unweighted, HLC_read_edge_list_weighted
from nclib.community.algorithms.CONGO import Congo_
from nclib.community.algorithms.CONGA import Conga_
import networkx as nx
import igraph as ig
from nclib.utils import convert_graph_formats
from collections import defaultdict


def kclique(g, k):
    """

    :param g:
    :param k:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    kc = list(nx.algorithms.community.k_clique_communities(g, k))
    coms = [tuple(x) for x in kc]
    return coms


def girvan_newman(g, level):
    """

    :param g:
    :param level:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    gn_hierarchy = nx.algorithms.community.girvan_newman(g)
    coms = []
    for _ in range(level):
        coms = next(gn_hierarchy)

    return list(coms)


def em(g, k):
    """

    :param g:
    :param k:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    algorithm = EM_nx(g, k)
    coms = algorithm.execute()
    return coms


def lfm(g, alpha):
    """

    :param g:
    :param alpha:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    algorithm = LFM_nx(g, alpha)
    coms = algorithm.execute()
    return coms


def scan(g, epsilon, mu):
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


def hierarchical_link_community(g, threshold=None, weighted=False):
    """

    :param g:
    :param threshold:
    :param weighted:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    ij2wij = None

    if weighted:
        adj, edges, ij2wij = HLC_read_edge_list_weighted(g)
    else:
        adj, edges = HLC_read_edge_list_unweighted(g)

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


def lais2(g):
    """

    :param g:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = LAIS2(g)
    return coms


def gdmp2(g, min_threshold=0.75):
    """

    :param g:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = GDMP2(g, min_threshold)
    return coms


def spinglass(g):
    """

    :param g:
    :return:
    """
    g = convert_graph_formats(g, ig.Graph)
    coms = g.community_spinglass()
    communities = []

    for c in coms:
        communities.append(c)

    return communities


def eigenvector(g):
    """

    :param g:
    :return:
    """

    g = convert_graph_formats(g, ig.Graph)
    coms = g.community_leading_eigenvector()
    communities = []

    for c in coms:
        communities.append(c)

    return communities


def congo(g, number_communities=0, height=2):
    """

    :param graph:
    :param number_communities:
    :param height: The lengh of the longest shortest paths that CONGO considers
    :return:

    """

    g = convert_graph_formats(g, ig.Graph)

    communities = Congo_(g, number_communities,height)

    return communities


def conga(g, number_communities=0):
    """

    :param graph:
    :param number_communities:
    :return:
    """

    g = convert_graph_formats(g, ig.Graph)

    communities = Conga_(g, number_communities)

    return communities
