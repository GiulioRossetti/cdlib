import community as louvain_modularity
import leidenalg
from collections import defaultdict
import networkx as nx
import igraph as ig
from networkx.algorithms import community
from nclib.utils import convert_graph_formats


def louvain(g, weight='weight', resolution=1., randomize=False):
    """

    :param g:
    :param weight:
    :param resolution:
    :param randomize:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = louvain_modularity.best_partition(g, weight=weight, resolution=resolution, randomize=randomize)

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in coms.items():
        coms_to_node[c].append(n)

    coms_louvain = [tuple(c) for c in coms_to_node.values()]
    return coms_louvain


def leiden(g):
    """

    :param g:
    :return:
    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    coms = [g.vs[x]['name'] for x in part]
    return coms


def greedy_modularity(g, weight=None):
    """

    :param g:
    :param weight:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    gc = community.greedy_modularity_communities(g, weight)
    coms = [tuple(x) for x in gc]
    return coms
