import community as louvain_modularity
import leidenalg
from collections import defaultdict
from networkx.algorithms import community
from nclib.utils import from_nx_to_igraph


def louvain(g, weight='weight', resolution=1., randomize=False):
    """

    :param g:
    :param weight:
    :param resolution:
    :param randomize:
    :return:
    """

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
    gi = from_nx_to_igraph(g)

    part = leidenalg.find_partition(gi, leidenalg.ModularityVertexPartition)
    coms = [gi.vs[x]['name'] for x in part]
    return coms


def greedy_modularity(g, weight=None):
    """

    :param g:
    :param weight:
    :return:
    """
    gc = community.greedy_modularity_communities(g, weight)
    coms = [tuple(x) for x in gc]
    return coms
