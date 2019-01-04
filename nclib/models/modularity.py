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


def rb_pots(g):
    """

    Reichardt, J., & Bornholdt, S. (2006).
    Statistical mechanics of community detection.
    Physical Review E, 74(1), 016110. 10.1103/PhysRevE.74.016110

    Leicht, E. A., & Newman, M. E. J. (2008).
    Community Structure in Directed Networks.
    Physical Review Letters, 100(11), 118703. 10.1103/PhysRevLett.100.118703

    :param g:
    :return:
    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition)
    coms = [g.vs[x]['name'] for x in part]
    return coms


def rber_pots(g):
    """

    Reichardt, J., & Bornholdt, S. (2006).
    Statistical mechanics of community detection.
    Physical Review E, 74(1), 016110. 10.1103/PhysRevE.74.016110

    :param g:
    :return:
    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.RBERVertexPartition)
    coms = [g.vs[x]['name'] for x in part]
    return coms


def cpm(g):
    """

    Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011).
    Narrow scope for resolution-limit-free community detection.
    Physical Review E, 84(1), 016114. 10.1103/PhysRevE.84.016114

    :param g:
    :return:
    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.CPMVertexPartition)
    coms = [g.vs[x]['name'] for x in part]
    return coms


def significance_communities(g):
    """

    Traag, V. A., Krings, G., & Van Dooren, P. (2013).
    Significant scales in community structure.
    Scientific Reports, 3, 2930. 10.1038/srep02930

    :param g:
    :return:
    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.SignificanceVertexPartition)
    coms = [g.vs[x]['name'] for x in part]
    return coms


def surprise_communities(g):
    """

    Traag, V. A., Aldecoa, R., & Delvenne, J.-C. (2015).
    Detecting communities using asymptotical surprise.
    Physical Review E, 92(2), 022816. 10.1103/PhysRevE.92.022816

    :param g:
    :return:
    """

    g = convert_graph_formats(g, ig.Graph)

    part = leidenalg.find_partition(g, leidenalg.SurpriseVertexPartition)
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
