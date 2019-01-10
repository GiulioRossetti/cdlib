from demon import Demon
from angel import Angel
from nclib.community.algorithms.NodePerception import NodePerception
from nclib.community.algorithms import OSSE
import networkx as nx
import igraph as ig
from nclib.utils import suppress_stdout, convert_graph_formats


def ego_networks(g, level=1):
    """

    :param g:
    :param level:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = []
    for n in g.nodes():
        coms.append(list(nx.ego_graph(g, n, radius=level).nodes()))
    return coms


def demon(g, epsilon, min_com_size=3):
    """

    :param g:
    :param epsilon:
    :param min_com_size:
    :param filename:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    with suppress_stdout():
        dm = Demon(graph=g, epsilon=epsilon, min_community_size=min_com_size)
        coms = dm.execute()

    return coms


def angel(g, threshold, min_community_size=3):
    """

    :param g:
    :param threshold:
    :param min_community_size:
    :return:
    """

    g = convert_graph_formats(g, ig.Graph)

    a = Angel(graph=g, min_comsize=min_community_size, threshold=threshold, save=False)
    coms = a.execute()

    return list(coms.values())


def node_perception(g, threshold, overlap_threshold, min_comm_size=3):
    """

    :param g:
    :param threshold:
    :param overlap_threshold:
    :param min_comm_size:
    :return:
    """
    g = convert_graph_formats(g, nx.Graph)

    with suppress_stdout():
        np = NodePerception(g, sim_threshold=threshold, overlap_threshold=overlap_threshold, min_comm_size=min_comm_size)
        coms = np.execute()

    return coms


def overlapping_seed_set_expansion(g, seeds, ninf=False, expansion='ppr', stopping='cond', nworkers=1,
                                   nruns=13, alpha=0.99, maxexpand=float('INF'), delta=0.2):
    """

    Overlapping Community Detection Using Seed Set Expansion (CIKM 2013)
    Joyce Jiyoung Whang, David F. Gleich, and Inderjit S. Dhillon

    :param g:
    :param seeds:
    :param ninf: Neighbourhood Inflation parameter (boolean)
    :param expansion: Seed expansion: ppr or vppr
    :param stopping: Stopping criteria: cond
    :param nworkers: Number of Workers: default 1
    :param nruns: Number of runs: default 13
    :param alpha: alpha value for Personalized PageRank expansion: default 0.99
    :param maxexpand: Maximum expansion allowed for approximate ppr: default INF
    :param delta: Minimum distance parameter for near duplicate communities: default 0.2
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    if ninf:
        seeds = OSSE.neighbor_inflation(g, seeds)

    communities = OSSE.growclusters(g, seeds, expansion, stopping, nworkers, nruns, alpha, maxexpand, False)
    communities = OSSE.remove_duplicates(g, communities, delta)
    communities = list(communities)
    return communities
