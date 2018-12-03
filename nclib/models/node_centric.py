import demon as d
import angel as an
from nclib.models.algorithms.NodePerception import NodePerception
import networkx as nx
import os
from nclib.utils import suppress_stdout


def ego_networks(g, level=1):
    """

    :param g:
    :param level:
    :return:
    """
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

    with suppress_stdout():
        dm = d.Demon(graph=g, epsilon=epsilon, min_community_size=min_com_size)
        coms = dm.execute()

    return coms


def angel(g, threshold, min_community_size=3):
    """

    :param g:
    :param threshold:
    :param min_community_size:
    :return:
    """

    nx.write_edgelist(g, "temp.ncol")
    with suppress_stdout():
        a = an.Angel("temp.ncol", min_comsize=min_community_size, threshold=threshold, save=False)
        coms = a.execute()
    os.remove("temp.ncol")

    return list(coms.values())


def node_perception(g, threshold, overlap_threshold, min_comm_size=3):
    """

    :param g:
    :param threshold:
    :param overlap_threshold:
    :param min_comm_size:
    :return:
    """
    with suppress_stdout():
        np = NodePerception(g, sim_threshold=threshold, overlap_threshold=overlap_threshold, min_comm_size=min_comm_size)
        coms = np.execute()

    return coms

