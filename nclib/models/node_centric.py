import demon as d
import angel as an
import networkx as nx
import os
from nclib.utils import suppress_stdout


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

    nx.write_edgelist(g, "temp.ncol")
    with suppress_stdout():
        a = an.Angel("temp.ncol", min_comsize=min_community_size, threshold=threshold, save=False)
        coms = a.execute()
    os.remove("temp.ncol")

    return coms