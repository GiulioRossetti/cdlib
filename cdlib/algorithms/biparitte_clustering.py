from BiMLPA import BiMLPA_SqrtDeg, relabeling, output_community
from cdlib import BiNodeClustering

import networkx as nx
from cdlib.utils import convert_graph_formats


__all__ = ['bimlpa']


def bimlpa(g, theta=0.3, lambd=7):
    """

    :param g:
    :param theta:
    :param lambd:
    :return:
    """
    g = convert_graph_formats(g, nx.Graph)

    bimlpa = BiMLPA_SqrtDeg(g, theta, lambd)
    bimlpa.start()
    relabeling(g)
    top_coms, bottom_coms = output_community(g)

    return BiNodeClustering(top_coms, bottom_coms, g, "BiMLPA", method_parameters={"theta": theta, "lambd": lambd})
