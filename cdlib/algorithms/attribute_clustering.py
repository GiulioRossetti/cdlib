try:
    import igraph as ig
except ModuleNotFoundError:
        ig = None

import Eva

from collections import defaultdict
from cdlib import AttrNodeClustering

import networkx as nx

from cdlib.utils import convert_graph_formats

__all__ = ['eva']

def eva(g, labels, weight='weight', resolution=1., randomize=False, alpha=0.5):

    """
       The Eva algorithm extends the Louvain approach in order to deal with the attributes of the nodes (aka Louvain Extended to Vertex Attributes).
       It optimizes - combining them linearly - two quality functions, a structural and a clustering one, namely the modularity and the purity.
       A parameter alpha tunes the importance of the two functions: an high value of alpha favors the clustering criterion instead of the structural one.

       :param g: a networkx/igraph object
       :param weight: str, optional the key in graph to use as weight. Default to 'weight'
       :param resolution: double, optional  Will change the size of the communities, default to 1.
       :param randomize:  boolean, optional  Will randomize the node evaluation order and the community evaluation  order to get different partitions at each call, default False
       :param alpha: a value assumed in [0,1] tuning the importance of modularity and purity criteria
       :return: AttrNodeClustering object

       :Example:

        >>> from cdlib.algorithms import eva
        >>> import networkx as nx
        >>> import random
        >>> l1 = ['A', 'B', 'C', 'D']
        >>> l2 = ["E", "F", "G"]
        >>> g = nx.barabasi_albert_graph(100, 5)
        >>> labels=dict()
        >>> for node in g.nodes():
        >>>    labels[node]={"l1":random.choice(l1), "l2":random.choice(l2)}
        >>> communities = eva(g_attr, labels, alpha=0.8)

       :References:

      1. #####

       .. note:: Reference implementation: https://github.com/GiulioRossetti/Eva/tree/master/Eva
       """

    g = convert_graph_formats(g, nx.Graph)
    nx.set_node_attributes(g, labels)

    coms, coms_labels = Eva.eva_best_partition(g, weight=weight, resolution=resolution, randomize=randomize, alpha=alpha)

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in coms.items():
        coms_to_node[c].append(n)

    coms_eva = [list(c) for c in coms_to_node.values()]
    return AttrNodeClustering(coms_eva, g, coms_labels, "Eva", method_parameters={"weight": weight, "resolution": resolution,
                                                                         "randomize": randomize, "alpha":alpha})