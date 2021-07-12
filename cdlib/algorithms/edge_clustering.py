from cdlib import EdgeClustering
from collections import defaultdict
import networkx as nx
from cdlib.utils import convert_graph_formats
from cdlib.algorithms.internal.HLC import HLC, HLC_read_edge_list_unweighted

__all__ = ["hierarchical_link_community"]


def hierarchical_link_community(g_original: object) -> EdgeClustering:
    """
    HLC (hierarchical link clustering) is a method to classify links into topologically related groups.
    The algorithm uses a similarity between links to build a dendrogram where each leaf is a link from the original network and branches represent link communities.
    At each level of the link dendrogram is calculated the partition density function, based on link density inside communities, to pick the best level to cut.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: EdgeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.hierarchical_link_community(G)

    :References:

    Ahn, Yong-Yeol, James P. Bagrow, and Sune Lehmann. `Link communities reveal multiscale complexity in networks. <https://www.nature.com/articles/nature09182/>`_ nature 466.7307 (2010): 761.
    """

    g = convert_graph_formats(g_original, nx.Graph)

    adj, edges = HLC_read_edge_list_unweighted(g)

    edge2cid, _, _, _ = HLC(adj, edges).single_linkage()

    coms = defaultdict(list)
    for e, com in edge2cid.items():
        coms[com].append(e)

    coms = [list(c) for c in coms.values()]
    return EdgeClustering(coms, g_original, "HLC", method_parameters={})
