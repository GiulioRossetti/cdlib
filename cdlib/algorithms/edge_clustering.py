from cdlib import EdgeClustering
from collections import defaultdict
import networkx as nx
from cdlib.utils import (
    convert_graph_formats,
    nx_node_integer_mapping,
    remap_edge_communities,
)
from cdlib.algorithms.internal.HLC import (
    HLC,
    HLC_read_edge_list_unweighted,
    HLC_read_edge_list_weighted,
    HLC_full,
)

__all__ = [
    "hierarchical_link_community",
    "hierarchical_link_community_w",
    "hierarchical_link_community_full",
]


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


def hierarchical_link_community_w(g_original: object) -> EdgeClustering:
    """
    HLC (hierarchical link clustering) is a method to classify links into topologically related groups.
    The algorithm uses a similarity between links to build a dendrogram where each leaf is a link from the original network and branches represent link communities.
    At each level of the link dendrogram is calculated the partition density function, based on link density inside communities, to pick the best level to cut.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: EdgeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.hierarchical_link_community_w(G)

    :References:

    Ahn, Yong-Yeol, James P. Bagrow, and Sune Lehmann. `Link communities reveal multiscale complexity in networks. <https://www.nature.com/articles/nature09182/>`_ nature 466.7307 (2010): 761.
    """

    g = convert_graph_formats(g_original, nx.Graph)

    adj, edges, ij2wij = HLC_read_edge_list_weighted(g)
    edge2cid, _, _, _ = HLC(adj, edges).single_linkage(w=ij2wij)

    coms = defaultdict(list)
    for e, com in edge2cid.items():
        coms[com].append(e)

    coms = [list(c) for c in coms.values()]
    return EdgeClustering(coms, g_original, "HLC_w", method_parameters={})


def hierarchical_link_community_full(
    g_original: object,
    weight="weight",
    simthr=None,
    hcmethod="single",
    min_edges=None,
    verbose=False,
) -> EdgeClustering:
    """
    HLC (hierarchical link clustering) is a method to classify links into topologically related groups.
    The algorithm uses a similarity between links to build a dendrogram where each leaf is a link from the original network and branches represent link communities.
    At each level of the link dendrogram is calculated the partition density function, based on link density inside communities, to pick the best level to cut.
    This implementation follows exactly the algorithm described in Ahn et al and uses numpy/scipy to improve the clustering computation (It is faster and consumes less memory.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param weight: None for unweighted networks, jaccard approximation is used. When defined with a string, edge attribute name (usually 'weight') to be used as weight and Tanimoto approximation is used.
    :param simthr: None by default. If set to float, all values less than threshold are set to 0 in similarity matrix (it could reduce memory usage).
    :param hcmethod: Linkage method used in hierarchical clustering, 'single' by default. See scipy.cluster.hierarchy.linkage to get full method list.
    :param min_edges: None by default. If set to float, minimum number of edges that a community must contain to be kept in the clustering
    :param verbose: If True, write intermediary steps to disk.
    :return: EdgeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.hierarchical_link_community_full(G)

    :References:

    Ahn, Yong-Yeol, James P. Bagrow, and Sune Lehmann. `Link communities reveal multiscale complexity in networks. <https://www.nature.com/articles/nature09182/>`_ nature 466.7307 (2010): 761.
    """

    g = convert_graph_formats(g_original, nx.Graph)
    g_number, dictio = nx_node_integer_mapping(g)

    coms = HLC_full(
        g_number,
        weight=weight,
        simthr=simthr,
        hcmethod=hcmethod,
        min_edges=min_edges,
        verbose=verbose,
        dictio=dictio,
    ).clusters
    clustering = EdgeClustering(coms, g_number, "HLC_f", method_parameters={})
    if dictio != None:
        clustering.communities = remap_edge_communities(clustering.communities, dictio)
    return clustering
