from cdlib import EdgeClustering
from collections import defaultdict
import networkx as nx
from cdlib.algorithms.internal.Markov import markov
from cdlib.utils import convert_graph_formats, nx_node_integer_mapping
from cdlib.algorithms.internal.HLC import HLC, HLC_read_edge_list_unweighted, HLC_read_edge_list_weighted

__all__ = ["hierarchical_link_community", "markov_clustering"]


def hierarchical_link_community(g, threshold=None, weighted=False):
    """
    HLC (hierarchical link clustering) is a method to classify links into topologically related groups.
    The algorithm uses a similarity between links to build a dendrogram where each leaf is a link from the original network and branches represent link communities.
    At each level of the link dendrogram is calculated the partition density function, based on link density inside communities, to pick the best level to cut.

    :param g: a networkx/igraph object
    :param threshold: the level where the dendrogram will be cut, default None
    :param weighted: the list of edge weighted, default False
    :return: EdgeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.hierarchical_link_community(G)

    :References:

    Ahn, Yong-Yeol, James P. Bagrow, and Sune Lehmann. **Link communities reveal multiscale complexity in networks.** nature 466.7307 (2010): 761.
    """

    g = convert_graph_formats(g, nx.Graph)

    ij2wij = None

    if weighted:
        adj, edges, ij2wij = HLC_read_edge_list_weighted(g)
    else:
        adj, edges = HLC_read_edge_list_unweighted(g)

    if threshold is not None:
        if weighted:
            edge2cid, _, _, _ = HLC(adj, edges).single_linkage(threshold, w=ij2wij)
        else:
            edge2cid, _, _, _ = HLC(adj, edges).single_linkage(threshold)
    else:
        if weighted:
            edge2cid, _, _, _ = HLC(adj, edges).single_linkage(w=ij2wij)
        else:
            edge2cid, _, _, _ = HLC(adj, edges).single_linkage()

    coms = defaultdict(list)
    for e, com in edge2cid.items():
        coms[com].append(e)

    coms = [tuple(c) for c in coms.values()]
    return EdgeClustering(coms, g, "HLC", method_parameters={"threshold": threshold, "weighted": weighted})


def markov_clustering(g,  max_loop=1000):
    """
    The Markov clustering algorithm (MCL) is based on simulation of (stochastic) flow in graphs.
    The MCL algorithm finds cluster structure in graphs by a mathematical bootstrapping procedure. The process deterministically computes (the probabilities of) random walks through the graph, and uses two operators transforming one set of probabilities into another. It does so using the language of stochastic matrices (also called Markov matrices) which capture the mathematical concept of random walks on a graph.
    The MCL algorithm simulates random walks within a graph by alternation of two operators called expansion and inflation.

    :param g: a networkx/igraph object
    :param max_loop: maximum number of iterations, default 1000
    :return: EdgeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.markov_clustering(G, max_loop=1000)

    :References:

    Enright, Anton J., Stijn Van Dongen, and Christos A. Ouzounis. **An efficient algorithm for large-scale detection of protein families.** Nucleic acids research 30.7 (2002): 1575-1584.
    """

    g = convert_graph_formats(g, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    coms = markov(g, max_loop)

    if maps is not None:
        communities = []
        for c in coms:
            com = []
            for e in c:
                com.append(tuple([maps[n] for n in e]))
            communities.append(com)

        nx.relabel_nodes(g, maps, False)
    else:
        communities = [tuple(c) for c in coms]

    return EdgeClustering(communities, g, "Markov Clustering", method_parameters={"max_loop": max_loop})
