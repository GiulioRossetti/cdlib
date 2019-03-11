from cdlib import EdgeClustering
from collections import defaultdict
import networkx as nx
from cdlib.algorithms.internal.Markov import markov
from cdlib.utils import convert_graph_formats, nx_node_integer_mapping
from cdlib.algorithms.internal.HLC import HLC, HLC_read_edge_list_unweighted

__all__ = ["hierarchical_link_community", "markov_clustering"]


def hierarchical_link_community(g):
    """
    HLC (hierarchical link clustering) is a method to classify links into topologically related groups.
    The algorithm uses a similarity between links to build a dendrogram where each leaf is a link from the original network and branches represent link communities.
    At each level of the link dendrogram is calculated the partition density function, based on link density inside communities, to pick the best level to cut.

    :param g: a networkx/igraph object
    :return: EdgeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.hierarchical_link_community(G)

    :References:

    Ahn, Yong-Yeol, James P. Bagrow, and Sune Lehmann. `Link communities reveal multiscale complexity in networks. <https://www.nature.com/articles/nature09182/>`_ nature 466.7307 (2010): 761.
    """

    g = convert_graph_formats(g, nx.Graph)

    adj, edges = HLC_read_edge_list_unweighted(g)

    edge2cid, _, _, _ = HLC(adj, edges).single_linkage()

    coms = defaultdict(list)
    for e, com in edge2cid.items():
        coms[com].append(e)

    coms = [list(c) for c in coms.values()]
    return EdgeClustering(coms, g, "HLC", method_parameters={})


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

    Enright, Anton J., Stijn Van Dongen, and Christos A. Ouzounis. `An efficient algorithm for large-scale detection of protein families. <https://www.ncbi.nlm.nih.gov/pubmed/11917018/>`_ Nucleic acids research 30.7 (2002): 1575-1584.

    .. note:: Reference implementation: https://github.com/HarshHarwani/markov-clustering-for-graphs
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
        communities = [list(c) for c in coms]

    return EdgeClustering(communities, g, "Markov Clustering", method_parameters={"max_loop": max_loop})
