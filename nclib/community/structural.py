from nclib import NodeClustering, EdgeClustering
from nclib.community.algorithms.em import EM_nx
from nclib.community.algorithms.lfm import LFM_nx
from nclib.community.algorithms.scan import SCAN_nx
from nclib.community.algorithms.LAIS2_nx import LAIS2
from nclib.community.algorithms.GDMP2_nx import GDMP2
from nclib.community.algorithms.HLC import HLC, HLC_read_edge_list_unweighted, HLC_read_edge_list_weighted
from nclib.community.algorithms.CONGO import Congo_
from nclib.community.algorithms.CONGA import Conga_
from nclib.community.algorithms.AGDL import Agdl
import networkx as nx
import igraph as ig
from nclib.utils import convert_graph_formats, nx_node_integer_mapping
from collections import defaultdict

__all__ = ["kclique", "girvan_newman", "em", "lfm", "scan", "hierarchical_link_community", "lais2", "gdmp2",
           "spinglass", "eigenvector", "conga", "congo", "agdl"]


def kclique(g, k):
    """

    :param g:
    :param k:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    kc = list(nx.algorithms.community.k_clique_communities(g, k))
    coms = [tuple(x) for x in kc]
    return NodeClustering(coms, g, "Klique", method_parameters={"k": k}, overlap=True)


def girvan_newman(g, level):
    """

    :param g:
    :param level:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    gn_hierarchy = nx.algorithms.community.girvan_newman(g)
    coms = []
    for _ in range(level):
        coms = next(gn_hierarchy)

    communities = []

    for c in coms:
        communities.append(list(c))

    return NodeClustering(communities, g, "Girvan Newman", method_parameters={"level": level})


def em(g, k):
    """

    :param g:
    :param k:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    algorithm = EM_nx(g, k)
    coms = algorithm.execute()

    communities = []
    for c in coms:
        communities.append([maps[n] for n in c])

    nx.relabel_nodes(g, maps, False)

    return NodeClustering(communities, g, "EM", method_parameters={"k": k})


def lfm(g, alpha):
    """

    :param g:
    :param alpha:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    algorithm = LFM_nx(g, alpha)
    coms = algorithm.execute()

    return NodeClustering(coms, g, "LFM", method_parameters={"alpha": alpha})


def scan(g, epsilon, mu):
    """

    :param g:
    :param epsilon:
    :param mu:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    algorithm = SCAN_nx(g, epsilon, mu)
    coms = algorithm.execute()
    return NodeClustering(coms, g, "SCAN", method_parameters={"epsilon": epsilon,
                                                              "mu": mu})


def hierarchical_link_community(g, threshold=None, weighted=False):
    """

    :param g:
    :param threshold:
    :param weighted:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    ij2wij = None

    if weighted:
        adj, edges, ij2wij = HLC_read_edge_list_weighted(g)
    else:
        adj, edges = HLC_read_edge_list_unweighted(g)

    if threshold is not None:
        if weighted:
            edge2cid, _ = HLC(adj, edges).single_linkage(threshold, w=ij2wij)
        else:
            edge2cid, _ = HLC(adj, edges).single_linkage(threshold)
    else:
        if weighted:
            edge2cid, _, _, _ = HLC(adj, edges).single_linkage(w=ij2wij)
        else:
            edge2cid, _, _, _ = HLC(adj, edges).single_linkage()

    coms = defaultdict(list)
    for e, com in edge2cid.items():
        coms[com].append(e)

    coms = [c for c in coms.values()]
    return EdgeClustering(coms, g, "HLC", method_parameters={"threshold": threshold, "weighted": weighted})


def lais2(g):
    """

    :param g:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = LAIS2(g)
    return NodeClustering(coms, g, "LAIS2")


def gdmp2(g, min_threshold=0.75):
    """
    Gdmp2 is a method for identifying a set of dense subgraphs of a given sparse graph.
    It is inspired by an effective technique designed for a similar problem—matrix blocking, from a different discipline (solving linear systems).

    :param g: a networkx/igraph object
    :param min_threshold:  the minimum density threshold parameter to control the density of the output subgraphs, default 0.75
    :return: list of communities


    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = community.gdmp2(G)

    :References:

    Chen, Jie, and Yousef Saad. **Dense subgraph extraction with application to community detection.** IEEE Transactions on Knowledge and Data Engineering 24.7 (2012): 1216-1230.
    """

    g = convert_graph_formats(g, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    coms = GDMP2(g, min_threshold)

    communities = []
    for c in coms:
        communities.append([maps[n] for n in c])

    nx.relabel_nodes(g, maps, False)

    return NodeClustering(communities, g, "GDMP2", method_parameters={"min_threshold": min_threshold})


def spinglass(g):
    """
    Spinglass relies on an analogy between a very popular statistical mechanic model called Potts spin glass, and the community structure.
    It applies the simulated annealing optimization technique on this model to optimize the modularity.

    :param g: a networkx/igraph object
    :return: list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = community.spinglass(G)

    :References:

    Reichardt, Jörg, and Stefan Bornholdt. **Statistical mechanics of community detection.** Physical Review E 74.1 (2006): 016110.
    """
    g = convert_graph_formats(g, ig.Graph)
    coms = g.community_spinglass()
    communities = []

    for c in coms:
        communities.append([g.vs[x]['name'] for x in c])

    return NodeClustering(communities, g, "Spinglass")


def eigenvector(g):
    """
    Newman's leading eigenvector method for detecting community structure based on modularity.
    This is the proper implementation of the recursive, divisive algorithm: each split is done by maximizing the modularity regarding the original network.

    :param g: a networkx/igraph object
    :return: list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = community.eigenvector(G)

    :References:

    Newman, Mark EJ. **Finding community structure in networks using the eigenvectors of matrices.** Physical review E 74.3 (2006): 036104.
    """

    g = convert_graph_formats(g, ig.Graph)
    coms = g.community_leading_eigenvector()

    communities = [g.vs[x]['name'] for x in coms]

    return NodeClustering(communities, g, "Eigenvector")


def congo(g, number_communities, height=2):
    """
    CONGO (CONGA Optimized) is an optimization of the CONGA algortithm.
    The CONGO algorithm is the same as CONGA but using local betweenness. The complete CONGO algorithm is as follows:

    1. Calculate edge betweenness of edges and split betweenness of vertices.
    2. Find edge with maximum edge betweenness or vertex with maximum split betweenness, if greater.
    3. Recalculate edge betweenness and split betweenness:
        a) Subtract betweenness of h-region centred on the removed edge or split vertex.
        b) Remove the edge or split the vertex.
        c) Add betweenness for the same region.
    4. Repeat from step 2 until no edges remain.

    :param g: a networkx/igraph object
    :param number_communities: the number of communities desired
    :param height: The lengh of the longest shortest paths that CONGO considers, default 2
    :return: list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = community.congo(G, number_communities=3, height=2)

    :References:

    Gregory, Steve. **A fast algorithm to find overlapping communities in networks.** Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2008.
    """

    g = convert_graph_formats(g, ig.Graph)

    communities = Congo_(g, number_communities, height)

    coms = []
    for c in communities:
        coms.append([g.vs[x]['name'] for x in c])

    return NodeClustering(coms, g, "Congo", method_parameters={"number_communities": number_communities,
                                                               "height": height})


def conga(g, number_communities):
    """
    CONGA (Cluster-Overlap Newman Girvan Algorithm) is an algorithm for discovering overlapping communities.
    It extends the  Girvan and Newman’s algorithm with a specific method of deciding when and how to split vertices. The algorithm is as follows:

    1. Calculate edge betweenness of all edges in network.
    2. Calculate vertex betweenness of vertices, from edge betweennesses.
    3. Find candidate set of vertices: those whose vertex betweenness is greater than the maximum edge betweenness.
    4. If candidate set is non-empty, calculate pair betweennesses of candidate vertices, and then calculate split betweenness of candidate vertices.
    5. Remove edge with maximum edge betweenness or split vertex with maximum split betweenness (if greater).
    6. Recalculate edge betweenness for all remaining edges in same component(s) as removed edge or split vertex.
    7. Repeat from step 2 until no edges remain.

    :param g: a networkx/igraph object
    :param number_communities: the number of communities desired
    :return: list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = community.conga(G, number_communities=3)

    :References:

    Gregory, Steve. **An algorithm to find overlapping community structure in networks.** European Conference on Principles of Data Mining and Knowledge Discovery. Springer, Berlin, Heidelberg, 2007.
    """

    g = convert_graph_formats(g, ig.Graph)

    communities = Conga_(g, number_communities=3)
    coms = []
    for c in communities:
        coms.append([g.vs[x]['name'] for x in c])

    return NodeClustering(coms, g, "Conga", method_parameters={"number_communities": number_communities})


def agdl(g, number_communities, number_neighbors, kc, a):
    """

    :param g: a networkx/igraph object
    :param number_communities: number of communities
    :param number_neighbors: Number of neighbors to use for KNN
    :param kc: size of the neighbor set for each cluster
    :param a: range(-infinity;+infinty). From the authors: a=np.arange(-2,2.1,0.5)
    :return: list of communities

     :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = community.Agdl(g, number_communities=3, number_neighbors=3, kc=4, a=1)

    :References:

    Zhang, W., Wang, X., Zhao, D., & Tang, X. (2012, October). **Graph degree linkage: Agglomerative clustering on a directed graph.** In European Conference on Computer Vision (pp. 428-441). Springer, Berlin, Heidelberg.

    """

    g = convert_graph_formats(g, nx.Graph)

    communities = Agdl(g, number_communities, number_neighbors, kc, a)
    nodes = {k: v for k, v in enumerate(g.nodes())}
    coms = []
    for com in communities:
        coms.append([nodes[n] for n in com])

    return NodeClustering(coms, g, "AGDL", method_parameters={"number_communities": number_communities,
                                                                     "number_neighbors": number_neighbors,
                                                                     "kc": kc, "a": a})
