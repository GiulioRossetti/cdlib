from nclib.community.algorithms import DER
from nclib.community.algorithms import BIGCLAM
from nclib.utils import convert_graph_formats
from nclib import NodeClustering, EdgeClustering
import networkx as nx

__all__ = ["der", "big_clam"]


def der(graph, walk_len=3, threshold=.00001, iter_bound=50):
    """
    DER is a Diffusion Entropy Reducer graph clustering algorithm.
    The algorithm uses random walks to embed the graph in a space of measures, after which a modification of k-means in that space is applied. It creates the walks, creates an initialization, runs the algorithm,
    and finally extracts the communities.

    :param graph: an undirected networkx graph object
    :param walk_len: length of the random walk, default 3
    :param threshold: threshold for stop criteria; if the likelihood_diff is less than threshold tha algorithm stops, default 0.00001
    :param iter_bound: maximum number of iteration, default 50
    :return: list of communities


    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.der(G, 3, .00001, 50)


    :References:

    M. Kozdoba and S. Mannor, **Community Detection via Measure Space Embedding**, NIPS 2015

    """

    graph = convert_graph_formats(graph, nx.Graph)

    communities, _ = DER.der_graph_clustering(graph, walk_len=walk_len,
                                              alg_threshold=threshold, alg_iterbound=iter_bound)

    maps = {k: v for k, v in enumerate(graph.nodes())}
    coms = []
    for c in communities:
        coms.append([maps[n] for n in c])

    return EdgeClustering(coms, graph, "DER", method_parameters={"walk_len": walk_len, "threshold": threshold,
                                                                        "iter_bound": iter_bound})


def big_clam(g, number_communities=5):
    """
    BigClam is an overlapping community detection method that scales to large networks.
    The model has three main ingredients:
    1)The node community memberships are represented with a bipartite affiliation network that links nodes of the social network to communities that they belong to.
    2)People tend to be involved in communities to various degrees. Therefore,  each affiliation edge in the bipartite affiliation network has a nonnegative weight. The higher the node’s weight of the affiliation to the community the more likely is the node to be connected to other members in the community.
    3)When people share multiple community affiliations, the links between them stem for one dominant reason. This means that for each community a pair of nodes shares we get an independent chance of connecting the nodes. Thus, naturally, the more communities a pair of nodes shares, the higher the probability of being connected.

    :param g: a networkx/igraph object
    :param number_communities: number communities desired, default 5
    :return: list of communities


    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.big_clam(G, 2)

    :References:

    Yang, J., & Leskovec, J. (2013, February). **Overlapping community detection at scale: a nonnegative matrix factorization approach.** In Proceedings of the sixth ACM international conference on Web search and data mining (pp. 587-596). ACM.
    """

    g = convert_graph_formats(g, nx.Graph)

    communities = BIGCLAM.big_Clam(g, number_communities)

    return NodeClustering(communities, g, "BigClam", method_parameters={"number_communities": number_communities})
