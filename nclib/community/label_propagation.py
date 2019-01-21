from nclib.community.internal.SLPA_nx import slpa_nx
from nclib.community.internal.multicom import MultiCom
from nclib.community.internal.Markov import markov
import networkx as nx
from nclib.utils import convert_graph_formats, nx_node_integer_mapping
from nclib import NodeClustering, EdgeClustering

__all__ = ["label_propagation", "async_fluid", "slpa", "multicom", "markov_clustering"]


def label_propagation(g):
    """
    The Label Propagation algorithm (LPA) detects communities using network structure alone.
    The algorithm doesn’t require a pre-defined objective function or prior information about the communities.
    It works as follows:
    -Every node is initialized with a unique label (an identifier)
    -These labels propagate through the network
    -At every iteration of propagation, each node updates its label to the one that the maximum numbers of its neighbours belongs to. Ties are broken uniformly and randomly.
    -LPA reaches convergence when each node has the majority label of its neighbours.

    :param g: a networkx/igraph object
    :return: list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.label_propagation(G)

    :References:

    Raghavan, U. N., Albert, R., & Kumara, S. (2007). **Near linear time algorithm to detect community structures in large-scale networks.** Physical review E, 76(3), 036106.
    """

    g = convert_graph_formats(g, nx.Graph)

    lp = list(nx.algorithms.community.label_propagation_communities(g))
    coms = [tuple(x) for x in lp]

    return NodeClustering(coms, g, "Label Propagation")


def async_fluid(g, k):
    """
    Fluid Communities (FluidC) is based on the simple idea of fluids (i.e., communities) interacting in an environment (i.e., a non-complete graph), expanding and contracting.
    It is propagation-based algorithm and it allows to specify the number of desired communities (k) and it is asynchronous, where each vertex update is computed using the latest partial state of the graph.


    :param g: a networkx/igraph object
    :param k: Number of communities to search
    :return: list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.async_fluid(G,k=2)


    :References:

    Ferran Parés, Dario Garcia-Gasulla, Armand Vilalta, Jonatan Moreno, Eduard Ayguadé, Jesús Labarta, Ulises Cortés, Toyotaro Suzumura T. **Fluid Communities: A Competitive and Highly Scalable Community Detection Algorithm.**
    """

    g = convert_graph_formats(g, nx.Graph)

    fluid = nx.algorithms.community.asyn_fluidc(g, k)
    coms = [tuple(x) for x in fluid]
    return NodeClustering(coms, g, "Fluid")


def slpa(g, t=21, r=0.1):
    """
    SLPA is an overlapping community discovery that extends tha LPA.
    SLPA consists of the following three stages:
    1) the initialization
    2) the evolution
    3) the post-processing


    :param g: a networkx/igraph object
    :param t: maximum number of iterations, default 20
    :param r: threshold  ∈ [0, 1]. It is used in the post-processing stage: if the probability of seeing a particular label during the whole process is less than r, this label is deleted from a node’s memory. Default 0.1
    :return: list of communities


    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.slpa(G,  t=21, r=0.1)



    :References:

    Xie Jierui, Boleslaw K. Szymanski, and Xiaoming Liu. **Slpa: Uncovering overlapping communities in social networks via a speaker-listener interaction dynamic process.** Data Mining Workshops (ICDMW), 2011 IEEE 11th International Conference on. IEEE, 2011.

    """

    g = convert_graph_formats(g, nx.Graph)

    coms = slpa_nx(g, T=t, r=r)
    return NodeClustering(coms, g, "SLPA", method_parameters={"T": t, "r": r})


def multicom(g, seed_node):
    """
    MULTICOM is an algorithm for detecting multiple local communities, possibly overlapping, by expanding the initial seed set.
    This algorithm uses local scoring metrics to define an embedding of the graph around the seed set. Based on this embedding, it picks new seeds in the neighborhood of the original seed set, and uses these new seeds to recover multiple communities.

    :param g: a networkx/igraph object
    :param seed_node: Id of the seed node around which we want to detect communities.
    :return: list of communities


    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.multicom(G, seed_node=0)

    :References:

    Hollocou, Alexandre, Thomas Bonald, and Marc Lelarge. **Multiple Local Community Detection.** ACM SIGMETRICS Performance Evaluation Review 45.2 (2018): 76-83.
    """

    g = convert_graph_formats(g, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    mc = MultiCom(g)
    coms = mc.execute(seed_node)

    communities = []
    for c in coms:
        communities.append([maps[n] for n in c])
    nx.relabel_nodes(g, maps, False)

    return NodeClustering(communities, g, "Multicom", method_parameters={"seeds": seed_node})


def markov_clustering(g,  max_loop=1000):
    """
    The Markov clustering algorithm (MCL) is based on simulation of (stochastic) flow in graphs.
    The MCL algorithm finds cluster structure in graphs by a mathematical bootstrapping procedure. The process deterministically computes (the probabilities of) random walks through the graph, and uses two operators transforming one set of probabilities into another. It does so using the language of stochastic matrices (also called Markov matrices) which capture the mathematical concept of random walks on a graph.
    The MCL algorithm simulates random walks within a graph by alternation of two operators called expansion and inflation.

    :param g: a networkx/igraph object
    :param max_loop: maximum number of iterations, default 1000
    :return: list of communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.markov_clustering(G, max_loop=1000)

    :References:

    Enright, Anton J., Stijn Van Dongen, and Christos A. Ouzounis. **An efficient algorithm for large-scale detection of protein families.** Nucleic acids research 30.7 (2002): 1575-1584.
    """

    g = convert_graph_formats(g, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    coms = markov(g, max_loop)

    communities = []
    for c in coms:
        com = []
        for e in c:
            com.append(tuple([maps[n] for n in e]))
        communities.append(com)

    nx.relabel_nodes(g, maps, False)

    return EdgeClustering(communities, g, "Markov Clustering", method_parameters={"max_loop": max_loop})
