from cdlib.benchmark.internal import xmark
import cdlib
from cdlib import NodeClustering
from collections import defaultdict

__all__ = ["LFR", "XMark", "GRP", "PP", "RPG", "SBM"]


def LFR(
    n: int,
    tau1: float,
    tau2: float,
    mu: float,
    average_degree: float = None,
    min_degree: int = None,
    max_degree: int = None,
    min_community: int = None,
    max_community: int = None,
    tol: float = 1e-07,
    max_iters: int = 500,
    seed: int = 42,
) -> [object, cdlib.NodeClustering]:
    """
    Returns the LFR benchmark graph and planted communities.

    :param n: Number of nodes in the created graph.
    :param tau1: Power law exponent for the degree distribution of the created graph. This value must be strictly greater than one.
    :param tau2: Power law exponent for the community size distribution in the created graph. This value must be strictly greater than one.
    :param mu: Fraction of intra-community edges incident to each node. This value must be in the interval [0, 1].
    :param average_degree: Desired average degree of nodes in the created graph. This value must be in the interval [0, n]. Exactly one of this and min_degree must be specified, otherwise a NetworkXError is raised.
    :param min_degree: Minimum degree of nodes in the created graph. This value must be in the interval [0, n]. Exactly one of this and average_degree must be specified, otherwise a NetworkXError is raised.
    :param max_degree: Maximum degree of nodes in the created graph. If not specified, this is set to n, the total number of nodes in the graph.
    :param min_community: Minimum size of communities in the graph. If not specified, this is set to min_degree.
    :param max_community:  Maximum size of communities in the graph. If not specified, this is set to n, the total number of nodes in the graph.
    :param tol: Tolerance when comparing floats, specifically when comparing average degree values.
    :param max_iters: Maximum number of iterations to try to create the community sizes, degree distribution, and community affiliations.
    :param seed: (integer, random_state, or None (default)) – Indicator of random number generation state. See Randomness.

    :return: A networkx synthetic graph, the set of communities  (NodeClustering object)

    :Example:

    >>> from cdlib.benchmark import LFR
    >>> n = 250
    >>> tau1 = 3
    >>> tau2 = 1.5
    >>> mu = 0.1
    >>> G, coms = LFR(n, tau1, tau2, mu, average_degree=5, min_community=20)

    :References:

    Andrea Lancichinetti, Santo Fortunato, and Filippo Radicchi. “Benchmark graphs for testing community detection algorithms”, Phys. Rev. E 78, 046110 2008

    .. note:: Reference implementation: https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.LFR_benchmark_graph.html#networkx.generators.community.LFR_benchmark_graph
    """

    from networkx.generators.community import LFR_benchmark_graph

    G = LFR_benchmark_graph(
        n=n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        average_degree=average_degree,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        tol=tol,
        max_iters=max_iters,
        seed=seed,
    )

    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    communities = [list(c) for c in communities]

    coms = NodeClustering(
        communities,
        G,
        "LFR",
        method_parameters={
            "n": n,
            "tau1": tau1,
            "tau2": tau2,
            "mu": mu,
            "average_degree": average_degree,
            "min_degree": min_degree,
            "max_degree": max_degree,
            "min_community": min_community,
            "max_community": max_community,
            "tol": tol,
            "max_iters": max_iters,
            "seed": seed,
        },
    )

    return G, coms


def XMark(
    n: int = 2000,
    gamma: float = 3,
    beta: float = 2,
    m_cat: tuple = ("auto", "auto"),
    theta: float = 0.3,
    mu: float = 0.5,
    avg_k: int = 10,
    min_com: int = 20,
    type_attr: str = "categorical",
) -> [object, cdlib.NodeClustering]:
    """
    Returns the XMark benchmark annotated graph and planted communities.

    :param n: Number of nodes in the created graph.
    :param gamma: Power law exponent for the degree distribution of the created graph. This value must be strictly greater than one.
    :param beta: Power law exponent for the community size distribution in the created graph. This value must be strictly greater than one.
    :param m_cat:
    :param theta:
    :param mu: Fraction of intra-community edges incident to each node. This value must be in the interval [0, 1].
    :param avg_k: esired average degree of nodes in the created graph. This value must be in the interval [0, n]. Exactly one of this and min_degree must be specified, otherwise a NetworkXError is raised.
    :param min_com: Minimum size of communities in the graph. If not specified, this is set to min_degree.
    :param type_attr:

    :return: A networkx synthetic graph, the set of communities  (NodeClustering object)

    :Example:

    >>> from cdlib.benchmark import XMark
    >>> N = 2000
    >>> gamma = 3
    >>> beta = 2
    >>> m_cat = ["auto", "auto"]
    >>> theta = 0.3
    >>> mu = 0.5
    >>> avg_k = 10
    >>> min_com = 20
    >>> g, coms = XMark(n=N, gamma=gamma, beta=beta, mu=mu,
    >>>                           m_cat=m_cat,
    >>>                           theta=theta,
    >>>                           avg_k=avg_k, min_com=min_com,
    >>>                           type_attr="categorical")

    :References:

    Salvatore Citraro, and Giulio Rossetiìti. “XMark: A Benchmark For Node-Attributed Community Discovery Algorithms”, 2021 (to appear)

    .. note:: Reference implementation: https://github.com/dsalvaz/XMark
    """

    G = xmark.XMark_benchmark(
        n=n,
        tau1=gamma,
        tau2=beta,
        mu=mu,
        labels=m_cat,
        noise=theta,
        average_degree=avg_k,
        min_community=min_com,
        type_attr=type_attr,
    )

    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    communities = [list(c) for c in communities]

    coms = NodeClustering(
        communities,
        G,
        "RDyn",
        method_parameters={
            "n": n,
            "gamma": gamma,
            "beta": beta,
            "mu": mu,
            "m_cat": m_cat,
            "theta": theta,
            "avg_k": avg_k,
            "min_com": min_com,
        },
    )

    return G, coms


def GRP(
    n: int,
    s: float,
    v: float,
    p_in: float,
    p_out: float,
    directed: bool = False,
    seed: object = 42,
) -> [object, cdlib.NodeClustering]:
    """
    Generate a Gaussian random partition graph.

    A Gaussian random partition graph is created by creating k partitions
    each with a size drawn from a normal distribution with mean s and variance
    s/v. Nodes are connected within clusters with probability p_in and
    between clusters with probability p_out.

    :param n: Number of nodes in the graph
    :param s: Mean cluster size
    :param v: Shape parameter. The variance of cluster size distribution is s/v.
    :param p_in: Probabilty of intra cluster connection.
    :param p_out: Probability of inter cluster connection.
    :param directed: hether to create a directed graph or not. Boolean, default False
    :param seed: Indicator of random number generation state.

    :return: A networkx synthetic graph, the set of communities  (NodeClustering object)

    :Example:

    >>> from cdlib.benchmark import GRP
    >>> G, coms = GRP(100, 10, 10, 0.25, 0.1)

    :References:

    Ulrik Brandes, Marco Gaertler, Dorothea Wagner, Experiments on Graph Clustering Algorithms,  In the proceedings of the 11th Europ. Symp. Algorithms, 2003.

    .. note:: Reference implementation: https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.gaussian_random_partition_graph.html#networkx.generators.community.gaussian_random_partition_graph
    """

    from networkx.generators.community import gaussian_random_partition_graph

    G = gaussian_random_partition_graph(
        n=n, s=s, v=v, p_in=p_in, p_out=p_out, directed=directed, seed=seed
    )
    communities = defaultdict(list)
    for n, data in G.nodes(data=True):
        communities[data["block"]].append(n)

    coms = NodeClustering(
        list(communities.values()),
        G,
        "GRP",
        method_parameters={
            "n": n,
            "s": s,
            "v": v,
            "p_in": p_in,
            "p_out": p_out,
            "directed": directed,
            "seed": seed,
        },
    )

    return G, coms


def PP(
    l: int, k: int, p_in: float, p_out: float, seed: object = 42, directed: bool = False
) -> [object, cdlib.NodeClustering]:
    """
    Returns the planted l-partition graph.

    This model partitions a graph with n=l*k vertices in l groups with k vertices each. Vertices of the same group are linked with a probability p_in, and vertices of different groups are linked with probability p_out.

    :param l: Number of groups
    :param k: Number of vertices in each group
    :param p_in: probability of connecting vertices within a group
    :param p_out:  probability of connected vertices between groups
    :param seed: Indicator of random number generation state.
    :param directed: hether to create a directed graph or not. Boolean, default False

    :return: A networkx synthetic graph, the set of communities  (NodeClustering object)

    :Example:

    >>> from cdlib.benchmark import planted_partitions
    >>> G, coms = planted_partitions(4, 3, 0.5, 0.1, seed=42)

    :References:

    A. Condon, R.M. Karp, Algorithms for graph partitioning on the planted partition model, Random Struct. Algor. 18 (2001) 116-140.
    Santo Fortunato ‘Community Detection in Graphs’ Physical Reports Volume 486, Issue 3-5 p. 75-174. https://arxiv.org/abs/0906.0612

    .. note:: Reference implementation: https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.planted_partition_graph.html#networkx.generators.community.planted_partition_graph
    """
    from networkx.generators.community import planted_partition_graph

    G = planted_partition_graph(
        l=l, k=k, p_in=p_in, p_out=p_out, seed=seed, directed=directed
    )
    communities = defaultdict(list)
    for n, data in G.nodes(data=True):
        communities[data["block"]].append(n)

    coms = NodeClustering(
        list(communities.values()),
        G,
        "planted_partitions",
        method_parameters={
            "l": l,
            "k": k,
            "p_in": p_in,
            "p_out": p_out,
            "seed": seed,
            "directed": directed,
        },
    )

    return G, coms


def RPG(
    sizes: list, p_in: float, p_out: float, seed: object = 42, directed: bool = False
) -> [object, cdlib.NodeClustering]:
    """
    Returns the random partition graph with a partition of sizes.

    A partition graph is a graph of communities with sizes defined by s in sizes. Nodes in the same group are connected with probability p_in and nodes of different groups are connected with probability p_out.

    :param sizes: Sizes of groups (list of ints)
    :param p_in: probability of connecting vertices within a group
    :param p_out:  probability of connected vertices between groups
    :param seed: Indicator of random number generation state.
    :param directed: hether to create a directed graph or not. Boolean, default False

    :return: A networkx synthetic graph, the set of communities  (NodeClustering object)

    :Example:

    >>> from cdlib.benchmark import RPG
    >>> G, coms = RPG([10, 10, 10], 0.25, 0.01)

    :References:

    Santo Fortunato ‘Community Detection in Graphs’ Physical Reports Volume 486, Issue 3-5 p. 75-174. https://arxiv.org/abs/0906.0612

    .. note:: Reference implementation: https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.random_partition_graph.html#networkx.generators.community.random_partition_graph
    """

    from networkx.generators.community import random_partition_graph

    G = random_partition_graph(
        sizes=sizes, p_in=p_in, p_out=p_out, seed=seed, directed=directed
    )
    communities = defaultdict(list)
    for n, data in G.nodes(data=True):
        communities[data["block"]].append(n)

    coms = NodeClustering(
        list(communities.values()),
        G,
        "RPG",
        method_parameters={
            "sizes": sizes,
            "p_in": p_in,
            "p_out": p_out,
            "seed": seed,
            "directed": directed,
        },
    )

    return G, coms


def SBM(
    sizes: list,
    p: list,
    nodelist: list = None,
    seed: object = 42,
    directed: bool = False,
    selfloops: bool = False,
    sparse: bool = True,
) -> [object, cdlib.NodeClustering]:
    """
    Returns a stochastic block model graph.

    This model partitions the nodes in blocks of arbitrary sizes, and places edges between pairs of nodes independently, with a probability that depends on the blocks.

    :param sizes: Sizes of blocks (list of ints)
    :param p: Element (r,s) gives the density of edges going from the nodes of group r to nodes of group s. p must match the number of groups (len(sizes) == len(p)), and it must be symmetric if the graph is undirected. (List of floats)
    :param nodelist: The block tags are assigned according to the node identifiers in nodelist. If nodelist is None, then the ordering is the range [0,sum(sizes)-1]. Optional, default None.
    :param seed: Indicator of random number generation state.
    :param directed: hether to create a directed graph or not. Boolean, default False.
    :param selfloops: Whether to include self-loops or not. Optional, default False.
    :param sparse: Use the sparse heuristic to speed up the generator. Optional, default True.

    :return: A networkx synthetic graph, the set of communities  (NodeClustering object)

    :Example:

    >>> from cdlib.benchmark import SBM
    >>> sizes = [75, 75, 300]
    >>> probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    >>> G, coms = SBM(sizes, probs, seed=0)

    :References:

    Holland, P. W., Laskey, K. B., & Leinhardt, S., “Stochastic blockmodels: First steps”, Social networks, 5(2), 109-137, 1983.

    .. note:: Reference implementation: https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.stochastic_block_model.html#networkx.generators.community.stochastic_block_model
    """

    from networkx.generators.community import stochastic_block_model

    G = stochastic_block_model(
        sizes=sizes,
        p=p,
        nodelist=nodelist,
        seed=seed,
        directed=directed,
        selfloops=selfloops,
        sparse=sparse,
    )
    communities = defaultdict(list)
    for n, data in G.nodes(data=True):
        communities[data["block"]].append(n)

    coms = NodeClustering(
        list(communities.values()),
        G,
        "SBM",
        method_parameters={
            "sizes": sizes,
            "p": p,
            "nodelist": nodelist,
            "seed": seed,
            "directed": directed,
            "selfloops": selfloops,
            "sparse": sparse,
        },
    )

    return G, coms
