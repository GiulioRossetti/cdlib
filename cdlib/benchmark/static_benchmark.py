from networkx.generators.community import LFR_benchmark_graph
from cdlib.benchmark.internal import xmark
import cdlib
from cdlib import NodeClustering
from collections import defaultdict

__all__ = ["LFR", "XMark"]


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
