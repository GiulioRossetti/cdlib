import networkx as nx
from cdlib.utils import convert_graph_formats
from collections import namedtuple
import numpy as np
import scipy
from cdlib.evaluation.internal.link_modularity import cal_modularity
import Eva
from typing import Callable
from collections import defaultdict

__all__ = [
    "FitnessResult",
    "link_modularity",
    "normalized_cut",
    "internal_edge_density",
    "average_internal_degree",
    "fraction_over_median_degree",
    "expansion",
    "cut_ratio",
    "edges_inside",
    "flake_odf",
    "avg_odf",
    "max_odf",
    "triangle_participation_ratio",
    "modularity_density",
    "z_modularity",
    "erdos_renyi_modularity",
    "newman_girvan_modularity",
    "significance",
    "surprise",
    "conductance",
    "size",
    "avg_embeddedness",
    "scaled_density",
    "avg_distance",
    "hub_dominance",
    "avg_transitivity",
    "purity",
    "modularity_overlap",
]

FitnessResult = namedtuple("FitnessResult", "min max score std")
FitnessResult.__new__.__defaults__ = (None,) * len(FitnessResult._fields)


def __median(lst: list) -> int:
    return np.median(np.array(lst))


def __out_degree_fraction(g: nx.Graph, coms: list) -> list:
    nds = []
    for n in coms:
        nds.append(g.degree(n) - coms.degree(n))
    return nds


def __quality_indexes(
    graph: nx.Graph,
    communities: object,
    scoring_function: Callable[[object, object], float],
    summary: bool = True,
) -> object:
    """

    :param graph: NetworkX/igraph graph
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-communitys ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in communities.communities:
        community = nx.subgraph(graph, com)
        values.append(scoring_function(graph, community))

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def size(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """Size is the number of nodes in the community

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> sz = evaluation.size(g,communities)
    """

    return __quality_indexes(graph, communities, lambda g, com: len(com), **kwargs)


def scaled_density(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """Scaled density.

    The scaled density of a community is defined as the ratio of the community density w.r.t. the complete graph density.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> scd = evaluation.scaled_density(g,communities)
    """

    return __quality_indexes(
        graph,
        communities,
        lambda graph, coms: nx.density(nx.subgraph(graph, coms)) / nx.density(graph),
        **kwargs
    )


def avg_distance(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """Average distance.

    The average distance of a community is defined average path length across all possible pair of nodes composing it.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> scd = evaluation.avg_distance(g,communities)
    """

    return __quality_indexes(
        graph,
        communities,
        lambda graph, coms: nx.average_shortest_path_length(nx.subgraph(graph, coms)),
        **kwargs
    )


def hub_dominance(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """Hub dominance.

    The hub dominance of a community is defined as the ratio of the degree of its most connected node w.r.t. the theoretically maximal degree within the community.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> scd = evaluation.hub_dominance(g,communities)
    """

    return __quality_indexes(
        graph,
        communities,
        lambda graph, coms: max(
            [x[1] for x in list(nx.degree(nx.subgraph(graph, coms)))]
        )
        / (len(coms) - 1),
        **kwargs
    )


def avg_transitivity(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """Average transitivity.

    The average transitivity of a community is defined the as the average clustering coefficient of its nodes w.r.t. their connection within the community itself.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> scd = evaluation.avg_transitivity(g,communities)
    """

    return __quality_indexes(
        graph,
        communities,
        lambda graph, coms: nx.average_clustering(nx.subgraph(graph, coms)),
        **kwargs
    )


def avg_embeddedness(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """Average embeddedness of nodes within the community.

    The embeddedness of a node n w.r.t. a community C is the ratio of its degree within the community and its overall degree.

    .. math:: emb(n,C) = \\frac{k_n^C}{k_n}

    The average embeddedness of a community C is:

    .. math:: avg_embd(c) = \\frac{1}{|C|} \sum_{i \in C} \\frac{k_n^C}{k_n}

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> ave = evaluation.avg_embeddedness(g,communities)

    :References:


    """

    return __quality_indexes(
        graph,
        communities,
        lambda g, com: np.mean(
            [float(nx.degree(nx.subgraph(g, com))[n]) / nx.degree(g)[n] for n in com]
        ),
        **kwargs
    )


def normalized_cut(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Normalized variant of the Cut-Ratio

    .. math:: f(S) = \\frac{c_S}{2m_S+c_S} + \\frac{c_S}{2(m−m_S )+c_S}

    where :math:`m` is the number of graph edges, :math:`m_S` is the number of community internal edges and :math:`c_S` is the number of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.


    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.normalized_cut(g,communities)

    :References:

    1.Shi, J., Malik, J.: Normalized cuts and image segmentation. Departmental Papers (CIS), 107 (2000)
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ms = len(coms.edges())
        edges_outside = 0
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += 1
        try:
            ratio = (float(edges_outside) / ((2 * ms) + edges_outside)) + float(
                edges_outside
            ) / (2 * (len(graph.edges()) - ms) + edges_outside)
        except:
            ratio = 0
        values.append(ratio)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def internal_edge_density(
    graph: nx.Graph, community: object, summary: bool = True
) -> object:
    """The internal density of the community set.

     .. math:: f(S) = \\frac{m_S}{n_S(n_S−1)/2}

     where :math:`m_S` is the number of community internal edges and :math:`n_S` is the number of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.internal_edge_density(g,communities)


    :References:

    1. Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    """
    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ms = len(coms.edges())
        ns = len(coms.nodes())
        try:
            internal_density = float(ms) / (float(ns * (ns - 1)) / 2)
        except:
            internal_density = 0
        values.append(internal_density)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def average_internal_degree(
    graph: nx.Graph, community: object, summary: bool = True
) -> object:
    """The average internal degree of the community set.

    .. math:: f(S) = \\frac{2m_S}{n_S}

     where :math:`m_S` is the number of community internal edges and :math:`n_S` is the number of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.average_internal_degree(g,communities)

    :References:

    1. Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ms = len(coms.edges())
        ns = len(coms.nodes())
        try:
            avg_id = float(2 * ms) / ns
        except:
            avg_id = 0
        values.append(avg_id)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def fraction_over_median_degree(
    graph: nx.Graph, community: object, summary: bool = True
) -> object:
    """Fraction of community nodes of having internal degree higher than the median degree value.

    .. math:: f(S) = \\frac{|\{u: u \\in S,| \{(u,v): v \\in S\}| > d_m\}| }{n_S}


    where :math:`d_m` is the internal degree median value

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.fraction_over_median_degree(g,communities)

    :References:

    1. Yang, J., Leskovec, J.: Defining and evaluating network communities based on ground-truth. Knowledge and Information Systems 42(1), 181–213 (2015)
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ns = coms.number_of_nodes()
        degs = coms.degree()

        med = __median([d[1] for d in degs])
        above_med = len([d[0] for d in degs if d[1] > med])
        try:
            ratio = float(above_med) / ns
        except:
            ratio = 0
        values.append(ratio)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def expansion(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Number of edges per community node that point outside the cluster.

    .. math:: f(S) = \\frac{c_S}{n_S}

    where :math:`n_S` is the number of edges on the community boundary, :math:`c_S` is the number of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.expansion(g,communities)

    :References:

    1. Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ns = len(coms.nodes())
        edges_outside = 0
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += 1
        try:
            exp = float(edges_outside) / ns
        except:
            exp = 0
        values.append(exp)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def cut_ratio(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Fraction of existing edges (out of all possible edges) leaving the community.

    ..math:: f(S) = \\frac{c_S}{n_S (n − n_S)}

    where :math:`c_S` is the number of community nodes and, :math:`n_S` is the number of edges on the community boundary

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.cut_ratio(g,communities)

    :References:

    1. Fortunato, S.: Community detection in graphs. Physics reports 486(3-5), 75–174 (2010)
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ns = len(coms.nodes())
        edges_outside = 0
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += 1
        try:
            ratio = float(edges_outside) / (ns * (len(graph.nodes()) - ns))
        except:
            ratio = 0
        values.append(ratio)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def edges_inside(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Number of edges internal to the community.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.edges_inside(g,communities)

    :References:

    1. Radicchi, F., Castellano, C., Cecconi, F., Loreto, V., & Parisi, D. (2004). Defining and identifying communities in networks. Proceedings of the National Academy of Sciences, 101(9), 2658-2663.
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)
        values.append(coms.number_of_edges())

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def conductance(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Fraction of total edge volume that points outside the community.

    .. math:: f(S) = \\frac{c_S}{2 m_S+c_S}

    where :math:`c_S` is the number of community nodes and, :math:`m_S` is the number of community edges

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.conductance(g,communities)

    :References:

    1.Shi, J., Malik, J.: Normalized cuts and image segmentation. Departmental Papers (CIS), 107 (2000)
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ms = len(coms.edges())
        edges_outside = 0
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += 1
        try:
            ratio = float(edges_outside) / ((2 * ms) + edges_outside)
        except:
            ratio = 0
        values.append(ratio)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def max_odf(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Maximum fraction of edges of a node of a community that point outside the community itself.

    .. math:: max_{u \\in S} \\frac{|\{(u,v)\\in E: v \\not\\in S\}|}{d(u)}

    where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S` and :math:`d(u)` is the degree of :math:`u`

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.max_odf(g,communities)

    :References:

    1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)
        values.append(max(__out_degree_fraction(graph, coms)))

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def avg_odf(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Average fraction of edges of a node of a community that point outside the community itself.

    .. math:: \\frac{1}{n_S} \\sum_{u \\in S} \\frac{|\{(u,v)\\in E: v \\not\\in S\}|}{d(u)}

    where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S`, :math:`d(u)` is the degree of :math:`u` and :math:`n_S` is the set of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.avg_odf(g,communities)

    :References:

    1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)
        values.append(float(sum(__out_degree_fraction(graph, coms))) / len(coms))

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def flake_odf(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Fraction of nodes in S that have fewer edges pointing inside than to the outside of the community.

    .. math:: f(S) = \\frac{| \{ u:u \in S,| \{(u,v) \in E: v \in S \}| < d(u)/2 \}|}{n_S}

    where :math:`E` is the graph edge set, :math:`v` is a node in :math:`S`, :math:`d(u)` is the degree of :math:`u` and :math:`n_S` is the set of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.flake_odf(g,communities)

    :References:

    1. Flake, G.W., Lawrence, S., Giles, C.L., et al.: Efficient identification of web communities. In: KDD, vol. 2000, pp. 150–160 (2000)
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)
        df = 0
        for n in coms:
            fr = coms.degree(n) - (graph.degree(n) - coms.degree(n))
            if fr < 0:
                df += 1
        score = float(df) / len(coms)
        values.append(score)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def triangle_participation_ratio(
    graph: nx.Graph, community: object, summary: bool = True
) -> object:
    """Fraction of community nodes that belong to a triad.

    .. math:: f(S) = \\frac{ | \{ u: u \in S,\{(v,w):v, w \in S,(u,v) \in E,(u,w) \in E,(v,w) \in E \} \\not = \\emptyset \} |}{n_S}

    where :math:`n_S` is the set of community nodes.

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.triangle_participation_ratio(g,communities)

    :References:

    1. Yang, J., Leskovec, J.: Defining and evaluating network communities based on ground-truth. Knowledge and Information Systems 42(1), 181–213 (2015)
    """

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)
        cls = nx.triangles(coms)
        nc = [n for n in cls if cls[n] > 0]
        score = float(len(nc)) / len(coms)
        values.append(score)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def link_modularity(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """
    Quality function designed for directed graphs with overlapping communities.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.link_modularity(g,communities)

    :References:

    1. Nicosia, V., Mangioni, G., Carchiolo, V., Malgeri, M.: Extending the definition of modularity to directed graphs with overlapping communities. Journal of Statistical Mechanics: Theory and Experiment 2009(03), 03024 (2009)

    """

    graph = convert_graph_formats(graph, nx.Graph)

    return FitnessResult(score=cal_modularity(graph, communities.communities))


def newman_girvan_modularity(
    graph: nx.Graph, communities: object, **kwargs: dict
) -> object:
    """Difference the fraction of intra community edges of a partition with the expected number of such edges if distributed according to a null model.

    In the standard version of modularity, the null model preserves the expected degree sequence of the graph under consideration. In other words, the modularity compares the real network structure with a corresponding one where nodes are connected without any preference about their neighbors.

    .. math:: Q(S) = \\frac{1}{m}\\sum_{c \\in S}(m_S - \\frac{(2 m_S + l_S)^2}{4m})

    where :math:`m` is the number of graph edges, :math:`m_S` is the number of community edges, :math:`l_S` is the number of edges from nodes in S to nodes outside S.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object


    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.newman_girvan_modularity(g,communities)

    :References:

    1. Newman, M.E.J. & Girvan, M. `Finding and evaluating community structure in networks. <https://www.ncbi.nlm.nih.gov/pubmed/14995526/>`_ Physical Review E 69, 26113(2004).
    """

    graph = convert_graph_formats(graph, nx.Graph)
    coms = {}
    for cid, com in enumerate(communities.communities):
        for node in com:
            coms[node] = cid

    inc = dict([])
    deg = dict([])
    links = graph.size(weight="weight")
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        try:
            com = coms[node]
            deg[com] = deg.get(com, 0.0) + graph.degree(node, weight="weight")
            for neighbor, dt in graph[node].items():
                weight = dt.get("weight", 1)
                if coms[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.0) + float(weight)
                    else:
                        inc[com] = inc.get(com, 0.0) + float(weight) / 2.0
        except:
            pass

    res = 0.0
    for com in set(coms.values()):
        res += (inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2

    return FitnessResult(score=res)


def erdos_renyi_modularity(
    graph: nx.Graph, communities: object, **kwargs: dict
) -> object:
    """Erdos-Renyi modularity is a variation of the Newman-Girvan one.
    It assumes that vertices in a network are connected randomly with a constant probability :math:`p`.

    .. math:: Q(S) = \\frac{1}{m}\\sum_{c \\in S} (m_S − \\frac{mn_S(n_S −1)}{n(n−1)})

    where :math:`m` is the number of graph edges, :math:`m_S` is the number of community edges, :math:`l_S` is the number of edges from nodes in S to nodes outside S.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.erdos_renyi_modularity(g,communities)

    :References:

    1. Erdos, P., & Renyi, A. (1959). `On random graphs I. <https://gnunet.org/sites/default/files/Erd%C5%91s%20%26%20R%C3%A9nyi%20-%20On%20Random%20Graphs.pdf/>`_ Publ. Math. Debrecen, 6, 290-297.
    """
    graph = convert_graph_formats(graph, nx.Graph)
    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    q = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        nc = c.number_of_nodes()
        q += mc - (m * nc * (nc - 1)) / (n * (n - 1))

    return FitnessResult(score=(1 / m) * q)


def modularity_density(
    graph: nx.Graph, communities: object, lmbd: float = 0.5, **kwargs: dict
) -> object:
    """The modularity density is one of several propositions that envisioned to palliate the resolution limit issue of modularity based measures.
    The idea of this metric is to include the information about community size into the expected density of community to avoid the negligence of small and dense communities.
    For each community :math:`C` in partition :math:`S`, it uses the average modularity degree calculated by :math:`d(C) = d^{int(C)} − d^{ext(C)}` where :math:`d^{int(C)}` and :math:`d^{ext(C)}` are the average internal and external degrees of :math:`C` respectively to evaluate the fitness of :math:`C` in its network.
    Finally, the modularity density can be calculated as follows:

    .. math:: Q(S) = \\sum_{C \\in S} \\frac{1}{n_C} ( \\sum_{i \\in C} 2 * \lambda * k^{int}_{iC} - \\sum_{i \\in C} 2 * (1 - \lambda) * k^{out}_{iC})

    where :math:`n_C` is the number of nodes in C, :math:`k^{int}_{iC}` is the degree of node i within :math:`C`, :math:`k^{out}_{iC}` is the deree of node i outside :math:`C` and :math:`\lambda` is a paramter that allows for tuning the measure resolution (its default value, 0.5, computes the standard modularity density score).

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param lmbd: resolution parameter, float in [0,1]. Default 0.5.
    :return: FitnessResult object


    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.modularity_density(g,communities)

    :References:

    1. Zhang, S., Ning, XM., Ding, C. et al. Determining modular organization of protein interaction networks by maximizing modularity density. <https://doi.org/10.1186/1752-0509-4-S2-S10>`_ BMC Syst Biol 4, S10 (2010).
    """
    graph = convert_graph_formats(graph, nx.Graph)
    q = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)

        nc = c.number_of_nodes()
        dint = []
        dext = []
        for node in c:
            dint.append(c.degree(node))
            dext.append(graph.degree(node) - c.degree(node))

        try:
            q += (1 / nc) * (
                (2 * lmbd * np.sum(dint)) - (2 * (1 - lmbd) * np.sum(dext))
            )
        except ZeroDivisionError:
            pass

    return FitnessResult(score=q)


def z_modularity(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """Z-modularity is another variant of the standard modularity proposed to avoid the resolution limit.
    The concept of this version is based on an observation that the difference between the fraction of edges inside communities and the expected number of such edges in a null model should not be considered as the only contribution to the final quality of community structure.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.z_modularity(g,communities)


    :References:

    1. Miyauchi, Atsushi, and Yasushi Kawase. `Z-score-based modularity for community detection in networks. <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147805/>`_ PloS one 11.1 (2016): e0147805.
    """
    graph = convert_graph_formats(graph, nx.Graph)
    m = graph.number_of_edges()

    mmc = 0
    dc2m = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        dc = 0

        for node in c:
            dc += graph.degree(node)

        mmc += mc / m
        dc2m += (dc / (2 * m)) ** 2

    res = 0
    try:
        res = (mmc - dc2m) / np.sqrt(dc2m * (1 - dc2m))
    except ZeroDivisionError:
        pass

    return FitnessResult(score=res)


def surprise(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """Surprise is statistical approach proposes a quality metric assuming that edges between vertices emerge randomly according to a hyper-geometric distribution.

    According to the Surprise metric, the higher the score of a partition, the less likely it is resulted from a random realization, the better the quality of the community structure.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.surprise(g,communities)

    :References:

    1. Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015). `Detecting communities using asymptotical surprise. <https://link.aps.org/doi/10.1103/PhysRevE.92.022816/>`_ Physical Review E, 92(2), 022816.
    """
    graph = convert_graph_formats(graph, nx.Graph)
    m = graph.number_of_edges()
    n = graph.number_of_nodes()

    q = 0
    qa = 0
    sp = 0

    for community in communities.communities:
        c = nx.subgraph(graph, community)
        mc = c.number_of_edges()
        nc = c.number_of_nodes()

        q += mc
        qa += scipy.special.comb(nc, 2, exact=True)
    try:
        q = q / m
        qa = qa / scipy.special.comb(n, 2, exact=True)

        sp = m * (q * np.log(q / qa) + (1 - q) * np.log((1 - q) / (1 - qa)))
    except ZeroDivisionError:
        pass

    return FitnessResult(score=sp)


def significance(graph: nx.Graph, communities: object, **kwargs: dict) -> object:
    """Significance estimates how likely a partition of dense communities appear in a random graph.

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.significance(g,communities)

    :References:

    1. Traag, V. A., Aldecoa, R., & Delvenne, J. C. (2015). `Detecting communities using asymptotical surprise. <https://link.aps.org/doi/10.1103/PhysRevE.92.022816/>`_ Physical Review E, 92(2), 022816.
    """
    graph = convert_graph_formats(graph, nx.Graph)
    m = graph.number_of_edges()

    binom = scipy.special.comb(m, 2, exact=True)
    p = m / binom

    q = 0

    for community in communities.communities:
        try:
            c = nx.subgraph(graph, community)
            nc = c.number_of_nodes()
            mc = c.number_of_edges()

            binom_c = scipy.special.comb(nc, 2, exact=True)
            pc = mc / binom_c

            q += binom_c * (pc * np.log(pc / p) + (1 - pc) * np.log((1 - pc) / (1 - p)))
        except ZeroDivisionError:
            pass
    return FitnessResult(score=q)


def purity(communities: object) -> FitnessResult:
    """Purity is the product of the frequencies of the most frequent labels carried by the nodes within the communities

    :param communities: AttrNodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import eva
    >>> from cdlib import evaluation
    >>> import random
    >>> l1 = ['A', 'B', 'C', 'D']
    >>> l2 = ["E", "F", "G"]
    >>> g = nx.barabasi_albert_graph(100, 5)
    >>> labels=dict()
    >>> for node in g.nodes():
    >>>    labels[node]={"l1":random.choice(l1), "l2":random.choice(l2)}
    >>> communities = eva(g_attr, labels, alpha=0.5)
    >>> pur = evaluation.purity(communities)

    :References:

    1. Citraro, Salvatore, and Giulio Rossetti. "Eva: Attribute-Aware Network Segmentation." International Conference on Complex Networks and Their Applications. Springer, Cham, 2019.
    """

    pur = Eva.purity(communities.coms_labels)
    return FitnessResult(score=pur)


def modularity_overlap(
    graph: nx.Graph, communities: object, weight: str = None
) -> FitnessResult:
    """Determines the Overlapping Modularity of a partition C on a graph G.

    Overlapping Modularity is defined as

     .. math:: M_{c_{r}}^{ov} = \\sum_{i \\in c_{r}} \\frac{\\sum_{j \\in c_{r}, i \\neq j}a_{ij} - \\sum_{j \\not \\in c_{r}}a_{ij}}{d_{i} \\cdot s_{i}} \\cdot \\frac{n_{c_{r}}^{e}}{n_{c_{r}} \\cdot \\binom{n_{c_{r}}}{2}}

    :param graph: a networkx/igraph object
    :param communities: NodeClustering object
    :param weight: label identifying the edge weight parameter name (if present), default None
    :return: FitnessResult object


    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.modularity_overlap(g, communities)

    :References:

    1. A. Lazar, D. Abel and T. Vicsek, "Modularity measure of networks with overlapping communities"  EPL, 90 (2010) 18001 doi: 10.1209/0295-5075/90/18001

    .. note:: Reference implementation: https://github.com/aonghus/nxtools/blob/master/nxtools/algorithms/community/quality.py
    """

    graph = convert_graph_formats(graph, nx.Graph)

    affiliation_dict = defaultdict(list)
    for cid, coms in enumerate(communities.communities):
        for n in coms:
            affiliation_dict[n].append(cid)

    mOvTotal = 0

    for nodes in communities.communities:
        nCommNodes = len(nodes)

        # the contribution of communities with 1 node is 0
        if nCommNodes <= 1:
            continue

        nInwardEdges = 0
        commStrength = 0

        for node in nodes:
            degree, inwardEdges, outwardEdges = 0, 0, 0
            for u, v, data in graph.edges(node, data=True):
                w = data.get(weight, 1)
                degree += w
                if v in nodes:
                    inwardEdges += w
                    nInwardEdges += 1
                else:
                    outwardEdges += w

            affiliationCount = len(affiliation_dict[node])
            commStrength += (inwardEdges - outwardEdges) / (degree * affiliationCount)

        binomC = nCommNodes * (nCommNodes - 1)
        v1 = commStrength / nCommNodes
        v2 = nInwardEdges / binomC
        mOv = v1 * v2
        mOvTotal += mOv

    score = mOvTotal / len(communities.communities)
    return FitnessResult(score=score)
