from demon import Demon
from angel import Angel
from nclib.community.algorithms.NodePerception import NodePerception
from nclib.community.algorithms import OSSE
import networkx as nx
import igraph as ig
from nclib import NodeClustering
from nclib.utils import suppress_stdout, convert_graph_formats


def ego_networks(g, level=1):
    """

    :param g:
    :param level:
    :return:
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = []
    for n in g.nodes():
        coms.append(list(nx.ego_graph(g, n, radius=level).nodes()))
    return NodeClustering(coms, g, "Ego Network", method_parameters={"level": 1})


def demon(g, epsilon, min_com_size=3):
    """
    Demon is a node-centric bottom-up community discovery algorithm.
    It leverages ego-network structures and overlapping label propagation to identify micro-scale communities that are subsequently merged in mesoscale ones.

    :param g: a networkx/igraph object
    :param epsilon: merging threshold in [0,1], default 0.25.
    :param min_com_size: minimum community size, default 3.
    :return: a list of overlapping communities


    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.demon(g, min_com_size=3, epsilon=0.25)

    :References:

    1. Coscia, M., Rossetti, G., Giannotti, F., & Pedreschi, D. (2012, August). **Demon: a local-first discovery method for overlapping communities.** In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 615-623). ACM.

    2. Coscia, M., Rossetti, G., Giannotti, F., & Pedreschi, D. (2014). **Uncovering hierarchical and overlapping communities with a local-first approach.** ACM Transactions on Knowledge Discovery from Data (TKDD), 9(1), 6.
    """

    g = convert_graph_formats(g, nx.Graph)

    with suppress_stdout():
        dm = Demon(graph=g, epsilon=epsilon, min_community_size=min_com_size)
        coms = dm.execute()

    return NodeClustering(coms, g, "DEMON", method_parameters={"epsilon": epsilon, "min_com_size": min_com_size},
                          overlap=True)


def angel(g, threshold, min_community_size=3):
    """
    Angel is a node-centric bottom-up community discovery algorithm.
    It leverages ego-network structures and overlapping label propagation to identify micro-scale communities that are subsequently merged in mesoscale ones.
    Angel is the, faster, successor of Demon.

    :param g: a networkx/igraph object
    :param threshold: merging threshold in [0,1].
    :param min_community_size: minimum community size, default 3.
    :return: a list of overlapping communities

    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = community.angel(g, min_com_size=3, threshold=0.25)

    :References:

    1. Rossetti G. **Angel: efficient, and effective, node-centric community discovery in static and dynamic networks.**
    """

    g = convert_graph_formats(g, ig.Graph)

    a = Angel(graph=g, min_comsize=min_community_size, threshold=threshold, save=False)
    coms = a.execute()

    return NodeClustering(list(coms.values()), g, "ANGEL", method_parameters={"threshold": threshold,
                                                                              "min_community_size": min_community_size},
                          overlap=True)


def node_perception(g, threshold, overlap_threshold, min_comm_size=3):
    """

    :param g: a networkx/igraph object
    :param threshold:
    :param overlap_threshold:
    :param min_comm_size:
    :return: a list of overlapping communities
    """
    g = convert_graph_formats(g, nx.Graph)

    with suppress_stdout():
        np = NodePerception(g, sim_threshold=threshold, overlap_threshold=overlap_threshold, min_comm_size=min_comm_size)
        coms = np.execute()

    return NodeClustering(coms, g, "Node Perception", method_parameters={"threshold": threshold,
                                                                         "overlap_threshold": overlap_threshold,
                                                                         "min_com_size": min_comm_size},
                          overlap=True)


def overlapping_seed_set_expansion(g, seeds, ninf=False, expansion='ppr', stopping='cond', nworkers=1,
                                   nruns=13, alpha=0.99, maxexpand=float('INF'), delta=0.2):
    """


    :param g: a networkx/igraph object
    :param seeds: Node list
    :param ninf: Neighbourhood Inflation parameter (boolean)
    :param expansion: Seed expansion: ppr or vppr
    :param stopping: Stopping criteria: cond
    :param nworkers: Number of Workers: default 1
    :param nruns: Number of runs: default 13
    :param alpha: alpha value for Personalized PageRank expansion: default 0.99
    :param maxexpand: Maximum expansion allowed for approximate ppr: default INF
    :param delta: Minimum distance parameter for near duplicate communities: default 0.2
    :return: a community


    :Example:

    >>> from nclib import community
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = community.overlapping_seed_set_expansion(g)

    :References:

    1.Whang, J. J., Gleich, D. F., & Dhillon, I. S. (2013, October). **Overlapping community detection using seed set expansion.** In Proceedings of the 22nd ACM international conference on Conference on information & knowledge management (pp. 2099-2108). ACM.

    """

    g = convert_graph_formats(g, nx.Graph)

    if ninf:
        seeds = OSSE.neighbor_inflation(g, seeds)

    communities = OSSE.growclusters(g, seeds, expansion, stopping, nworkers, nruns, alpha, maxexpand, False)
    communities = OSSE.remove_duplicates(g, communities, delta)
    communities = list(communities)
    return NodeClustering(communities, g, "Overlapping SSE", method_parameters={"seeds": seeds, "ninf": ninf,
                                                                                "expansion": expansion,
                                                                                "stopping": stopping,
                                                                                "nworkers": nworkers,
                                                                                "nruns": nruns, "alpha":alpha,
                                                                                "maxexpand": maxexpand,
                                                                                "delta": delta},
                          overlap=True)
