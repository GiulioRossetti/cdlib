try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None
try:
    from angel import Angel
except ModuleNotFoundError:
    Angel = None

try:
    from ASLPAw_package import ASLPAw
except ModuleNotFoundError:
    ASLPAw = None

from random import sample
from demon import Demon
from cdlib.algorithms.internal.NodePerception import NodePerception
from cdlib.algorithms.internal import OSSE
import networkx as nx
import numpy as np
from collections import defaultdict
from cdlib import NodeClustering
from cdlib.utils import suppress_stdout, convert_graph_formats, nx_node_integer_mapping
from cdlib.algorithms.internal.CONGO import Congo_
from cdlib.algorithms.internal.CONGA import Conga_
from cdlib.algorithms.internal.LAIS2_nx import LAIS2
from cdlib.algorithms.internal.lfm import LFM_nx
from cdlib.algorithms.internal import LEMON
from cdlib.algorithms.internal.SLPA_nx import slpa_nx
from cdlib.algorithms.internal.multicom import MultiCom
from cdlib.algorithms.internal.PercoMCV import percoMVC
from cdlib.algorithms.internal.LPAM import LPAM
from cdlib.algorithms.internal.core_exp import findCommunities as core_exp_find
from karateclub import DANMF, EgoNetSplitter, NNSED, MNMF, BigClam, SymmNMF
from cdlib.algorithms.internal.weightedCommunity import weightedCommunity
from cdlib.algorithms.internal.LPANNI import LPANNI, GraphGenerator
from cdlib.algorithms.internal.DCS import main_dcs
from cdlib.algorithms.internal.UMSTMO import UMSTMO
from cdlib.algorithms.internal.walkscan import WalkSCAN
from cdlib.algorithms.internal.IPCA import i_pca
from cdlib.algorithms.internal.DPCLUS import dp_clus
from cdlib.algorithms.internal.COACH import co_ach
from cdlib.algorithms.internal.graph_entropy import graphentropy
from cdlib.algorithms.internal.EBGC import EBGC
from cdlib.algorithms.internal.EnDNTM import (
    endntm_find_overlap_cluster,
    endntm_evalFuction,
)

__all__ = [
    "ego_networks",
    "demon",
    "angel",
    "node_perception",
    "overlapping_seed_set_expansion",
    "kclique",
    "lfm",
    "lais2",
    "congo",
    "conga",
    "lemon",
    "slpa",
    "multicom",
    "big_clam",
    "danmf",
    "egonet_splitter",
    "nnsed",
    "mnmf",
    "aslpaw",
    "percomvc",
    "wCommunity",
    "core_expansion",
    "lpanni",
    "lpam",
    "dcs",
    "umstmo",
    "symmnmf",
    "walkscan",
    "endntm",
    "ipca",
    "dpclus",
    "coach",
    "graph_entropy",
    "ebgc",
]


def ego_networks(g_original: object, level: int = 1) -> NodeClustering:
    """
    Ego-networks returns overlapping communities centered at each nodes within a given radius.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param level: extrac communities with all neighbors of distance<=level from a node. Deafault 1
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.ego_networks(G)
    """

    g = convert_graph_formats(g_original, nx.Graph)

    coms = []
    for n in g.nodes():
        coms.append(list(nx.ego_graph(g, n, radius=level).nodes()))
    return NodeClustering(
        coms, g_original, "Ego Network", method_parameters={"level": 1}, overlap=True
    )


def demon(g_original: object, epsilon: float, min_com_size: int = 3) -> NodeClustering:
    """
    Demon is a node-centric bottom-up overlapping community discovery algorithm.
    It leverages ego-network structures and overlapping label propagation to identify micro-scale communities that are subsequently merged in mesoscale ones.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param epsilon: merging threshold in [0,1], default 0.25.
    :param min_com_size: minimum community size, default 3.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.demon(G, min_com_size=3, epsilon=0.25)

    :References:

    1. Coscia, M., Rossetti, G., Giannotti, F., & Pedreschi, D. (2012, August). `Demon: a local-first discovery method for overlapping communities. <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.721.1788&rep=rep1&type=pdf/>`_ In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 615-623). ACM.

    2. Coscia, M., Rossetti, G., Giannotti, F., & Pedreschi, D. (2014). `Uncovering hierarchical and overlapping communities with a local-first approach. <https://dl.acm.org/citation.cfm?id=2629511/>`_ ACM Transactions on Knowledge Discovery from Data (TKDD), 9(1), 6.

    .. note:: Reference implementation: https://github.com/GiulioRossetti/DEMON

    """

    g = convert_graph_formats(g_original, nx.Graph)

    with suppress_stdout():
        dm = Demon(graph=g, epsilon=epsilon, min_community_size=min_com_size)
        coms = dm.execute()
        coms = [list(c) for c in coms]

    return NodeClustering(
        coms,
        g_original,
        "DEMON",
        method_parameters={"epsilon": epsilon, "min_com_size": min_com_size},
        overlap=True,
    )


def angel(
    g_original: object, threshold: float, min_community_size: int = 3
) -> NodeClustering:
    """
    Angel is a node-centric bottom-up community discovery algorithm.
    It leverages ego-network structures and overlapping label propagation to identify micro-scale communities that are subsequently merged in mesoscale ones.
    Angel is the, faster, successor of Demon.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param threshold: merging threshold in [0,1].
    :param min_community_size: minimum community size, default 3.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.angel(G, min_com_size=3, threshold=0.25)

    :References:

    1. Rossetti, Giulio. "Exorcising the Demon: Angel, Efficient Node-Centric Community Discovery." International Conference on Complex Networks and Their Applications. Springer, Cham, 2019.

    .. note:: Reference implementation: https://github.com/GiulioRossetti/ANGEL
    """
    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )
    if Angel is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install angel-cd library to use the selected feature (likely pip install angel-cd). If using a notebook, you need also to restart your runtime/kernel."
        )

    g = convert_graph_formats(g_original, ig.Graph)
    with suppress_stdout():
        a = Angel(
            graph=g, min_comsize=min_community_size, threshold=threshold, save=False
        )
        coms = a.execute()

    return NodeClustering(
        list(coms.values()),
        g_original,
        "ANGEL",
        method_parameters={
            "threshold": threshold,
            "min_community_size": min_community_size,
        },
        overlap=True,
    )


def node_perception(
    g_original: object,
    threshold: float,
    overlap_threshold: float,
    min_comm_size: int = 3,
) -> NodeClustering:
    """Node perception is based on the idea of joining together small sets of nodes.
    The algorithm first identifies sub-communities corresponding to each node’s perception of the network around it.
    To perform this step, it considers each node individually, and partition that node’s neighbors into communities using some existing community detection method.
    Next, it creates a new network in which every node corresponds to a sub-community, and two nodes are linked if their associated sub-communities overlap by at least some threshold amount.
    Finally, the algorithm identifies overlapping communities in this new network, and for every such community, merge together the associated sub-communities to identify communities in the original network.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param threshold: the tolerance required in order to merge communities
    :param overlap_threshold: the overlap tolerance
    :param min_comm_size: minimum community size default 3
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.node_perception(G, threshold=0.25, overlap_threshold=0.25)

    :References:

    Sucheta Soundarajan and John E. Hopcroft. 2015. `Use of Local Group Information to Identify Communities in Networks. <https://dl.acm.org/citation.cfm?id=2737800.2700404/>`_ ACM Trans. Knowl. Discov. Data 9, 3, Article 21 (April 2015), 27 pages. DOI=http://dx.doi.org/10.1145/2700404

    """
    g = convert_graph_formats(g_original, nx.Graph)
    tp = type(list(g.nodes())[0])

    with suppress_stdout():
        np = NodePerception(
            g,
            sim_threshold=threshold,
            overlap_threshold=overlap_threshold,
            min_comm_size=min_comm_size,
        )
        coms = np.execute()
        if tp != str:
            communities = []
            for c in coms:
                c = list(map(tp, c))
                communities.append(c)
            coms = communities

    return NodeClustering(
        coms,
        g_original,
        "Node Perception",
        method_parameters={
            "threshold": threshold,
            "overlap_threshold": overlap_threshold,
            "min_com_size": min_comm_size,
        },
        overlap=True,
    )


def overlapping_seed_set_expansion(
    g_original: object,
    seeds: list,
    ninf: bool = False,
    expansion: str = "ppr",
    stopping: str = "cond",
    nworkers: int = 1,
    nruns: int = 13,
    alpha: float = 0.99,
    maxexpand: float = float("INF"),
    delta: float = 0.2,
) -> NodeClustering:
    """
    OSSE is an overlapping community detection algorithm optimizing the conductance community score
    The algorithm uses a seed set expansion approach; the key idea is to find good seeds, and then expand these seed sets using the personalized PageRank clustering procedure.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param seeds: Node list
    :param ninf: Neighbourhood Inflation parameter (boolean)
    :param expansion: Seed expansion: ppr or vppr
    :param stopping: Stopping criteria: cond
    :param nworkers: Number of Workers: default 1
    :param nruns: Number of runs: default 13
    :param alpha: alpha value for Personalized PageRank expansion: default 0.99
    :param maxexpand: Maximum expansion allowed for approximate ppr: default INF
    :param delta: Minimum distance parameter for near duplicate communities: default 0.2
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.overlapping_seed_set_expansion(G)

    :References:

    1.Whang, J. J., Gleich, D. F., & Dhillon, I. S. (2013, October). `Overlapping community detection using seed set expansion. <http://www.cs.utexas.edu/~inderjit/public_papers/overlapping_commumity_cikm13.pdf/>`_ In Proceedings of the 22nd ACM international conference on Conference on information & knowledge management (pp. 2099-2108). ACM.

    .. note:: Reference implementation: https://github.com/pratham16/algorithms-detection-by-seed-expansion
    """

    g = convert_graph_formats(g_original, nx.Graph)

    g, maps = nx_node_integer_mapping(g)
    if maps is not None:
        rev_map = {v: k for k, v in maps.items()}
        seeds = [rev_map[s] for s in seeds]

    if ninf:
        seeds = OSSE.neighbor_inflation(g, seeds)

    communities = OSSE.growclusters(
        g, seeds, expansion, stopping, nworkers, nruns, alpha, maxexpand, False
    )
    communities = OSSE.remove_duplicates(g, communities, delta)
    communities = list(communities)

    if maps is not None:
        coms = []
        for com in communities:
            coms.append([maps[n] for n in com])
        nx.relabel_nodes(g, maps, False)
    else:
        coms = communities

    return NodeClustering(
        coms,
        g_original,
        "Overlapping SSE",
        method_parameters={
            "seeds": seeds,
            "ninf": ninf,
            "expansion": expansion,
            "stopping": stopping,
            "nworkers": nworkers,
            "nruns": nruns,
            "alpha": alpha,
            "maxexpand": maxexpand,
            "delta": delta,
        },
        overlap=True,
    )


def kclique(g_original: object, k: int) -> NodeClustering:
    """
    Find k-clique communities in graph using the percolation method.
    A k-clique community is the union of all cliques of size k that can be reached through adjacent (sharing k-1 nodes) k-cliques.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param k: Size of smallest clique
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.kclique(G, k=3)

    :References:

    Gergely Palla, Imre Derényi, Illés Farkas1, and Tamás Vicsek, `Uncovering the overlapping community structure of complex networks in nature and society <https://www.nature.com/articles/nature03607/>`_ Nature 435, 814-818, 2005, doi:10.1038/nature03607
    """

    g = convert_graph_formats(g_original, nx.Graph)

    coms = list(nx.algorithms.community.k_clique_communities(g, k))
    coms = [list(x) for x in coms]
    return NodeClustering(
        coms, g_original, "Klique", method_parameters={"k": k}, overlap=True
    )


def lfm(g_original: object, alpha: float) -> NodeClustering:
    """LFM is based on the local optimization of a fitness function.
    It finds both overlapping communities and the hierarchical structure.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param alpha: parameter to controll the size of the communities:  Large values of alpha yield very small communities, small values instead deliver large modules. If alpha is small enough, all nodes end up in the same cluster, the network itself. In most cases, for alpha < 0.5 there is only one community, for alpha > 2 one recovers the smallest communities. A natural choise is alpha =1.
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.lfm(G, alpha=0.8)

    :References:

    Lancichinetti, Andrea, Santo Fortunato, and János Kertész. `Detecting the overlapping and hierarchical community structure in complex networks <https://arxiv.org/abs/0802.1218/>`_ New Journal of Physics 11.3 (2009): 033015.
    """

    g = convert_graph_formats(g_original, nx.Graph)

    algorithm = LFM_nx(g, alpha)
    coms = algorithm.execute()

    return NodeClustering(
        coms, g_original, "LFM", method_parameters={"alpha": alpha}, overlap=True
    )


def lais2(g_original: object) -> NodeClustering:
    """
    LAIS2 is an overlapping community discovery algorithm based on the density function.
    In the algorithm considers the density of a group is defined as the average density of the communication exchanges between the actors of the group.
    LAIS2 IS composed of two procedures LA (Link Aggregate Algorithm) and IS2 (Iterative Scan Algorithm).


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.lais2(G)

    :References:

    Baumes, Jeffrey, Mark Goldberg, and Malik Magdon-Ismail. `Efficient identification of overlapping communities. <https://link.springer.com/chapter/10.1007/11427995_3/>`_ International Conference on Intelligence and Security Informatics. Springer, Berlin, Heidelberg, 2005.

    .. note:: Reference implementation: https://github.com/kritishrivastava/CommunityDetection-Project2GDM

    """

    g = convert_graph_formats(g_original, nx.Graph)

    coms = LAIS2(g)
    return NodeClustering(
        coms, g_original, "LAIS2", method_parameters={"": ""}, overlap=True
    )


def congo(
    g_original: object, number_communities: int, height: int = 2
) -> NodeClustering:
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


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param number_communities: the number of communities desired
    :param height: The lengh of the longest shortest paths that CONGO considers, default 2
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.congo(G, number_communities=3, height=2)

    :References:

    Gregory, Steve. `A fast algorithm to find overlapping communities in networks. <https://link.springer.com/chapter/10.1007/978-3-540-87479-9_45/>`_ Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2008.

    .. note:: Reference implementation: https://github.com/Lab41/Circulo/tree/master/circulo/algorithms

    """

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)

    communities = Congo_(g, number_communities, height)

    coms = []
    for c in communities:
        coms.append([g.vs[x]["name"] for x in c])

    return NodeClustering(
        coms,
        g_original,
        "Congo",
        method_parameters={"number_communities": number_communities, "height": height},
        overlap=True,
    )


def conga(g_original: object, number_communities: int) -> NodeClustering:
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


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param number_communities: the number of communities desired
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> com = algorithms.conga(G, number_communities=3)

    :References:

    Gregory, Steve. `An algorithm to find overlapping community structure in networks. <https://link.springer.com/chapter/10.1007/978-3-540-74976-9_12/>`_ European Conference on Principles of Data Mining and Knowledge Discovery. Springer, Berlin, Heidelberg, 2007.

    .. note:: Reference implementation: https://github.com/Lab41/Circulo/tree/master/circulo/algorithms
    """

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)

    communities = Conga_(g, number_communities=3)
    coms = []
    for c in communities:
        coms.append([g.vs[x]["name"] for x in c])

    return NodeClustering(
        coms,
        g_original,
        "Conga",
        method_parameters={"number_communities": number_communities},
        overlap=True,
    )


def lemon(
    g_original: object,
    seeds: list,
    min_com_size: int = 20,
    max_com_size: int = 50,
    expand_step: int = 6,
    subspace_dim: int = 3,
    walk_steps: int = 3,
    biased: bool = False,
) -> NodeClustering:
    """Lemon is a large scale overlapping community detection method based on local expansion via minimum one norm.

    The algorithm adopts a local expansion method in order to identify the community members from a few exemplary seed members.
    The algorithm finds the community by seeking a sparse vector in the span of the local spectra such that the seeds are in its support. LEMON can achieve the highest detection accuracy among state-of-the-art proposals. The running time depends on the size of the community rather than that of the entire graph.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param seeds: Node list
    :param min_com_size: the minimum size of a single community in the network, default 20
    :param max_com_size: the maximum size of a single community in the network, default 50
    :param expand_step: the step of seed set increasement during expansion process, default 6
    :param subspace_dim: dimension of the subspace; choosing a large dimension is undesirable because it would increase the computation cost of generating local spectra default 3
    :param walk_steps: the number of step for the random walk, default 3
    :param biased: boolean; set if the random walk starting from seed nodes, default False
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> seeds = ["$0$", "$2$", "$3$"]
    >>> coms = algorithms.lemon(G, seeds, min_com_size=2, max_com_size=5)

    :References:

    Yixuan Li, Kun He, David Bindel, John Hopcroft `Uncovering the small community structure in large networks: A local spectral approach. <https://dl.acm.org/citation.cfm?id=2736277.2741676/>`_ Proceedings of the 24th international conference on world wide web. International World Wide Web Conferences Steering Committee, 2015.

    .. note:: Reference implementation: https://github.com/YixuanLi/LEMON
    """

    graph = convert_graph_formats(g_original, nx.Graph)
    graph_m = nx.convert_matrix.to_numpy_array(graph)

    node_to_pos = {n: p for p, n in enumerate(graph.nodes())}
    pos_to_node = {p: n for n, p in node_to_pos.items()}

    seeds = np.array([node_to_pos[s] for s in seeds])

    community = LEMON.lemon(
        graph_m,
        seeds,
        min_com_size,
        max_com_size,
        expand_step,
        subspace_dim=subspace_dim,
        walk_steps=walk_steps,
        biased=biased,
    )

    return NodeClustering(
        [[pos_to_node[n] for n in community]],
        g_original,
        "LEMON",
        method_parameters=dict(
            seeds=str(list(seeds)),
            min_com_size=min_com_size,
            max_com_size=max_com_size,
            expand_step=expand_step,
            subspace_dim=subspace_dim,
            walk_steps=walk_steps,
            biased=biased,
        ),
        overlap=True,
    )


def slpa(g_original: object, t: int = 21, r: float = 0.1) -> NodeClustering:
    """
    SLPA is an overlapping community discovery that extends tha LPA.
    SLPA consists of the following three stages:
    1) the initialization
    2) the evolution
    3) the post-processing


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========


    :param g_original: a networkx/igraph object
    :param t: maximum number of iterations, default 20
    :param r: threshold  ∈ [0, 1]. It is used in the post-processing stage: if the probability of seeing a particular label during the whole process is less than r, this label is deleted from a node’s memory. Default 0.1
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.slpa(G,  t=21, r=0.1)



    :References:

    Xie Jierui, Boleslaw K. Szymanski, and Xiaoming Liu. `Slpa: Uncovering overlapping communities in social networks via a speaker-listener interaction dynamic process. <https://ieeexplore.ieee.org/document/6137400/>`_ Data Mining Workshops (ICDMW), 2011 IEEE 11th International Conference on. IEEE, 2011.

    .. note:: Reference implementation: https://github.com/kbalasu/SLPA
    """

    g = convert_graph_formats(g_original, nx.Graph)

    coms = slpa_nx(g, T=t, r=r)
    return NodeClustering(
        coms, g_original, "SLPA", method_parameters={"T": t, "r": r}, overlap=True
    )


def multicom(g_original: object, seed_node: object) -> NodeClustering:
    """
    MULTICOM is an algorithm for detecting multiple local communities, possibly overlapping, by expanding the initial seed set.
    This algorithm uses local scoring metrics to define an embedding of the graph around the seed set. Based on this embedding, it picks new seeds in the neighborhood of the original seed set, and uses these new seeds to recover multiple communities.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param seed_node: Id of the seed node around which we want to detect communities.
    :return: EdgeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.multicom(G, seed_node=0)

    :References:

    Hollocou, Alexandre, Thomas Bonald, and Marc Lelarge. `Multiple Local Community Detection. <https://hal.archives-ouvertes.fr/hal-01625444/document/>`_ ACM SIGMETRICS Performance Evaluation Review 45.2 (2018): 76-83.

    .. note:: Reference implementation: https://github.com/ahollocou/multicom

    """

    g = convert_graph_formats(g_original, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    mc = MultiCom(g)
    coms = mc.execute(seed_node)

    if maps is not None:
        communities = []
        for c in coms:
            communities.append([maps[n] for n in c])
        nx.relabel_nodes(g, maps, False)
    else:
        communities = [list(c) for c in coms]

    return NodeClustering(
        communities,
        g_original,
        "Multicom",
        method_parameters={"seeds": seed_node},
        overlap=True,
    )


def big_clam(
    g_original: object,
    dimensions: int = 8,
    iterations: int = 50,
    learning_rate: float = 0.005,
) -> NodeClustering:
    """
    BigClam is an overlapping community detection method that scales to large networks.
    The procedure uses gradient ascent to create an embedding which is used for deciding the node-cluster affiliations.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param dimensions: Number of embedding dimensions. Default 8.
    :param iterations: Number of training iterations. Default 50.
    :param learning_rate: Gradient ascent learning rate. Default is 0.005.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.big_clam(G)

    :References:

    Yang, Jaewon, and Jure Leskovec. "Overlapping community detection at scale: a nonnegative matrix factorization approach." Proceedings of the sixth ACM international conference on Web search and data mining. 2013.

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """

    g = convert_graph_formats(g_original, nx.Graph)

    model = BigClam(
        dimensions=dimensions, iterations=iterations, learning_rate=learning_rate
    )
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "BigClam",
        method_parameters={
            "dimensions": dimensions,
            "iterations": iterations,
            "learning_rate": learning_rate,
        },
        overlap=True,
    )


def danmf(
    g_original: object,
    layers: tuple = (32, 8),
    pre_iterations: int = 100,
    iterations: int = 100,
    seed: int = 42,
    lamb: float = 0.01,
) -> NodeClustering:
    """
    The procedure uses telescopic non-negative matrix factorization in order to learn a cluster memmbership distribution over nodes. The method can be used in an overlapping and non-overlapping way.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param layers: Autoencoder layer sizes in a list of integers. Default [32, 8].
    :param pre_iterations: Number of pre-training epochs. Default 100.
    :param iterations: Number of training epochs. Default 100.
    :param seed: Random seed for weight initializations. Default 42.
    :param lamb: Regularization parameter. Default 0.01.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.danmf(G)

    :References:

    Ye, Fanghua, Chuan Chen, and Zibin Zheng. "Deep autoencoder-like nonnegative matrix factorization for community detection." Proceedings of the 27th ACM International Conference on Information and Knowledge Management. 2018.

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """
    g = convert_graph_formats(g_original, nx.Graph)
    model = DANMF(layers, pre_iterations, iterations, seed, lamb)

    mapping = {node: i for i, node in enumerate(g.nodes())}
    rev = {i: node for node, i in mapping.items()}
    H = nx.relabel_nodes(g, mapping)

    model.fit(H)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(rev[n])

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "DANMF",
        method_parameters={
            "layers": layers,
            "pre_iteration": pre_iterations,
            "iterations": iterations,
            "seed": seed,
            "lamb": lamb,
        },
        overlap=True,
    )


def egonet_splitter(g_original: object, resolution: float = 1.0) -> NodeClustering:
    """
    The method first creates the egonets of nodes. A persona-graph is created which is clustered by the Louvain method.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param resolution: Resolution parameter of Python Louvain. Default 1.0.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.egonet_splitter(G)

    :References:

    Epasto, Alessandro, Silvio Lattanzi, and Renato Paes Leme. "Ego-splitting framework: From non-overlapping to overlapping clusters." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2017.

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """
    g = convert_graph_formats(g_original, nx.Graph)
    model = EgoNetSplitter(resolution=resolution)

    mapping = {node: i for i, node in enumerate(g.nodes())}
    rev = {i: node for node, i in mapping.items()}
    H = nx.relabel_nodes(g, mapping)

    model.fit(H)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, cs in members.items():
        for c in cs:
            coms_to_node[c].append(rev[n])

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "EgoNetSplitter",
        method_parameters={"resolution": resolution},
        overlap=True,
    )


def nnsed(
    g_original: object, dimensions: int = 32, iterations: int = 10, seed: int = 42
) -> NodeClustering:
    """
    The procedure uses non-negative matrix factorization in order to learn an unnormalized cluster membership distribution over nodes. The method can be used in an overlapping and non-overlapping way.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param dimensions: Embedding layer size. Default is 32.
    :param iterations: Number of training epochs. Default 10.
    :param seed:  Random seed for weight initializations. Default 42.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.nnsed(G)

    :References:

    Sun, Bing-Jie, et al. "A non-negative symmetric encoder-decoder approach for community detection." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 2017.

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """
    g = convert_graph_formats(g_original, nx.Graph)
    model = NNSED(dimensions=dimensions, iterations=iterations, seed=seed)
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "NNSED",
        method_parameters={
            "dimension": dimensions,
            "iterations": iterations,
            "seed": seed,
        },
        overlap=True,
    )


def mnmf(
    g_original: object,
    dimensions: int = 128,
    clusters: int = 10,
    lambd: float = 0.2,
    alpha: float = 0.05,
    beta: float = 0.05,
    iterations: int = 200,
    lower_control: float = 1e-15,
    eta: float = 5.0,
) -> NodeClustering:
    """
    The procedure uses joint non-negative matrix factorization with modularity based regul;arization in order to learn a cluster memmbership distribution over nodes.
    The method can be used in an overlapping and non-overlapping way.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param dimensions: Number of dimensions. Default is 128.
    :param clusters: Number of clusters. Default is 10.
    :param lambd: KKT penalty. Default is 0.2
    :param alpha: Clustering penalty. Default is 0.05.
    :param beta: Modularity regularization penalty. Default is 0.05.
    :param iterations:  Number of power iterations. Default is 200.
    :param lower_control: Floating point overflow control. Default is 10**-15.
    :param eta: Similarity mixing parameter. Default is 5.0.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.mnmf(G)

    :References:

    Wang, Xiao, et al. "Community preserving network embedding." Thirty-first AAAI conference on artificial intelligence. 2017.

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """
    g = convert_graph_formats(g_original, nx.Graph)
    model = MNMF(
        dimensions=dimensions,
        clusters=clusters,
        lambd=lambd,
        alpha=alpha,
        beta=beta,
        iterations=iterations,
        lower_control=lower_control,
        eta=eta,
    )
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "MNMF",
        method_parameters={
            "dimension": dimensions,
            "clusters": clusters,
            "lambd": lambd,
            "alpha": alpha,
            "beta": beta,
            "iterations": iterations,
            "lower_control": lower_control,
            "eta": eta,
        },
        overlap=True,
    )


def aslpaw(g_original: object) -> NodeClustering:
    """
    ASLPAw can be used for disjoint and overlapping community detection and works on weighted/unweighted and directed/undirected networks.
    ASLPAw is adaptive with virtually no configuration parameters.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.aslpaw(G)

    :References:

    Xie J, Szymanski B K, Liu X. Slpa: Uncovering Overlapping Communities in Social Networks via a Speaker-Listener Interaction Dynamic Process[C]. IEEE 11th International Conference on Data Mining Workshops (ICDMW). Ancouver, BC: IEEE, 2011: 344–349.

    .. note:: Reference implementation: https://github.com/fsssosei/ASLPAw
    """

    if ASLPAw is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install gmpy (conda install gmpy2) and ASLPAw (pip install shuffle_graph>=2.1.0 similarity-index-of-label-graph>=2.0.1 ASLPAw>=2.1.0). If using a notebook, you need also to restart your runtime/kernel."
        )

    g = convert_graph_formats(g_original, nx.Graph)
    coms = ASLPAw(g).adj

    communities = defaultdict(list)
    for i, c in coms.items():
        if len(c) > 0:
            for cid in c:
                communities[cid].append(i)

    return NodeClustering(
        list(communities.values()),
        g_original,
        "ASLPAw",
        method_parameters={},
        overlap=True,
    )


def percomvc(g_original: object) -> NodeClustering:
    """
    The PercoMVC approach composes of two steps.
    In the first step, the algorithm attempts to determine all communities that the clique percolation algorithm may find.
    In the second step, the algorithm computes the Eigenvector Centrality method on the output of the first step to measure the influence of network nodes and reduce the rate of the unclassified nodes


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.percomvc(G)

    :References:

    Kasoro, Nathanaël, et al. "PercoMCV: A hybrid approach of community detection in social networks." Procedia Computer Science 151 (2019): 45-52.

    .. note:: Reference implementation: https://github.com/sedjokas/PercoMCV-Code-source
    """
    g = convert_graph_formats(g_original, nx.Graph)
    communities = percoMVC(g)

    return NodeClustering(
        communities, g_original, "PercoMVC", method_parameters={}, overlap=True
    )


def wCommunity(
    g_original: object,
    min_bel_degree: float = 0.7,
    threshold_bel_degree: float = 0.7,
    weightName: str = "weight",
) -> NodeClustering:
    """
    Algorithm to identify overlapping communities in weighted graphs


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param min_bel_degree: the tolerance, in terms of beloging degree, required in order to add a node in a community
    :param threshold_bel_degree: the tolerance, in terms of beloging degree, required in order to add a node in a 'NLU' community
    :param weightName: name of the edge attribute containing the weights
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> nx.set_edge_attributes(G, values=1, name='weight')
    >>> coms = algorithms.wCommunity(G, min_bel_degree=0.6, threshold_bel_degree=0.6)

    :References:

    Chen, D., Shang, M., Lv, Z., & Fu, Y. (2010). Detecting overlapping communities of weighted networks via a local algorithm. Physica A: Statistical Mechanics and its Applications, 389(19), 4177-4187.

    .. note:: Implementation provided by Marco Cardia <cardiamc@gmail.com> and Francesco Sabiu <fsabiu@gmail.com> (Computer Science Dept., University of Pisa, Italy)
    """

    if ig is None:
        raise ModuleNotFoundError(
            "Optional dependency not satisfied: install igraph to use the selected feature."
        )

    g = convert_graph_formats(g_original, ig.Graph)
    # Initialization
    comm = weightedCommunity(
        g,
        min_bel_degree=min_bel_degree,
        threshold_bel_degree=threshold_bel_degree,
        weightName=weightName,
    )
    # Community computation
    comm.computeCommunities()
    # Result
    coms = comm.getCommunities()
    coms = [list(c) for c in coms]

    # renaming and deduplicate communities
    coms_res = set()
    for c in coms:
        coms_res.add(frozenset([g.vs[x]["name"] for x in c]))

    coms_res = [list(c) for c in coms_res]

    return NodeClustering(
        coms_res,
        g_original,
        "wCommunity",
        method_parameters={
            "min_bel_degree": min_bel_degree,
            "threshold_bel_degree": threshold_bel_degree,
            "weightName": weightName,
        },
        overlap=True,
    )


def core_expansion(g_original: object, tolerance: float = 0.0001) -> NodeClustering:
    """
    Core Expansion automatically detect the core of each possible community in the network. Then, it iteratively expand each core by adding the nodes to form the fnal communities. The expansion process is based on the neighborhood overlap measure.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param tolerance: numerical tollerance, default 0.0001
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.core_expansion(G)

    :References:

    Choumane, Ali, Ali Awada, and Ali Harkous. "Core expansion: a new community detection algorithm based on neighborhood overlap." Social Network Analysis and Mining 10 (2020): 1-11.

    .. note:: Reference implementation: https://github.com/pkalkunte18/CoreExpansionAlgorithm
    """
    g = convert_graph_formats(g_original, nx.Graph)
    communities = core_exp_find(g, tolerance)

    return NodeClustering(
        communities,
        g_original,
        "Core Expansion",
        method_parameters={"tolerance": tolerance},
        overlap=True,
    )


def lpanni(g_original: object, threshold: float = 0.1) -> NodeClustering:
    """

    LPANNI (Label Propagation Algorithm with Neighbor Node Influence) detects overlapping community structures by adopting fixed label propagation sequence based on the ascending order of node importance and label update strategy based on neighbor node influence and historical label preferred strategy.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param threshold: Default 0.0001

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.lpanni(G)

    :References:

    Lu, Meilian, et al. "LPANNI: Overlapping community detection using label propagation in large-scale complex networks." IEEE Transactions on Knowledge and Data Engineering 31.9 (2018): 1736-1749.

    .. note:: Reference implementation: https://github.com/wxwmd/LPANNI
    """
    g = convert_graph_formats(g_original, nx.Graph)
    LPANNI(g)
    gen = GraphGenerator(threshold, g)
    communities = [list(c) for c in gen.get_Overlapping_communities()]

    return NodeClustering(
        communities,
        g_original,
        "LPANNI",
        method_parameters={"threshold": threshold},
        overlap=True,
    )


def lpam(
    g_original: object,
    k: int = 2,
    threshold: float = 0.5,
    distance: str = "amp",
    seed: int = 0,
) -> NodeClustering:
    """
    Link Partitioning Around Medoids


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param k: number of clusters
    :param threshold: merging threshold in [0,1], default 0.5
    :param distance: type of distance: "amp" - amplified commute distance, or "cm" - commute distance, or distance matrix between all edges as np ndarray
    :param seed: random seed for k-medoid heuristic
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.lpam(G, k=2, threshold=0.4, distance = "amp")

    :References:

    Alexander Ponomarenko, Leonidas Pitsoulis, Marat Shamshetdinov. "Link Partitioning Around Medoids". https://arxiv.org/abs/1907.08731

    """
    g = convert_graph_formats(g_original, nx.Graph)
    return LPAM(graph=g, k=k, threshold=threshold, distance=distance, seed=seed)


def dcs(g_original: object) -> NodeClustering:
    """
    Divide and Conquer Strategy


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.dcs(G)

    :References:

     Syed Agha Muhammad and Kristof Van Laerhoven. "DCS: Divide and Conquer Strategy For Detecting Overlapping Communities in Social Graphs". https://bit.ly/33m7t3r

    .. note:: Reference implementation: https://github.com/SyedAgha/Divide-and-Conquer/tree/master/DCS_code_and_paper

    """
    g = convert_graph_formats(g_original, nx.Graph)
    communities = main_dcs(g)
    return NodeClustering(
        communities, g_original, "DCS", method_parameters={}, overlap=True
    )


def umstmo(g_original: object) -> NodeClustering:
    """
    Overlapping community detection based on the union of all maximum spanning trees


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.umstmo(G)

    :References:

     Asmi, Khawla, Dounia Lotfi, and Mohamed El Marraki. "Overlapping community detection based on the union of all maximum spanning trees." Library Hi Tech (2020).

    .. note:: Reference implementation: https://github.com/khawka/UMSTMO

    """
    g = convert_graph_formats(g_original, nx.Graph)
    communities = UMSTMO(g)
    return NodeClustering(
        communities, g_original, "UMSTMO", method_parameters={}, overlap=True
    )


def symmnmf(
    g_original: object,
    dimensions: int = 32,
    iterations: int = 200,
    rho: float = 100.0,
    seed: int = 42,
) -> NodeClustering:
    """
    The procedure decomposed the second power od the normalized adjacency matrix with an ADMM based non-negative matrix factorization based technique.
    This results in a node embedding and each node is associated with an embedding factor in the created latent space.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param dimensions: Number of dimensions. Default is 32.
    :param iterations:  Number of power iterations. Default is 200.
    :param rho: Regularization tuning parameter. Default is 100.0.
    :param seed: Random seed value. Default is 42.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.symmnmf(G)

    :References:

    Kuang, Da, Chris Ding, and Haesun Park. "Symmetric nonnegative matrix factorization for graph clustering." Proceedings of the 2012 SIAM international conference on data mining. Society for Industrial and Applied Mathematics, 2012.

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """
    g = convert_graph_formats(g_original, nx.Graph)
    model = SymmNMF(dimensions=dimensions, iterations=iterations, rho=rho, seed=seed)
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(
        coms,
        g_original,
        "SymmNMF",
        method_parameters={
            "dimension": dimensions,
            "iterations": iterations,
            "rho": rho,
            "seed": seed,
        },
        overlap=True,
    )


def walkscan(
    g_original: object,
    nb_steps: int = 2,
    eps: float = 0.1,
    min_samples: int = 3,
    init_vector: dict = None,
) -> NodeClustering:
    """
    Random walk community detection method leveraging PageRank node scoring.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param nb_steps: the length of the random walk
    :param eps: DBSCAN eps
    :param min_samples: DBSCAN min_samples
    :param init_vector: dictionary node_id -> initial_probability to initialize the random walk. Default, random selected node with probability set to 1.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.walkscan(G)

    :References:

    Hollocou, A., Bonald, T., & Lelarge, M. (2016). Improving PageRank for local community detection. arXiv preprint arXiv:1610.08722.

    .. note:: Reference implementation: https://github.com/ahollocou/walkscan
    """
    g = convert_graph_formats(g_original, nx.Graph)
    ws = WalkSCAN(nb_steps=nb_steps, eps=eps, min_samples=min_samples)

    # Initialization vector for the random walk
    if init_vector is None:
        n = sample(list(g.nodes()), 1)[0]
        init_vector = {n: 1}

    ws.detect_communities(g, init_vector)
    coms = [list(c) for c in ws.communities_]

    return NodeClustering(
        coms,
        g_original,
        "walkscan",
        method_parameters={
            "nb_steps": nb_steps,
            "eps": eps,
            "min_samples": min_samples,
            "init_vector": init_vector,
        },
        overlap=True,
    )


def endntm(
    g_original: object, clusterings: list = None, epsilon: float = 2
) -> NodeClustering:
    """
    Overlapping community detection algorithm based on an ensemble  approach with a distributed neighbourhood threshold method (EnDNTM).
    EnDNTM uses pre-partitioned disjoint communities generated by the ensemble mechanism and then analyzes the neighbourhood distribution  of boundary nodes in disjoint communities to detect overlapping communities.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param clusterings: an iterable of Clustering objects (non overlapping node partitions only)
    :param epsilon: neighbourhood threshold, default 2.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms_l = [algorithms.louvain(G), algorithms.label_propagation(G), algorithms.walktrap(G)]
    >>> coms = algorithms.endntm(G, coms_l)

    :References:

    Jaiswal, R., & Ramanna, S. Detecting overlapping communities using ensemble-based distributed neighbourhood threshold method in social networks. Intelligent Decision Technologies, (2021), doi:10.3233/IDT-200059.

    """
    g = convert_graph_formats(g_original, nx.Graph)
    algClstr = {1: None, 2: []}
    shrDict = {1: 0.0, 2: algClstr}

    if clusterings is None:
        raise ValueError("No precomputed node clusterings provided.")

    for clustering in clusterings:
        if clustering.overlap:
            raise ValueError(
                f"endntm requires non overlapping node clusterings: {clustering.method_name} is overlapping."
            )
        c_name = clustering.method_name
        communities = [set(c) for c in clustering.communities]
        val = endntm_evalFuction(g, communities)

        if val > shrDict[1]:
            shrDict[1] = val
            shrDict[2][1] = c_name
            shrDict[2][2] = communities

    clusters_list = shrDict[2][2]
    overlap_cluster_list = endntm_find_overlap_cluster(g, clusters_list, epsilon)

    coms = [list(c) for c in overlap_cluster_list]

    return NodeClustering(
        coms,
        g_original,
        "endntm",
        method_parameters={
            "clusterings": [clustering.method_name for clustering in clusterings],
            "epsilon": epsilon,
        },
        overlap=True,
    )


def ipca(
    g_original: object,
    weights: str = None,
    t_in: float = 0.5,
) -> NodeClustering:
    """
    IPCA was introduced by Li et al. (2008) as a modiﬁed version of DPClus.
    In contrast to DPClus, this method focuses on the maintaining the diameter of a cluster, deﬁned as the maximum shortest distance between all pairs of vertices, rather than its density.
    In doing so, the seed growth aspect of IPCA emphasizes structural closeness of a predicted protein complex, as well as structural connectivity.

    Like DPClus, IPCA computes local vertex and edge weights by counting the number of common neighbors shared between two vertices.
    However, IPCA calculates these values only once at the beginning of the algorithm, rather than updating them every time a discovered cluster is removed from the graph.
    This allows overlap to occur naturally between clusters, as cluster nodes are not permanently removed from the graph; however, it can also lead to a large amount of cluster overlap.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param weights: label used for the edge weights. Default, None.
    :param t_in:
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.ipca(G)

    :References:

    Li, M., Chen, J., Wang, J., Hu, B., Chen, G. 2008. Modifying the DPClus algorithm for identifying protein complexes based on new topological structures. BMC Bioinformatics 9, 398.


    .. note:: Reference Implementation: https://github.com/trueprice/python-graph-clustering
    """

    g = convert_graph_formats(g_original, nx.Graph)
    clustering = i_pca(
        g,
        weights=weights,
        t_in=t_in,
    )

    return NodeClustering(
        clustering,
        g_original,
        "ipca",
        method_parameters={
            "t_in": t_in,
        },
        overlap=True,
    )


def dpclus(
    g_original: object,
    weights: str = None,
    d_threshold: float = 0.9,
    cp_threshold: float = 0.5,
    overlap: bool = True,
) -> NodeClustering:
    """
    DPClus projects weights onto an unweighted graph using a common neighbors approach.
    In DPClus, the weight of an edge (u, v) is deﬁned as the number of common neighbors between u and v.
    Similarly, the weight of a vertex is its weighted degree – the sum of all edges connected to the vertex-

    DPClus does not natively generate overlapping clusters but does allow for overlapping cluster nodes to be added in a post-processing step.

    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param weights: label used for the edge weights. Default, None.
    :param d_threshold: cluster density threshold, default 0.9
    :param cp_threshold: cluster property threshold, default 0.5
    :param overlap: wheter to output overlapping or crisp communities. Default, True.
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.dpclus(G)

    :References:

    Altaf-Ul-Amin, M., Shinbo, Y., Mihara, K., Kurokawa, K., Kanaya, S. 2006. Development and implementation of an algorithm for detection of protein complexes in large interaction networks. BMC Bioinformatics 7, 207.

    .. note:: Reference Implementation: https://github.com/trueprice/python-graph-clustering
    """

    g = convert_graph_formats(g_original, nx.Graph)
    clustering = dp_clus(
        g,
        weights=weights,
        overlap=overlap,
        d_threshold=d_threshold,
        cp_threshold=cp_threshold,
    )

    return NodeClustering(
        clustering,
        g_original,
        "dpclus",
        method_parameters={"d_threshold": d_threshold, "cp_threshold": cp_threshold},
        overlap=overlap,
    )


def coach(
    g_original: object,
    density_threshold: float = 0.7,
    affinity_threshold: float = 0.225,
    closeness_threshold: float = 0.5,
) -> NodeClustering:
    """
    The motivation behind the core-attachment (CoAch) algorithm  comes from the observation that protein complexes often have a dense core of highly interactive proteins.
    CoAch works in two steps, ﬁrst discovering highly connected regions (“preliminary cores”) of a network and then expanding these regions by adding strongly associated neighbors.

    The algorithm operates with three user-speciﬁed parameters: minimum core density (for preliminary cores), maximum core affinity (similarity threshold for distinct preliminary cores), and minimum neighbor closeness (for attaching non-core neighbors to preliminary cores).


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param density_threshold: minimum core density. Default, 0.7
    :param affinity_threshold: maximum core affinity. Default, 0.225
    :param closeness_threshold:  minimum neighbor closeness. Default, 0.5
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.coach(G)

    :References:

    Wu, M., Li, X., Kwoh, C.-K., Ng, S.-K. A core-attachment based method to detect protein complexes. 2009. In PPI networks. BMC Bioinformatics 10, 169.

    .. note:: Reference Implementation: https://github.com/trueprice/python-graph-clustering
    """

    g = convert_graph_formats(g_original, nx.Graph)
    clustering = co_ach(
        g,
        density_threshold=density_threshold,
        affinity_threshold=affinity_threshold,
        closeness_threshold=closeness_threshold,
    )

    return NodeClustering(
        clustering,
        g_original,
        "coach",
        method_parameters={
            "density_threshold": density_threshold,
            "affinity_threshold": affinity_threshold,
            "closeness_threshold": closeness_threshold,
        },
        overlap=True,
    )


def graph_entropy(
    g_original: object,
    weights: str = None,
) -> NodeClustering:
    """
    This method takes advantage of the use of entropy with regard to information theory.
    Entropy is a measure of uncertainty involved in a random variable.

    This approach uses a new deﬁnition, Graph Entropy, as a measure of structural complexity in a graph.
    This algorithm incorporates a seed-growth technique.
    Unlike the other seed-growth style methods, however, the graph entropy approach does not require any predetermined threshold because it searches for an optimal solution by minimizing graph entropy.

    This method ﬁnds locally optimal clusters with minimal graph entropy.
    A seed vertex is selected at random from a candidate set of seed vertices.
    Then, an initial cluster which is composed of the seed vertex and its immediate neighbors is created.
    Next, the neighbors are iteratively evaluated for removal to minimize the initial entropy of the graph.
    Finally, outer boundary vertices are added recursively if their addition causes the entropy of the graph to decrease.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param weights: label used for the edge weights.. Default, None
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.graph_entropy(G)

    :References:

    Kenley, E.C., Cho, Y.-R. 2011. Detecting protein complexes and functional modules from protein interaction networks: A graph entropy approach. Proteomics 11, 3835-3844.

    .. note:: Reference Implementation: https://github.com/trueprice/python-graph-clustering
    """

    g = convert_graph_formats(g_original, nx.Graph)
    clustering = graphentropy(g, weight=weights)

    return NodeClustering(
        clustering,
        g_original,
        "graph_entropy",
        method_parameters={},
        overlap=True,
    )


def ebgc(
    g_original: object,
) -> NodeClustering:
    """
    The entropy-based clustering approach finds locally optimal clusters by growing a random seed in a manner that minimizes graph entropy.

    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        No       No
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.ebgc(G)

    :References:

    Kenley, Edward Casey, and Young-Rae Cho. "Entropy-based graph clustering: Application to biological and social networks." 2011 IEEE 11th International Conference on Data Mining. IEEE, 2011.

    .. note:: Reference Implementation: https://github.com/SubaiDeng/EBGC-Entropy-Based-Graph-Clustering
    """

    g = convert_graph_formats(g_original, nx.Graph)

    dmap = {n: i for i, n in enumerate(g.nodes)}
    reverse_map = {i: n for n, i in dmap.items()}
    nx.relabel_nodes(g, dmap, False)

    EBGC_cluster = EBGC()
    cluster_result = EBGC_cluster.fit(g)
    _, node_labels = np.nonzero(cluster_result)

    clustering = defaultdict(list)
    for idn, v in enumerate(node_labels):
        clustering[v].append(reverse_map[idn])

    clustering = [c for c in clustering.values()]

    return NodeClustering(
        clustering,
        g_original,
        "ebgc",
        method_parameters={},
        overlap=True,
    )
