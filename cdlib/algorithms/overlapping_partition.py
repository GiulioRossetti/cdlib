from demon import Demon
from angel import Angel
from cdlib.algorithms.internal.NodePerception import NodePerception
from cdlib.algorithms.internal import OSSE
import networkx as nx
import igraph as ig
import numpy as np
from cdlib import NodeClustering
from cdlib.utils import suppress_stdout, convert_graph_formats, nx_node_integer_mapping
from cdlib.algorithms.internal.CONGO import Congo_
from cdlib.algorithms.internal.CONGA import Conga_
from cdlib.algorithms.internal.LAIS2_nx import LAIS2
from cdlib.algorithms.internal.lfm import LFM_nx
from cdlib.algorithms.internal import LEMON
from cdlib.algorithms.internal.SLPA_nx import slpa_nx
from cdlib.algorithms.internal.multicom import MultiCom
from cdlib.algorithms.internal import BIGCLAM


__all__ = ["ego_networks", "demon", "angel", "node_perception", "overlapping_seed_set_expansion", "kclique", "lfm",
           "lais2", "congo", "conga", "lemon", "slpa", "multicom", "big_clam"]


def ego_networks(g, level=1):
    """
    Ego-networks returns overlapping communities centered at each nodes within a given radius.

    :param g: a networkx/igraph object
    :param level: extrac communities with all neighbors of distance<=level from a node. Deafault 1
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.ego_networks(G)
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = []
    for n in g.nodes():
        coms.append(list(nx.ego_graph(g, n, radius=level).nodes()))
    return NodeClustering(coms, g, "Ego Network", method_parameters={"level": 1}, overlap=True)


def demon(g, epsilon, min_com_size=3):
    """
    Demon is a node-centric bottom-up overlapping community discovery algorithm.
    It leverages ego-network structures and overlapping label propagation to identify micro-scale communities that are subsequently merged in mesoscale ones.

    :param g: a networkx/igraph object
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

    g = convert_graph_formats(g, nx.Graph)

    with suppress_stdout():
        dm = Demon(graph=g, epsilon=epsilon, min_community_size=min_com_size)
        coms = dm.execute()
        coms = [list(c) for c in coms]

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
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.angel(G, min_com_size=3, threshold=0.25)

    :References:

    1. Rossetti G. **Angel: efficient, and effective, node-centric community discovery in static and dynamic networks.**

    .. note:: Reference implementation: https://github.com/GiulioRossetti/ANGEL
    """

    g = convert_graph_formats(g, ig.Graph)
    with suppress_stdout():
        a = Angel(graph=g, min_comsize=min_community_size, threshold=threshold, save=False)
        coms = a.execute()

    return NodeClustering(list(coms.values()), g, "ANGEL", method_parameters={"threshold": threshold,
                                                                              "min_community_size": min_community_size},
                          overlap=True)


def node_perception(g, threshold, overlap_threshold, min_comm_size=3):
    """Node perception is based on the idea of joining together small sets of nodes.
    The algorithm first identifies sub-communities corresponding to each node’s perception of the network around it.
    To perform this step, it considers each node individually, and partition that node’s neighbors into communities using some existing community detection method.
    Next, it creates a new network in which every node corresponds to a sub-community, and two nodes are linked if their associated sub-communities overlap by at least some threshold amount.
    Finally, the algorithm identifies overlapping communities in this new network, and for every such community, merge together the associated sub-communities to identify communities in the original network.

    :param g: a networkx/igraph object
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
    g = convert_graph_formats(g, nx.Graph)
    tp = type(list(g.nodes())[0])

    with suppress_stdout():
        np = NodePerception(g, sim_threshold=threshold, overlap_threshold=overlap_threshold,
                            min_comm_size=min_comm_size)
        coms = np.execute()
        if tp != str:
            communities = []
            for c in coms:
                c = list(map(tp, c))
                communities.append(c)
            coms = communities

    return NodeClustering(coms, g, "Node Perception", method_parameters={"threshold": threshold,
                                                                         "overlap_threshold": overlap_threshold,
                                                                         "min_com_size": min_comm_size},
                          overlap=True)


def overlapping_seed_set_expansion(g, seeds, ninf=False, expansion='ppr', stopping='cond', nworkers=1,
                                   nruns=13, alpha=0.99, maxexpand=float('INF'), delta=0.2):
    """
    OSSE is an overlapping community detection algorithm optimizing the conductance community score
    The algorithm uses a seed set expansion approach; the key idea is to find good seeds, and then expand these seed sets using the personalized PageRank clustering procedure.

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

    g = convert_graph_formats(g, nx.Graph)

    g, maps = nx_node_integer_mapping(g)
    if maps is not None:
        rev_map = {v: k for k, v in maps.items()}
        seeds = [rev_map[s] for s in seeds]

    if ninf:
        seeds = OSSE.neighbor_inflation(g, seeds)

    communities = OSSE.growclusters(g, seeds, expansion, stopping, nworkers, nruns, alpha, maxexpand, False)
    communities = OSSE.remove_duplicates(g, communities, delta)
    communities = list(communities)

    if maps is not None:
        coms = []
        for com in communities:
            coms.append([maps[n] for n in com])
        nx.relabel_nodes(g, maps, False)
    else:
        coms = communities

    return NodeClustering(coms, g, "Overlapping SSE", method_parameters={"seeds": seeds, "ninf": ninf,
                                                                         "expansion": expansion,
                                                                         "stopping": stopping,
                                                                         "nworkers": nworkers,
                                                                         "nruns": nruns, "alpha": alpha,
                                                                         "maxexpand": maxexpand,
                                                                         "delta": delta},
                          overlap=True)


def kclique(g, k):
    """
    Find k-clique communities in graph using the percolation method.
    A k-clique community is the union of all cliques of size k that can be reached through adjacent (sharing k-1 nodes) k-cliques.

    :param g: a networkx/igraph object
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

    g = convert_graph_formats(g, nx.Graph)

    coms = list(nx.algorithms.community.k_clique_communities(g, k))
    coms = [list(x) for x in coms]
    return NodeClustering(coms, g, "Klique", method_parameters={"k": k}, overlap=True)


def lfm(g, alpha):
    """LFM is based on the local optimization of a fitness function.
    It finds both overlapping communities and the hierarchical structure.

    :param g: a networkx/igraph object
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

    g = convert_graph_formats(g, nx.Graph)

    algorithm = LFM_nx(g, alpha)
    coms = algorithm.execute()

    return NodeClustering(coms, g, "LFM", method_parameters={"alpha": alpha}, overlap=True)


def lais2(g):
    """
    LAIS2 is an overlapping community discovery algorithm based on the density function.
    In the algorithm considers the density of a group is defined as the average density of the communication exchanges between the actors of the group.
    LAIS2 IS composed of two procedures LA (Link Aggregate Algorithm) and IS2 (Iterative Scan Algorithm).

    :param g: a networkx/igraph object
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

    g = convert_graph_formats(g, nx.Graph)

    coms = LAIS2(g)
    return NodeClustering(coms, g, "LAIS2", overlap=True)


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

    g = convert_graph_formats(g, ig.Graph)

    communities = Congo_(g, number_communities, height)

    coms = []
    for c in communities:
        coms.append([g.vs[x]['name'] for x in c])

    return NodeClustering(coms, g, "Congo", method_parameters={"number_communities": number_communities,
                                                               "height": height}, overlap=True)


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

    g = convert_graph_formats(g, ig.Graph)

    communities = Conga_(g, number_communities=3)
    coms = []
    for c in communities:
        coms.append([g.vs[x]['name'] for x in c])

    return NodeClustering(coms, g, "Conga", method_parameters={"number_communities": number_communities}, overlap=True)


def lemon(graph, seeds, min_com_size=20, max_com_size=50, expand_step=6, subspace_dim=3, walk_steps=3, biased=False):
    """Lemon is a large scale overlapping community detection method based on local expansion via minimum one norm.

    The algorithm adopts a local expansion method in order to identify the community members from a few exemplary seed members.
    The algorithm finds the community by seeking a sparse vector in the span of the local spectra such that the seeds are in its support. LEMON can achieve the highest detection accuracy among state-of-the-art proposals. The running time depends on the size of the community rather than that of the entire graph.

    :param graph: a networkx/igraph object
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

    graph = convert_graph_formats(graph, nx.Graph)
    graph_m = nx.convert_matrix.to_numpy_array(graph)

    node_to_pos = {n: p for p, n in enumerate(graph.nodes())}
    pos_to_node = {p: n for n, p in node_to_pos.items()}

    seeds = np.array([node_to_pos[s] for s in seeds])

    community = LEMON.lemon(graph_m, seeds, min_com_size, max_com_size, expand_step,
                            subspace_dim=subspace_dim, walk_steps=walk_steps, biased=biased)

    return NodeClustering([[pos_to_node[n] for n in community]], graph,
                          "LEMON", method_parameters=dict(seeds=str(list(seeds)), min_com_size=min_com_size,
                                                          max_com_size=max_com_size, expand_step=expand_step,
                                                          subspace_dim=subspace_dim, walk_steps=walk_steps,
                                                          biased=biased), overlap=True)


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
    :return: EdgeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.slpa(G,  t=21, r=0.1)



    :References:

    Xie Jierui, Boleslaw K. Szymanski, and Xiaoming Liu. `Slpa: Uncovering overlapping communities in social networks via a speaker-listener interaction dynamic process. <https://ieeexplore.ieee.org/document/6137400/>`_ Data Mining Workshops (ICDMW), 2011 IEEE 11th International Conference on. IEEE, 2011.

    .. note:: Reference implementation: https://github.com/kbalasu/SLPA
    """

    g = convert_graph_formats(g, nx.Graph)

    coms = slpa_nx(g, T=t, r=r)
    return NodeClustering(coms, g, "SLPA", method_parameters={"T": t, "r": r}, overlap=True)


def multicom(g, seed_node):
    """
    MULTICOM is an algorithm for detecting multiple local communities, possibly overlapping, by expanding the initial seed set.
    This algorithm uses local scoring metrics to define an embedding of the graph around the seed set. Based on this embedding, it picks new seeds in the neighborhood of the original seed set, and uses these new seeds to recover multiple communities.

    :param g: a networkx/igraph object
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

    g = convert_graph_formats(g, nx.Graph)
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

    return NodeClustering(communities, g, "Multicom", method_parameters={"seeds": seed_node}, overlap=True)


def big_clam(g, number_communities=5):
    """
    BigClam is an overlapping community detection method that scales to large networks.
    The model has three main ingredients:
    1)The node community memberships are represented with a bipartite affiliation network that links nodes of the social network to communities that they belong to.
    2)People tend to be involved in communities to various degrees. Therefore,  each affiliation edge in the bipartite affiliation network has a nonnegative weight. The higher the node’s weight of the affiliation to the community the more likely is the node to be connected to other members in the community.
    3)When people share multiple community affiliations, the links between them stem for one dominant reason. This means that for each community a pair of nodes shares we get an independent chance of connecting the nodes. Thus, naturally, the more communities a pair of nodes shares, the higher the probability of being connected.

    :param g: a networkx/igraph object
    :param number_communities: number communities desired, default 5
    :return: NodeClustering object


    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.big_clam(G, 2)

    :References:

    Yang, J., & Leskovec, J. (2013, February). `Overlapping community detection at scale: a nonnegative matrix factorization approach. <https://dl.acm.org/citation.cfm?id=2433471/>`_ In Proceedings of the sixth ACM international conference on Web search and data mining (pp. 587-596). ACM.

    .. note:: Reference implementation: https://github.com/RobRomijnders/bigclam
    """

    g = convert_graph_formats(g, nx.Graph)

    communities = BIGCLAM.big_Clam(g, number_communities)

    return NodeClustering(communities, g, "BigClam", method_parameters={"number_communities": number_communities}, overlap=True)
