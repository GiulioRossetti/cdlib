try:
    import igraph as ig
except ModuleNotFoundError:
        ig = None
try:
    from angel import Angel
except ModuleNotFoundError:
    Angel = None
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
from karateclub import DANMF, EgoNetSplitter, NNSED, MNMF, BigClam


__all__ = ["ego_networks", "demon", "angel", "node_perception", "overlapping_seed_set_expansion", "kclique", "lfm",
           "lais2", "congo", "conga", "lemon", "slpa", "multicom", "big_clam", "danmf", "egonet_splitter", "nnsed",
           "nmnf"]


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

    1. Rossetti, Giulio. "Exorcising the Demon: Angel, Efficient Node-Centric Community Discovery." International Conference on Complex Networks and Their Applications. Springer, Cham, 2019.

    .. note:: Reference implementation: https://github.com/GiulioRossetti/ANGEL
    """
    if ig is None:
        raise ModuleNotFoundError("Optional dependency not satisfied: install igraph to use the selected feature.")
    if Angel is None:
        raise ModuleNotFoundError("Optional dependency not satisfied: install angel-cd library to use the selected feature (likely pip install angel-cd). If using a notebook, you need also to restart your runtime/kernel.")

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
    return NodeClustering(coms, g, "LAIS2", method_parameters={"":""}, overlap=True)


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

    if ig is None:
        raise ModuleNotFoundError("Optional dependency not satisfied: install igraph to use the selected feature.")

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

    if ig is None:
        raise ModuleNotFoundError("Optional dependency not satisfied: install igraph to use the selected feature.")

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


def big_clam(g, dimensions=8, iterations=50, learning_rate=0.005):
    """
    BigClam is an overlapping community detection method that scales to large networks.
    The procedure uses gradient ascent to create an embedding which is used for deciding the node-cluster affiliations.

    :param g: a networkx/igraph object
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

    g = convert_graph_formats(g, nx.Graph)

    model = BigClam(dimensions=dimensions, iterations=iterations, learning_rate=learning_rate)
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(coms, g, "BigClam", method_parameters={"dimensions": dimensions, "iterations": iterations,
                                                                 "learning_rate": learning_rate}, overlap=True)


def danmf(g, layers=(32, 8), pre_iterations=100, iterations=100, seed=42, lamb=0.01):
    """
    The procedure uses telescopic non-negative matrix factorization in order to learn a cluster memmbership distribution over nodes. The method can be used in an overlapping and non-overlapping way.

    :param g: a networkx/igraph object
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
    g = convert_graph_formats(g, nx.Graph)
    model = DANMF(layers, pre_iterations, iterations, seed, lamb)
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(coms, g, "DANMF", method_parameters={"layers": layers, "pre_iteration": pre_iterations,
                                                               "iterations": iterations, "seed": seed, "lamb": lamb},
                          overlap=True)


def egonet_splitter(g, resolution=1.0):
    """
    The method first creates the egonets of nodes. A persona-graph is created which is clustered by the Louvain method.

    :param g: a networkx/igraph object
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
    g = convert_graph_formats(g, nx.Graph)
    model = EgoNetSplitter(resolution=resolution)
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, cs in members.items():
        for c in cs:
            coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(coms, g, "EgoNetSplitter", method_parameters={"resolution":resolution}, overlap=True)


def nnsed(g, dimensions=32, iterations=10, seed=42):
    """
    The procedure uses non-negative matrix factorization in order to learn an unnormalized cluster membership distribution over nodes. The method can be used in an overlapping and non-overlapping way.

    :param g: a networkx/igraph object
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
    g = convert_graph_formats(g, nx.Graph)
    model = NNSED(dimensions=dimensions,iterations=iterations, seed=seed)
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(coms, g, "NNSED", method_parameters={"dimension": dimensions, "iterations": iterations,
                                                               "seed": seed}, overlap=True)


def nmnf(g, dimensions=128, clusters=10, lambd=0.2, alpha=0.05, beta=0.05, iterations=200, lower_control=1e-15, eta=5.0):
    """
    The procedure uses joint non-negative matrix factorization with modularity based regul;arization in order to learn a cluster memmbership distribution over nodes. The method can be used in an overlapping and non-overlapping way.

    :param g: a networkx/igraph object
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
    >>> coms = algorithms.nmnf(G)

    :References:

    Wang, Xiao, et al. "Community preserving network embedding." Thirty-first AAAI conference on artificial intelligence. 2017.

    .. note:: Reference implementation: https://karateclub.readthedocs.io/
    """
    g = convert_graph_formats(g, nx.Graph)
    model = MNMF(dimensions=dimensions, clusters=clusters, lambd=lambd, alpha=alpha, beta=beta, iterations=iterations,
                 lower_control=lower_control, eta=eta)
    model.fit(g)
    members = model.get_memberships()

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in members.items():
        coms_to_node[c].append(n)

    coms = [list(c) for c in coms_to_node.values()]

    return NodeClustering(coms, g, "MNMF", method_parameters={"dimension": dimensions, "clusters": clusters,
                                                              "lambd": lambd, "alpha": alpha, "beta": beta,
                                                              "iterations": iterations, "lower_control": lower_control,
                                                              "eta": eta}, overlap=True)

