###############################################################################
#  Link Partitioning Around Medoids https://arxiv.org/abs/1907.08731          #
#  Alexander Ponomarenko, Leonidas Pitsoulis, Marat Shamshetdinov             #
#                                                                             #
# Contact us:                                                                 #
# aponomarenko@hse.ru - Alexander                                             #
# pitsouli@auth.gr    - Leonidas                                              #
###############################################################################

from collections import defaultdict
import networkx as nx
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from cdlib import NodeClustering


def LPAM(graph, k=2, threshold=0.5, distance="amp", seed=0):
    """
    Link Partitioning Around Medoids

    :param graph: a networkx object
    :param k: number of clusters
    :param threshold: merging threshold in [0,1], default 0.5
    :param distance: type of distance: "amp" - amplified commute distance, or
    "cm" - commute distance, or distance matrix between all edges as np ndarray
    :param seed: random seed for k-medoid heuristic

    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.lpam(G, k=2, threshold=0.4, distance = "amp")

    :References:
    Link Partitioning Around Medoids https://arxiv.org/abs/1907.08731
    Alexander Ponomarenko, Leonidas Pitsoulis, Marat Shamshetdinov
    """

    def getCommuteDistace(G):
        """
        Returns commute distance matrix
        """
        verts = list(G.nodes)
        n = len(verts)
        vol = nx.volume(G, verts)

        # use NetworkX to get Laplacian
        L = nx.laplacian_matrix(G)
        L = L.todense()
        Gamma = L + (1 / n) * np.ones([n, n])
        CM = np.zeros([n, n])

        # get Moore-Penrose pseudo inverse
        Gamma_pinv = np.linalg.pinv(Gamma, rcond=1e-4)
        for i in range(n):
            for j in range(i + 1, n):
                CM[i, j] = vol * (
                    Gamma_pinv[i, i] + Gamma_pinv[j, j] - 2 * Gamma_pinv[i, j]
                )
                CM[j, i] = CM[i, j]
        return CM

    def getAmp(G):
        """
        Returns amplified commute distance matrix
        """
        verts = list(G.nodes)
        n = len(verts)

        # get adj matrix
        A = nx.adjacency_matrix(G)
        A = A.todense()

        # use NetworkX to get Laplacian
        L = nx.laplacian_matrix(G)
        L = L.todense()
        Gamma = L + (1 / n) * np.ones([n, n])
        C_AMP = np.zeros([n, n])

        # get Moore-Penrose pseudo inverse
        Gamma_pinv = np.linalg.pinv(Gamma, rcond=1e-4)
        for i in range(n):
            for j in range(i + 1, n):
                r_ij = (
                    Gamma_pinv[i, i] + Gamma_pinv[j, j] - 2 * Gamma_pinv[i, j]
                )  # resistance dist
                d_i = G.degree(list(G.nodes())[i])
                d_j = G.degree(list(G.nodes())[j])
                if d_i != 0 and d_j != 0:
                    s_ij = r_ij - (1 / d_i) - (1 / d_j)
                    w_ij = A[i, j]
                    w_ii = A[i, i]
                    w_jj = A[j, j]
                    u_ij = (
                        ((2 * w_ij) / (d_i * d_j))
                        - (w_ii / (d_i ** 2))
                        - (w_jj / (d_j ** 2))
                    )
                    C_AMP[i, j] = s_ij + u_ij
                    C_AMP[j, i] = s_ij + u_ij
                else:
                    C_AMP[i, j] = np.NaN
                    C_AMP[j, i] = np.NaN
        return C_AMP

    line_graph = nx.line_graph(graph)
    D = None
    distance_name = distance
    if distance == "amp":
        D = getAmp(line_graph)
    if distance == "cm":
        D = getCommuteDistace
    if isinstance(distance, np.ndarray):
        D = distance
        distance_name = "custom"
    if D is None:
        raise TypeError('Parameter distance should be "amp"/"cm", or numpy.ndarray')
    _n = len(line_graph.nodes())
    np.random.seed(0)
    initial_medoids = np.random.choice(_n, k, replace=False)
    kmedoids_instance = kmedoids(D, initial_medoids, data_type="distance_matrix")
    # run cluster analysis and obtain results
    kmedoids_instance.process()

    clusters = kmedoids_instance.get_clusters()

    final_clusters = {}
    for c_i, c in enumerate(clusters):
        for line_vertex in c:
            source, target = list(line_graph.nodes())[line_vertex]
            if source not in final_clusters:
                final_clusters[source] = []
            final_clusters[source].append(c_i)
            if target not in final_clusters:
                final_clusters[target] = []

            final_clusters[target].append(c_i)

    res_clusters = {}
    for v, l in final_clusters.items():
        degree = len(l)
        res = defaultdict(list)
        for x in l:
            res[x].append(x)
        covering = np.zeros(k)
        for c_i, _l in res.items():
            covering[c_i] = len(_l) / degree

        res_clusters[v] = covering

    _res_clusters = [[] for i in range(k)]

    for v, l in res_clusters.items():
        for i in range(k):
            if l[i] >= threshold:
                _res_clusters[i].append(v)

    return NodeClustering(
        communities=[c for c in _res_clusters if len(c) > 0],
        graph=graph,
        method_name="lpam " + distance_name,
        method_parameters={
            "k": k,
            "threshold": threshold,
            "distance": distance_name,
            "seed": seed,
        },
        overlap=True,
    )
