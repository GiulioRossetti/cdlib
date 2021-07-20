import warnings
import networkx as nx
import numpy as np
import scipy

warnings.filterwarnings("ignore")

"""
Reconstructing [1]
[1] Understanding Regularized Spectral Clustering via Graph Conductance Yilin Zhang, Karl Rohe: NIPS'18
https://arxiv.org/pdf/1806.01468.pdf
"""


def __regularized_laplacian_matrix(adj_matrix, tau):
    """
    Using ARPACK solver, compute the first K eigen vector.
    The laplacian is computed using the regularised formula from [2]
    [2]Kamalika Chaudhuri, Fan Chung, and Alexander Tsiatas 2018.
        Spectral clustering of graphs with general degrees in the extended planted partition model.
    L = I - D^-1/2 * A * D ^-1/2
    :param adj_matrix: adjacency matrix representation of graph where [m][n] >0 if there is edge and [m][n] = weight
    :param tau: the regularisation constant
    :return: the first K eigenvector
    """
    import scipy.sparse

    # Code inspired from nx.normalized_laplacian_matrix, with changes to allow regularisation
    n, m = adj_matrix.shape
    I = np.eye(n, m)
    diags = adj_matrix.sum(axis=1).flatten()
    # add tau to the diags to produce a regularised diags
    if tau != 0:
        diags = np.add(diags, tau)

    # diags will be zero at points where there is no edge and/or the node you are at
    #  ignore the error and make it zero later
    with scipy.errstate(divide="ignore"):
        diags_sqrt = 1.0 / scipy.sqrt(diags)
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    D = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format="csr")

    L = I - (D.dot(adj_matrix.dot(D)))
    return L


def __eigen_solver(laplacian, n_clusters):
    """
    ARPACK eigen solver in Shift-Invert Mode based on http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    """
    from scipy.sparse.linalg import eigsh

    lap = laplacian * -1
    v0 = np.random.uniform(-1, 1, lap.shape[0])
    eigen_values, eigen_vectors = eigsh(lap, k=n_clusters, sigma=1.0, v0=v0)
    eigen_vectors = eigen_vectors.T[n_clusters::-1]
    return eigen_values, eigen_vectors[:n_clusters].T


def __regularized_spectral_clustering(adj_matrix, tau, n_clusters, algo="scan"):
    """
    :param adj_matrix: adjacency matrix representation of graph where [m][n] >0 if there is edge and [m][n] = weight
    :param n_clusters: cluster partitioning constant
    :param algo: the clustering separation algorithm, possible value kmeans++ or scan
    :return: labels, number of clustering iterations needed, smallest set of cluster found, execution time
    """
    from sklearn.cluster import k_means
    from sklearn.cluster._spectral import discretize

    regularized_laplacian = __regularized_laplacian_matrix(adj_matrix, tau)
    eigen_values, eigen_vectors = __eigen_solver(
        regularized_laplacian, n_clusters=n_clusters
    )
    if algo == "kmeans++":
        _, labels, _, num_iterations = k_means(
            eigen_vectors, n_clusters=n_clusters, return_n_iter=True
        )
    else:
        if n_clusters == 2:  # cluster based on sign
            second_eigen_vector_index = np.argsort(eigen_values)[1]
            second_eigen_vector = eigen_vectors.T[second_eigen_vector_index]
            labels = [
                0 if val <= 0 else 1 for val in second_eigen_vector
            ]  # use only the second eigenvector
            num_iterations = 1
        else:  # bisecting it into k-ways, use all eigenvectors
            labels = discretize(eigen_vectors)
            num_iterations = 20  # assume worst case scenario that it tooks 20 restarts

    smallest_cluster_size = min(np.sum(labels), abs(np.sum(labels) - len(labels)))
    return labels, num_iterations, smallest_cluster_size


def __sklearn_kmeans(adj_matrix, n_clusters):
    from sklearn.cluster import k_means

    _, labels, _, num_iterations = k_means(
        adj_matrix, n_clusters=n_clusters, return_n_iter=True
    )

    smallest_cluster_size = min(np.sum(labels), abs(np.sum(labels) - labels.size))
    return labels, num_iterations, smallest_cluster_size


def __sklearn_spectral_clustering(adj_matrix, n_clusters):
    """
    :param adj_matrix: adjacency matrix representation of graph where [m][n] >0 if there is edge and [m][n] = weight
    :param n_clusters: cluster partitioning constant
    :return: labels, number of clustering iterations needed, smallest set of cluster found, execution time
    """
    from sklearn.cluster import k_means
    from sklearn.neighbors import kneighbors_graph
    from sklearn.manifold import spectral_embedding

    connectivity = kneighbors_graph(adj_matrix, n_neighbors=10, include_self=True)
    affinity_matrix_ = 0.5 * (connectivity + connectivity.T)

    eigen_vectors = spectral_embedding(
        affinity_matrix_,
        n_components=n_clusters,
        eigen_solver="arpack",
        eigen_tol=0.0,
        norm_laplacian=True,
        drop_first=False,
    )

    _, labels, _, num_iterations = k_means(
        eigen_vectors, n_clusters=n_clusters, return_n_iter=True
    )

    smallest_cluster_size = min(np.sum(labels), abs(np.sum(labels) - labels.size))
    return labels, num_iterations, smallest_cluster_size


def rsc_evaluate_graph(
    graph: nx.Graph, n_clusters=2, method="vanilla", percentile=None
):
    """
    Reconsutrction of [1]Understanding Regularized Spectral Clustering via Graph Conductance, Yilin Zhang, Karl Rohe
    :param graph: Graph to be evaluated
    :param n_clusters: How many clusters to look at
    :param method: one among "vanilla", "regularized", "regularized_with_kmeans", "sklearn_spectral_embedding", "sklearn_kmeans", "percentile".
    :param percentile: value in [0, 100]. Mandatory if method="percentile", otherwise None
    :return:
    """
    # Experiment only on undirected graphs
    if graph.is_directed():
        graph = graph.to_undirected()

    # Before computing anything, largest connected component identified and used
    graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

    adj_matrix = nx.to_scipy_sparse_matrix(graph, format="csr")

    if method == "sklearn_spectral_embedding":
        (
            labels,
            num_iterations,
            smallest_cluster_size,
        ) = __sklearn_spectral_clustering(adj_matrix, n_clusters)

    elif method == "sklearn_kmeans":
        labels, num_iterations, smallest_cluster_size = __sklearn_kmeans(
            adj_matrix, n_clusters
        )

    elif method == "vanilla":
        (
            labels,
            num_iterations,
            smallest_cluster_size,
        ) = __regularized_spectral_clustering(adj_matrix, 0, n_clusters)

    elif method == "regularized_with_kmeans":

        graph_degree = graph.degree()
        graph_average_degree = (
            np.sum(val for (node, val) in graph_degree) / graph.number_of_nodes()
        )
        (
            labels,
            num_iterations,
            smallest_cluster_size,
        ) = __regularized_spectral_clustering(
            adj_matrix, graph_average_degree, n_clusters, "kmeans++"
        )
    else:
        graph_degree = graph.degree()
        np.percentile(graph_degree, percentile)
        (
            labels,
            num_iterations,
            smallest_cluster_size,
        ) = __regularized_spectral_clustering(
            adj_matrix, percentile, n_clusters, "scan"
        )

    return labels
