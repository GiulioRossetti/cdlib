# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:05:52 2021
@author: Gianmarco
References: Desmond J.Highama Gabriela Kalnaa Milla Kibble, Spectral clustering and its use in bioinformatics
            Journal of Computational and Applied Mathematics
"""
import networkx as nx
import numpy as np

# Use this function to calculate a diagonal matrix useful to build  Laplacian matrix
def __diagonal_matrix(W):
    D = np.zeros([W.shape[0], W.shape[1]])
    for i in range(W.shape[0]):
        s = 0
        for j in range(W.shape[1]):
            s = s + W[i, j]

        D[i, i] = s
    return D


def spectral_communities(g, kmax, projection_on_smaller_class=True, scaler=None):
    """
    Constructor
    :param g: a networkx Graph object
    :param kmax: maximum number of desired communities
    :param projection_on_smaller_class: a boolean value that if True then it project a bipartite network in the smallest class of node. (default is True)
    :param scaler: the function to scale the fielder’s vector to apply KMeans
    """
    from networkx import bipartite
    from numpy import linalg as LA
    from scipy.linalg import eigh

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import MinMaxScaler

    # conditional statement to build weight matrix in two different cases
    if not bipartite.is_bipartite(g):
        b = list(g.nodes())
        W = nx.adjacency_matrix(g, b)

    else:
        top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
        bottom_nodes = set(g) - top_nodes

        b = list(bottom_nodes)
        t = list(top_nodes)

        A = np.array(bipartite.biadjacency_matrix(g, t, b).todense())
        A_t = A.transpose()
        if projection_on_smaller_class:
            W = np.matmul(A_t, A)
        else:
            W = np.matmul(A, A_t)

    D = __diagonal_matrix(W)
    I = np.linalg.inv(D)
    L = LA.multi_dot([np.sqrt(I), (D - W), np.sqrt(I)])
    E, V = eigh(L)  # returns eigenvalue and eigenvector sorted by eigenvalue
    fielder_vec = np.matmul(
        np.sqrt(I), V[:, 1]
    )  # the fielder vector is the second eigenvector ov V

    x = [i for i in range(len(fielder_vec))]
    fielder_nodes = [(fielder_vec[i], x[i]) for i in range(len(x))]
    fielder_nodes.sort()

    y = [e[0] for e in fielder_nodes]

    X = [[y[i], x[i]] for i in range(len(x))]
    # normalizing X before applying kMeans
    X = scaler.fit_transform(X)

    # Compute silhouette score to choose the best number of communities
    sill = []
    K = range(1, kmax + 1)

    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        if k != 1:
            sill.append(silhouette_score(X, km.labels_))

    k_best = 2 + max(range(len(sill)), key=sill.__getitem__)
    kmeans = KMeans(n_clusters=k_best)
    kmeans.fit(X)

    # build communities from clustering label
    communities = [[] for _ in range(k_best)]
    for i, e in enumerate(kmeans.labels_):
        communities[e].append(b[fielder_nodes[i][1]])

    return communities  # , b, fielder_vec
