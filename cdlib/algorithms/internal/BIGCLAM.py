"""
Implementation of the bigClAM algorithm.
Throughout the code, we will use tho following variables
  * F refers to the membership preference matrix. It's in [NUM_PERSONS, NUM_COMMUNITIES]
   so index (p,c) indicates the preference of person p for algorithms c.
  * A refers to the adjency matrix, also named friend matrix or edge set. It's in [NUM_PERSONS, NUM_PERSONS]
    so index (i,j) indicates is 1 when person i and person j are friends.
"""

import numpy as np
import networkx as nx


def sigm(x):
    return np.divide(np.exp(-1.0 * x), 1.0 - np.exp(-1.0 * x))


def log_likelihood(F, A):
    """implements equation 2 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf"""
    A_soft = F.dot(F.T)

    # Next two lines are multiplied with the adjacency matrix, A
    # A is a {0,1} matrix, so we zero out all elements not contributing to the sum
    FIRST_PART = A * np.log(1.0 - np.exp(-1.0 * A_soft))
    sum_edges = np.sum(FIRST_PART)
    SECOND_PART = (1 - A) * A_soft
    sum_nedges = np.sum(SECOND_PART)

    log_likeli = sum_edges - sum_nedges
    return log_likeli


def gradient(F, A, i):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf

      * i indicates the row under consideration

    The many forloops in this function can be optimized, but for
    educational purposes we write them out clearly
    """
    N, C = F.shape
    neighbours = np.where(A[i])
    nneighbours = np.where(1 - A[i])

    sum_neigh = np.zeros((C,))
    for nb in neighbours[0]:
        dotproduct = F[nb].dot(F[i])
        sum_neigh += F[nb] * sigm(dotproduct)

    sum_nneigh = np.zeros((C,))
    # Speed up this computation using eq.4
    for nnb in nneighbours[0]:
        sum_nneigh += F[nnb]

    grad = sum_neigh - sum_nneigh
    return grad


def train(A, C, iterations=100):
    # initialize an F
    N = A.shape[0]
    F = np.random.rand(N, C)

    for n in range(iterations):
        for person in range(N):
            grad = gradient(F, A, person)

            F[person] += 0.005 * grad

            F[person] = np.maximum(0.001, F[person])  # F should be nonnegative
        log_likelihood(F, A)
    return F


def big_Clam(graph, number_communities):
    adj = nx.to_numpy_matrix(graph)
    F = train(adj, number_communities)
    F_argmax = np.argmax(F, 1)
    dict_communities = {}
    for i in range(0, number_communities):
        dict_communities[i] = []
    for node, com in zip(graph.nodes(), F_argmax):
        dict_communities[com].append(node)

    list_communities = []
    for com in dict_communities:
        list_communities.append(dict_communities[com])

    return list_communities
