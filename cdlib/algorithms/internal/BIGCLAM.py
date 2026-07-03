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
    """exp(-x) / (1 - exp(-x)), rewritten as 1 / expm1(x) (algebraically
    identical) to avoid the catastrophic cancellation of computing 1 - exp(-x)
    for small x. The function still diverges as x -> 0+ (a property of the
    underlying equation, not of this implementation), so x is clipped away
    from 0 to keep the result finite regardless of how close F's non-negativity
    floor pushes dot products to zero."""
    x = np.maximum(x, 1e-10)
    return 1.0 / np.expm1(x)


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

def gradient_fast(F, A, i):
    r"""Fast implementation of the gradient function, considering
    equation 4 of https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf

    .. math::

        \nabla l(F_u) =
        \sum_{v \in N(u)} F_v \left(1 + \frac{e^{-F_u^T F_v}}{1-e^{-F_u^T F_v}}\right) 
        - \sum_v F_v + F_u

    """
    _, C = F.shape
    neighbours = np.where(A[i])[0]

    grad = np.zeros((C,))
    for nb in neighbours:
        dotproduct = F[nb].dot(F[i])
        grad += F[nb] * (1 + sigm(dotproduct))
    grad -= np.sum(F, axis=0)
    grad += F[i]
    return grad

def get_embeddings(A, C, iterations=100, learning_rate=0.005, naive=False):
    # initialize an F
    N = A.shape[0]
    F = np.random.rand(N, C)

    for n in range(iterations):
        for person in range(N):
            if naive:
                grad = gradient(F, A, person)
            else:
                grad = gradient_fast(F, A, person)

            F[person] += learning_rate * grad

            F[person] = np.maximum(0.00001, F[person])  # F should be nonnegative
        # log_likelihood(F, A)
    return F

def get_communities(F, graph, number_communities, method='argmax'):
    if method == 'argmax':
        F_argmax = np.argmax(F, 1)
        dict_communities = {com: [] for com in range(number_communities)}
        for node, com in zip(graph.nodes(), F_argmax.tolist()):
            dict_communities[com].append(node)
    elif method == 'threshold':
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        if n < 2:
            raise ValueError(
                "The 'threshold' affiliation method requires a graph with at least 2 nodes."
            )
        epsilon = min(2 * m / (n * (n - 1)), 1 - 1e-10)
        delta = np.sqrt(-np.log(1 - epsilon))
        memberships = np.where(F >= delta, 1, 0)
        # in this case, a node can belong to multiple communities
        dict_communities = {com: [] for com in range(number_communities)}
        for node, membership in zip(graph.nodes(), memberships):
            for com in np.nonzero(membership)[0].tolist():
                dict_communities[com].append(node)
    else:
        raise ValueError(
            f"Unknown affiliation_method: '{method}'. Supported values are 'argmax' and 'threshold'."
        )

    list_communities = []
    for com in dict_communities:
        list_communities.append(dict_communities[com])

    return list_communities

def big_clam_communities(graph, number_communities, iterations=100, learning_rate=0.005, naive=False, affiliation_method='argmax'):
    adj = nx.to_numpy_array(graph, weight=None)
    F = get_embeddings(adj, number_communities, iterations=iterations, learning_rate=learning_rate, naive=naive)

    return get_communities(F, graph, number_communities, method=affiliation_method)
