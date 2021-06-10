"""

A reference internal for Diffusion Entropy Reducer graph clustering algorithm. See

M. Kozdoba and S. Mannor, Community Detection via Measure Space Embedding, NIPS 2015


Code from: https://github.com/komarkdev/der_graph_clustering

The main function is

der_graph_clustering(G, NCOMPONENTS = 2,
                            WALK_LEN = 3,
                            alg_threshold = None,
                            alg_iterbound = 100,
                            do_soften = True
                        )
Arguments:
G - an undirected networkx graph object,
The rest of the parameters are as described in the paper.
Usage example is in block_model_test.py


Code structure:
* Kmeans class implements a generic K-means like skeleton.
* WeightedMeasNodes encapuslates the specific likelihood computations.
* der_graph_clustering is the main function.
  It creates the walks, creates an initialization, runs the algorithm,
  and finally extracts the communities.


"""

import numpy as np
import scipy
import networkx as nx


def _multiply_matrix_rows(mults, M):
    N = M.shape[0]
    diag = scipy.sparse.dia_matrix((mults.reshape((1, N)), np.array([0])), shape=(N, N))
    return diag.dot(M)


class KMeans(object):
    STRICT_INCREASE_FLAG = True

    def __init__(self, init_params, data, node_implementation):
        assert scipy.sparse.isspmatrix_csr(
            data
        ), "data should be scipy sparse csr matrix"

        THR = 0.00000001

        assert (
            max(np.abs(data.sum(axis=1) - 1)) < THR
        ), "Non probabilities on input! - {}".format(max(np.abs(data.sum(axis=1) - 1)))

        data = _multiply_matrix_rows(1 / data.sum(axis=1), data)

        assert max(np.abs(data.sum(axis=1) - 1)) < THR, "Non probabilities on input!"
        assert (
            max(np.abs(init_params.sum(axis=1) - 1)) < THR
        ), "Non probabilities on params!"
        init_params = init_params / (
            init_params.sum(axis=1).reshape(init_params.shape[0], 1)
        )
        assert (
            max(np.abs(init_params.sum(axis=1) - 1)) < THR
        ), "Non probabilities on params!"

        self.params = init_params
        self.ncomps = init_params.shape[0]

        self.node_implementation = node_implementation
        self.data = data
        self.N = data.shape[0]

        self.Q = np.zeros((self.N, self.ncomps))
        self.node_probabilities = np.zeros((self.N, self.ncomps))

    def computeQ(self):

        for i in range(self.ncomps):
            self.node_probabilities[
                :, i
            ] = self.node_implementation.node_log_probabilities(
                self.data, self.params[i]
            )
        max_idxs = np.argmax(self.node_probabilities, axis=1)

        self.Q = np.zeros((self.N, self.ncomps))
        dist_count = 0

        for i in range(self.N):
            self.Q[i, max_idxs[i]] = 1.0
            dist_count += self.node_probabilities[i, max_idxs[i]]

        return dist_count

    def optimize_step(self):
        self.params = self.node_implementation.optimize(self.data, self.Q)

    def optimize(self, threshold, iterbound=100):

        self.loglikelihood = self.computeQ()

        step_cnt = 1
        while True:

            if step_cnt > iterbound:
                break

            self.optimize_step()

            loglikelihood = self.computeQ()

            if not self.STRICT_INCREASE_FLAG:
                likelihood_diff = np.abs(loglikelihood - self.loglikelihood)
            else:
                likelihood_diff = loglikelihood - self.loglikelihood
                assert likelihood_diff > -1.0e-10, "Likelihood decrease!! : {}".format(
                    likelihood_diff
                )

            self.loglikelihood = loglikelihood

            step_cnt += 1

            if (threshold != None) and (likelihood_diff < threshold):
                break

        return


def __rand_measure(k, smoother=0.01):
    rm = np.random.random(size=k) + smoother
    rm /= rm.sum()
    return rm


class WeightedMeasNodes(object):
    def __init__(self, weights, k):

        self.k = k
        self.kzeros = np.zeros(k)
        N = weights.shape[0]
        self.weights = weights.reshape((N, 1))

    def node_log_probabilities__(self, data, param):

        N = data.shape[0]
        k = self.k
        log_param = np.log(param)
        zero_idx = log_param == -np.inf
        log_param[zero_idx] = 0.0
        res = data.dot(log_param.reshape((k, 1))).reshape((N,))
        self.kzeros[zero_idx] = 1
        non_abs_cont = (data.dot(self.kzeros.reshape((k, 1))) > 0).reshape((N,))
        self.kzeros[zero_idx] = 0
        res[non_abs_cont] = -np.inf
        return res

    def node_log_probabilities(self, data, param):

        w = self.weights.reshape((data.shape[0],))
        log_probs = self.node_log_probabilities__(data, param)
        inf_idx = log_probs == -np.inf
        log_probs[inf_idx] = 0
        log_probs = (log_probs * w).reshape((data.shape[0],))
        log_probs[inf_idx] = -np.inf
        return log_probs.reshape((data.shape[0],))

    def optimize__(self, data, Q):
        k = self.k
        ncomp = Q.shape[1]
        params = np.zeros((ncomp, k))
        empty_components = []
        for i in range(ncomp):
            s = Q[:, i].sum()
            if s > 0:
                pos_idx = Q[:, i] > 0
                params[i, :] = _multiply_matrix_rows(
                    Q[pos_idx, i] / s, data[pos_idx, :]
                ).sum(axis=0)
            else:
                empty_components.append(i)

        assert len(empty_components) != ncomp, "All components empty!"

        for i in empty_components:
            params[i, :] = 0.0

        return params

    def optimize(self, data, Q):
        # this currently assumes data is very specific, i.e. of the length of weights
        return self.optimize__(data, self.weights * Q)

    def get_communities(self, params, data):

        communities = []
        NCOMPONENTS = params.shape[0]
        node_log_probs = np.zeros((self.k, NCOMPONENTS))
        for i in range(NCOMPONENTS):
            node_log_probs[:, i] = self.node_log_probabilities(data, params[i])

        labels = np.argmax(node_log_probs, axis=1)
        for i in range(NCOMPONENTS):
            communities.append(list(np.arange(self.k)[labels == i]))

        return communities

    def init_params_soften(self, params, alpha=0.000001):

        ncomp, k = params.shape
        unif = np.ones(k) / k

        return (1 - alpha) * params + alpha * unif.reshape((1, k))

    def init_params_random_subset_data(self, ncomp, data, weights=None):

        Ndata = data.shape[0]
        params = np.zeros((ncomp, Ndata))

        if weights is None:
            weights = np.ones(Ndata)

        step = Ndata / ncomp

        for i in range(ncomp):

            if i == ncomp - 1:
                params[i, int(i * step) :] = 1.0
            else:
                params[i, int(i * step) : int((i + 1) * step)] = 1.0

        perm_idx = np.random.permutation(Ndata)
        params = params[:, perm_idx]

        params = params * weights.reshape((1, Ndata))
        params = params / params.sum(axis=1).reshape((ncomp, 1))

        params = ((data.T).dot(params.T)).T

        return params


def __graph_transition_matrix(G, sparse=True):
    A = nx.adjacency_matrix(G).astype("float")
    # normalize rows to sum to 1
    degs = A.sum(axis=1)

    # take care of zero degree
    degs[degs == 0] = 1

    N = len(degs)

    if sparse == True:
        rev_degs = 1 / degs
        diag = scipy.sparse.dia_matrix(
            (rev_degs.reshape((1, N)), np.array([0])), shape=(N, N)
        )
        A = diag.dot(A)
    else:
        A = A.todense()
        A = A / degs.reshape((A.shape[0], 1))

    return A


def __create_walks(TM, WALK_LEN):
    # Should be faster, TM is sparse

    N = TM.shape[0]

    powers = [TM]
    for i in range(1, WALK_LEN):
        powers.append(powers[-1].dot(TM))

    totals = scipy.sparse.csr_matrix((N, N))
    for m in powers:
        totals = totals + m

    totals = totals / WALK_LEN

    return totals


def der_graph_clustering(
    graph,
    ncomponents=2,
    walk_len=3,
    alg_threshold=None,
    alg_iterbound=100,
    do_soften=True,
):

    TM = __graph_transition_matrix(graph, sparse=True)
    graph_size = TM.shape[0]

    degs = graph.degree()
    weights = np.array(list(map(lambda i: degs[i], graph.nodes())))

    assert sum(weights > 0) == len(weights), "Zero weights found!"

    data = __create_walks(TM, walk_len)

    MN = WeightedMeasNodes(weights, k=graph_size)

    init_params = MN.init_params_random_subset_data(ncomponents, data, weights)

    if do_soften:
        init_params = MN.init_params_soften(init_params, alpha=0.000001)

    alg = KMeans(init_params, data, MN)

    alg.optimize(alg_threshold, iterbound=alg_iterbound)
    communities = MN.get_communities(alg.params, data)

    return communities, alg
