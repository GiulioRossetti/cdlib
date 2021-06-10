import numpy as np
from scipy import sparse
from collections import deque
from collections import defaultdict
import networkx as nx


class MultiCom(object):
    # https://github.com/ahollocou/multicom
    #
    # Hollocou, Alexandre and Bonald, Thomas and Lelarge, Marc
    # "Multiple Local Community Detection"
    # International Symposium on Computer Performance, Modeling, Measurements and Evaluation 2017

    def __init__(self, g):
        self.g = g

    def __load_graph(self):
        """
        Load an undirected and unweighted graph from an edge-list file.
        :param edgelist_filename: string or unicode
            Path to the edge-list file.
            Id of nodes are assumed to be non-negative integers.
        :param delimiter: str, default '\t'
        :param comment: str, default '#'
        :return: Compressed Sparse Row Matrix
            Adjacency matrix of the graph
        """
        edge_df = nx.to_pandas_edgelist(self.g)
        edge_list = edge_df.values
        n_nodes = int(np.max(edge_list) + 1)
        adj_matrix = sparse.coo_matrix(
            (np.ones(edge_list.shape[0]), (edge_list[:, 0], edge_list[:, 1])),
            shape=tuple([n_nodes, n_nodes]),
            dtype=edge_list.dtype,
        )
        adj_matrix = adj_matrix.tocsr()
        adj_matrix = adj_matrix + adj_matrix.T
        return adj_matrix

    def __convert_adj_matrix(self, adj_matrix):
        """
        Convert an adjacency matrix to the Compressed Sparse Row type.
        :param adj_matrix: An adjacency matrix.
        :return: Compressed Sparse Row Matrix
            Adjacency matrix with the expected type.
        """
        if type(adj_matrix) == sparse.csr_matrix:
            return adj_matrix
        elif type(adj_matrix) == np.ndarray:
            return sparse.csr_matrix(adj_matrix)
        else:
            raise TypeError(
                "The argument should be a Numpy Array or a Compressed Sparse Row Matrix."
            )

    def __approximate_ppr(self, adj_matrix, seed_set, alpha=0.85, epsilon=1e-5):
        """
        Compute the approximate Personalized PageRank (PPR) from a set set of seed node.
        This function implements the push method introduced by Andersen et al.
        in "Local graph partitioning using pagerank vectors", FOCS 2006.
        :param adj_matrix: compressed sparse row matrix or numpy array
            Adjacency matrix of the graph
        :param seed_set: list or set of int
            Set of seed nodes.
        :param alpha: float, default 0.85
            1 - alpha corresponds to the probability for the random walk to restarts from the seed set.
        :param epsilon: float, default 1e-3
            Precision parameter for the approximation
        :return: numpy 1D array
            Vector containing the approximate PPR for each node of the graph.
        """
        adj_matrix = self.__convert_adj_matrix(adj_matrix)
        degree = np.array(np.sum(adj_matrix, axis=0))[0]
        n_nodes = adj_matrix.shape[0]

        prob = np.zeros(n_nodes)
        res = np.zeros(n_nodes)
        res[list(seed_set)] = 1.0 / len(seed_set)

        next_nodes = deque(seed_set)

        while len(next_nodes) > 0:
            node = next_nodes.pop()
            push_val = res[node] - 0.5 * epsilon * degree[node]
            res[node] = 0.5 * epsilon * degree[node]
            prob[node] += (1.0 - alpha) * push_val
            put_val = alpha * push_val
            for neighbor in adj_matrix[node].indices:
                old_res = res[neighbor]
                res[neighbor] += put_val * adj_matrix[node, neighbor] / degree[node]
                threshold = epsilon * degree[neighbor]
                if res[neighbor] >= threshold > old_res:
                    next_nodes.appendleft(neighbor)
        return prob

    def __conductance_sweep_cut(self, adj_matrix, score, window=10):
        """
        Return the sweep cut for conductance based on a given score.
        During the sweep process, we detect a local minimum of conductance using a given window.
        The sweep process is described by Andersen et al. in
        "Communities from seed sets", 2006.
        :param adj_matrix: compressed sparse row matrix or numpy array
            Adjacency matrix of the graph.
        :param score: numpy vector
            Score used to order the nodes in the sweep process.
        :param window: int, default 10
            Window parameter used for the detection of a local minimum of conductance.
        :return: set of int
             Set of nodes corresponding to the sweep cut.
        """
        adj_matrix = self.__convert_adj_matrix(adj_matrix)
        n_nodes = adj_matrix.shape[0]
        degree = np.array(np.sum(adj_matrix, axis=0))[0]
        total_volume = np.sum(degree)
        sorted_nodes = [node for node in range(n_nodes) if score[node] > 0]
        sorted_nodes = sorted(sorted_nodes, key=lambda node: score[node], reverse=True)
        sweep_set = set()
        volume = 0.0
        cut = 0.0
        best_conductance = 1.0
        best_sweep_set = {sorted_nodes[0]}
        inc_count = 0
        for node in sorted_nodes:
            volume += degree[node]
            for neighbor in adj_matrix[node].indices:
                if neighbor in sweep_set:
                    cut -= 1
                else:
                    cut += 1
            sweep_set.add(node)

            if volume == total_volume:
                break
            conductance = cut / min(volume, total_volume - volume)
            if conductance < best_conductance:
                best_conductance = conductance
                # Make a copy of the set
                best_sweep_set = set(sweep_set)
                inc_count = 0
            else:
                inc_count += 1
                if inc_count >= window:
                    break
        return best_sweep_set

    def execute(self, seed_node, clustering=None, n_steps=5, explored_ratio=0.8):
        """
        Algorithm for multiple local algorithms detection from a seed node.
        It implements the algorithm presented by Hollocou, Bonald and Lelarge in
        "Multiple Local Community Detection".
        :param g: networkx graph
        :param seed_node: int
            Id of the seed node around which we want to detect communities.
        :param clustering: Scikit-Learn Cluster Estimator
            Algorithm used to cluster nodes in the local embedding space.
            Example: sklearn.cluster.DBSCAN()
        :param n_steps: int, default 5
            Parameter used to control the number of detected communities.
        :param explored_ratio: float, default 0.8
            Parameter used to control the number of new seeds at each step.
        :return:
        seeds: list of int
            Seeds used to detect communities around the initial seed (including this original seed).
        communities: list of set
            Communities detected around the seed node.
        """
        seeds = dict()
        scores = dict()
        communities = list()
        explored = set()

        if clustering is None:
            from sklearn.cluster import DBSCAN

            clustering = DBSCAN()

        adj_matrix = self.__load_graph()
        adj_matrix = self.__convert_adj_matrix(adj_matrix)
        n_nodes = adj_matrix.shape[0]
        degree = np.array(np.sum(adj_matrix, axis=0))[0]

        new_seeds = [seed_node]
        step = -1
        n_iter = 0

        while step < n_steps and len(new_seeds) > 0:
            n_iter += 1

            for new_seed in new_seeds:
                step += 1
                seeds[step] = new_seed
                scores[step] = self.__approximate_ppr(adj_matrix, [seeds[step]])
                community = self.__conductance_sweep_cut(adj_matrix, scores[step])
                communities.append(community)
                # We add the algorithms to the explored nodes
                explored |= set(community)

            new_seeds = list()

            # Clustering of the nodes in the space (scores[seed1], scores[seed2],...,)
            embedding = np.zeros((n_nodes, step + 1))
            for i in range(step + 1):
                embedding[:, i] = scores[i][:]
            indices = np.where(np.sum(embedding, axis=1))[0]
            y = clustering.fit_predict(embedding[indices, :])
            clusters = defaultdict(set)
            for i in range(y.shape[0]):
                if y[i] != -1:
                    clusters[y[i]].add(indices[i])

            # Pick new seeds in unexplored clusters
            for c in range(len(clusters)):
                cluster_size = 0
                cluster_explored = 0
                for node in clusters[c]:
                    cluster_size += 1
                    if node in explored:
                        cluster_explored += 1
                if float(cluster_explored) / float(cluster_size) < explored_ratio:
                    candidates = list(set(clusters[c]) - explored)
                    candidate_degrees = np.array([degree[node] for node in candidates])
                    new_seeds.append(candidates[np.argmax(candidate_degrees)])

        return list(communities)

    @staticmethod
    def __get_node_membership(communities):
        """
        Get the algorithms membership for each node given a list of communities.
        :param communities: list of list of int
            List of communities.
        :return: membership: dict (defaultdict) of set of int
            Dictionary such that, for each node,
            membership[node] is the set of algorithms ids to which the node belongs.
        """
        membership = defaultdict(set)
        for i, community in enumerate(communities):
            for node in community:
                membership[node].add(i)
        return membership
