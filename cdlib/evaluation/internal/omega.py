from __future__ import division
from itertools import combinations
from collections import Counter


class Omega(object):
    def __init__(self, comms1, comms2):
        self.nodes1 = self.get_node_assignment(comms1)
        self.nodes2 = self.get_node_assignment(comms2)
        self.nodes = list(
            set().union(
                [node for i, com in comms2.items() for node in com],
                [node for i, com in comms1.items() for node in com],
            )
        )
        J, K, N, obs, tuples1, tuples2 = self.__observed()
        exp = self.__expected(J, K, N, tuples1, tuples2)
        self.omega_score = self.__calc_omega(obs, exp)

    @staticmethod
    def get_node_assignment(comms):
        """
        returns a dictionary with node-cluster assignments of the form {node_id :[cluster1, cluster_3]}
        :param comms:
        :return:
        """
        nodes = {}
        for i, com in comms.items():
            for node in com:
                try:
                    nodes[node].append(i)
                except KeyError:
                    nodes[node] = [i]
        return nodes

    @staticmethod
    def num_of_common_clusters(u, v, nodes_dict):
        """
        return the number of clusters in which the pair u,v appears in the
        :param u:
        :param v:
        :param nodes_dict:
        :return:
        """
        try:
            _sum = len(set(nodes_dict[u]) & set(nodes_dict[v]))
        except KeyError:
            _sum = 0
        return _sum

    def __observed(self):
        N = 0
        tuples1 = {}
        J = 0
        for u, v in combinations(self.nodes, 2):
            N += 1
            n = self.num_of_common_clusters(u, v, self.nodes1)
            tuples1[(u, v)] = self.num_of_common_clusters(u, v, self.nodes1)
            J = n if n > J else J
        tuples2 = {}
        K = 0
        for u, v in combinations(self.nodes, 2):
            n = self.num_of_common_clusters(u, v, self.nodes2)
            tuples2[(u, v)] = self.num_of_common_clusters(u, v, self.nodes2)
            K = n if n > K else K

        A = {j: 0 for j in range(min(J, K) + 1)}
        for (u, v), n in tuples1.items():
            try:
                if n == tuples2[(u, v)]:
                    A[n] += 1
            except KeyError:
                pass
        obs = sum(A[j] / N for j in range(min(J, K) + 1))
        return J, K, N, obs, tuples1, tuples2

    @staticmethod
    def __expected(J, K, N, tuples1, tuples2):
        N1 = Counter(tuples1.values())
        N2 = Counter(tuples2.values())
        exp = sum((N1[j] * N2[j]) / (N ** 2) for j in range(min(J, K) + 1))
        return exp

    @staticmethod
    def __calc_omega(obs, exp):
        if exp == obs == 1:
            return 1.0
        else:
            return (obs - exp) / (1 - exp)
