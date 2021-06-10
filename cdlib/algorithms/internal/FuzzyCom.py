import numpy as np
import networkx as nx
from collections import defaultdict

"""
 Reference implementation 
 https://github.com/nidhisridhar/Fuzzy-Community-Detection
"""


def __reachable(i, theta_cores, fuzz_d, visited):
    # Returns indices of cores(in theta_cores) that are reachable from theta_cores[ i ]
    reach = []
    flag = True
    index = -1
    num_cores = len(theta_cores)
    while flag:
        if index == len(reach):
            flag = False
        if index == -1:
            flag = False
            for j in range(num_cores):
                if visited[j] == 0 and i != j:
                    if fuzz_d[theta_cores[i]][theta_cores[j]] > 0:
                        visited[j] = 1
                        reach.append(j)
                        flag = True
        else:
            for j in range(num_cores):
                if visited[j] == 0 and index != j:
                    if fuzz_d[theta_cores[index]][theta_cores[j]] > 0:
                        visited[j] = 1
                        reach.append(j)
                        flag = True
        index += 1
    return np.array(reach)


def __gran_embed(core, c, fuzz_d):
    # Return Normalized Granular Embeddedness of theta-core with community C
    num = 0
    den = 0
    c = np.array(c)
    c = np.append(c, core)
    n = len(fuzz_d[0])
    for i in range(n):
        num += np.min(fuzz_d[c, i])
        den += np.max(fuzz_d[c, i])
    return float(num) / den


def fuzzy_comm(graph, theta, eps, r):
    """
    Takes adjacency_mat(n*n) , theta , eps (epsilon) , and radius(r)
    and returns an n*c matrix where c is the number of communities and the
    i,jth value is the membership of node i in community j

    :param graph: networkx graph
    :param theta:
    :param eps:
    :param r:
    :return:
    """

    adjacency_mat = nx.to_numpy_matrix(graph)

    theta_cores = []
    num_vertices = adjacency_mat.shape[0]

    # Fuzzy granule initialization
    # gran = [i for i in range(num_vertices)]

    # Calculate distance between all vertices
    dist = list(nx.all_pairs_shortest_path_length(graph))

    # Membership values between all nodes
    fuzz_d = np.zeros(shape=adjacency_mat.shape).astype(float)
    for i in range(num_vertices):
        nid, n_dist = dist[i]
        for j in graph.nodes():
            if j in n_dist and n_dist[j] <= r:
                fuzz_d[nid][j] = 1 / float(1 + n_dist[j])
    _sum = np.sum(fuzz_d, axis=1)

    # Normalization of Membership
    for i in range(num_vertices):
        fuzz_d[i] = fuzz_d[i] / float(_sum[i])

        # Theta-cores Finding
    for i in range(num_vertices):
        if np.sum(fuzz_d[:, i]) >= theta:
            theta_cores.append(i)
    theta_cores = np.array(theta_cores)
    num_cores = len(theta_cores)
    _sum = np.sum(fuzz_d[:, theta_cores], axis=1)
    k = 0
    for i in range(num_vertices):
        fuzz_d[i] = fuzz_d[i] / _sum[k]
        k += 1

    # Finding Fuzzy Communities
    communities = []
    visited = np.zeros(num_cores)

    for i in range(num_cores):
        if visited[i] == 0:
            c = [theta_cores[i]]
            visited[i] = 1
            reach = __reachable(i, theta_cores, fuzz_d, visited.copy())
            for core_ind in reach:
                if __gran_embed(theta_cores[core_ind], c, fuzz_d) > eps:
                    c.append(theta_cores[core_ind])
                    visited[core_ind] = 1
            communities.append(c)

    cms = []
    for c in communities:
        cms.append([int(n) for n in c])

    # fuzzy association to communities
    fuzz_assoc = defaultdict(dict)
    for i in range(num_vertices):
        for j in range(len(cms)):
            fuzz_assoc[i][j] = np.sum(fuzz_d[i, cms[j]])

    return cms, fuzz_assoc
