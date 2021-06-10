import numpy as np
import networkx as nx


def __knn(k, x):
    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(k + 1, metric="euclidean", n_jobs=-1).fit(x)
    k_neighbors = neigh.kneighbors(
        x,
        k + 1,
    )
    distance = np.array(k_neighbors[0][:, 1:])
    indices = np.array(k_neighbors[1][:, 1:])
    return distance, indices


def __w_matrix(distance, indices, ks, a=10):
    n = len(distance)

    weight_matrix = np.zeros([n, n])

    sigma2 = (a / n / ks) * np.linalg.norm(distance) ** 2

    if ks == 1:
        for i in range(n):
            j = indices[i][0]
            weight_matrix[i][j] = np.exp(-1 * (distance[i, 0] ** 2) / sigma2)
    else:
        for i in range(n):
            for k, j in enumerate(indices[i]):
                weight_matrix[i][j] = np.exp(-1 * (distance[i, k] ** 2) / sigma2)

    return weight_matrix, sigma2


def __k0graph(similarity):
    x = np.arange(len(similarity))
    y = [np.argmax(row) for row in similarity]

    vc = []

    for i in range(len(x)):
        x_index, y_index = -1, -1
        for k in range(len(vc)):
            if y[i] in vc[k]:
                y_index = k
            if x[i] in vc[k]:
                x_index = k

        if x_index == y_index and x_index != -1 and y_index != -1:
            continue
        elif x_index < 0 and y_index < 0:
            vc.append([x[i], y[i]])
        elif x_index >= 0 > y_index:
            vc[x_index].append(y[i])
        elif x_index < 0 <= y_index:
            vc[y_index].append(x[i])
        else:
            vc[x_index].extend(vc[y_index])
            del vc[y_index]

    return vc


def __get_affinity_matrix(vc, w):
    nc = len(vc)

    affinity = np.zeros([nc, nc])

    for i in range(nc):
        for j in range(i + 1, nc):
            ij = np.ix_(vc[i], vc[j])
            ji = np.ix_(vc[j], vc[i])

            w_ij, w_ji = w[ij], w[ji]
            ci, cj = len(vc[i]), len(vc[j])

            ones_i = np.ones((ci, 1))
            ones_j = np.ones((cj, 1))
            affinity[i][j] = (1 / ci ** 2) * np.transpose(ones_i).dot(w_ij).dot(
                w_ji
            ).dot(ones_i) + (1 / cj ** 2) * np.transpose(ones_j).dot(w_ji).dot(
                w_ij
            ).dot(
                ones_j
            )
            affinity[j][i] = affinity[i][j]
    return affinity


def __get_affinity_btw_cluster(c1, c2, w):
    ij = np.ix_(c1, c2)
    ji = np.ix_(c2, c1)

    w_ij, w_ji = w[ij], w[ji]
    ci, cj = len(c1), len(c2)

    ones_i = np.ones((ci, 1))
    ones_j = np.ones((cj, 1))
    affinity = (1 / ci ** 2) * np.transpose(ones_i).dot(w_ij).dot(w_ji).dot(ones_i) + (
        1 / cj ** 2
    ) * np.transpose(ones_j).dot(w_ji).dot(w_ij).dot(ones_j)
    return affinity[0, 0]


def __get_neighbor(vc, kc, w):
    ns, as_ = [], []
    A = __get_affinity_matrix(vc, w)

    for i in range(len(A)):
        as_.append([x for x in sorted(list(A[i]))[-1 * kc :] if x > 0])
        n = len(as_[i])
        if n == 0:
            ns.append([])
        else:
            ns.append(A[i].argsort()[-1 * n :].tolist())

    return ns, as_


def preprocess(data, ks, a):
    """
    From data to graph.
    __knn and __w_matrix only use for preprocess
    """
    distance, indices = __knn(ks, data)
    # convert distance to similarity
    similarity, _ = __w_matrix(distance, indices, ks, a)
    g = nx.from_numpy_matrix(similarity, create_using=nx.DiGraph)
    return g


def Agdl(g, target_cluster_num, kc):
    similarity = nx.to_numpy_matrix(g)
    # Using k0grpha to initilize cluster

    cluster = __k0graph(similarity)

    neighbor_set, affinity_set = __get_neighbor(cluster, kc, similarity)
    current_cluster_num = len(cluster)

    while current_cluster_num > target_cluster_num:

        max_affinity = 0
        max_index1 = 0
        max_index2 = 0
        for i in range(len(neighbor_set)):
            if len(neighbor_set[i]) == 0:
                continue
            aff = max(affinity_set[i])
            if aff > max_affinity:
                j = int(neighbor_set[i][affinity_set[i].index(aff)])
                max_affinity = aff

                if i < j:
                    max_index1 = i
                    max_index2 = j
                else:
                    max_index1 = j
                    max_index2 = i

        if max_index1 == max_index2:
            break

        # merge two cluster
        cluster[max_index1].extend(cluster[max_index2])
        cluster[max_index2] = []

        if max_index2 in neighbor_set[max_index1]:
            p = neighbor_set[max_index1].index(max_index2)
            del neighbor_set[max_index1][p]
        if max_index1 in neighbor_set[max_index2]:
            p = neighbor_set[max_index2].index(max_index1)
            del neighbor_set[max_index2][p]

        for i in range(len(neighbor_set)):
            if i == max_index1 or i == max_index2:
                continue

            if max_index1 in neighbor_set[i]:
                aff_update = __get_affinity_btw_cluster(
                    cluster[i], cluster[max_index1], similarity
                )

                p = neighbor_set[i].index(max_index1)
                affinity_set[i][p] = aff_update  # fix the affinity values

            if max_index2 in neighbor_set[i]:

                p = neighbor_set[i].index(max_index2)
                del neighbor_set[i][p]
                del affinity_set[i][p]

                if max_index1 not in neighbor_set[i]:
                    aff_update = __get_affinity_btw_cluster(
                        cluster[i], cluster[max_index1], similarity
                    )
                    neighbor_set[i].append(max_index1)
                    affinity_set[i].append(aff_update)

        neighbor_set[max_index1].extend(neighbor_set[max_index2])
        neighbor_set[max_index1] = list(set(neighbor_set[max_index1]))

        affinity_set[max_index1] = []

        neighbor_set[max_index2] = []
        affinity_set[max_index2] = []

        # Fine the Kc-nearest clusters for Cab

        for i in range(len(neighbor_set[max_index1])):
            target_index = neighbor_set[max_index1][i]
            new_affinity = __get_affinity_btw_cluster(
                cluster[target_index], cluster[max_index1], similarity
            )
            affinity_set[max_index1].append(new_affinity)

        if len(affinity_set[max_index1]) > kc:
            index = np.argsort(affinity_set[max_index1])
            new_neighbor = []
            new_affinity = []
            for j in range(kc):
                new_neighbor.append(neighbor_set[max_index1][index[-1 * j]])
                new_affinity.append(affinity_set[max_index1][index[-1 * j]])

            neighbor_set[max_index1] = new_neighbor
            affinity_set[max_index1] = new_affinity

        current_cluster_num = current_cluster_num - 1

    reduced_cluster = []
    for i in range(len(cluster)):
        if len(cluster[i]) != 0:
            reduced_cluster.append(cluster[i])

    return reduced_cluster
