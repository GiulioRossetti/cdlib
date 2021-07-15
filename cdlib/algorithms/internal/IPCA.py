# True Price
from itertools import combinations
from collections import defaultdict


def i_pca(g, weights=None, t_in=0.5):
    if weights is None:
        return __ipca(g, t_in)
    else:
        return __ipca_weighted(g, weights, t_in)


def __ipca(g, t_in=0.5):
    res = []
    data = defaultdict(set)  # node id => neighboring node ids

    # read in graph
    for a, b in g.edges():
        data[a].add(b)
        data[b].add(a)
    unvisited = []
    weights = defaultdict(int)
    for a, b in combinations(data, 2):
        if b in data[a]:
            shared = len(data[a] & data[b])
            weights[a] += shared
            weights[b] += shared

        unvisited = set(data)
    num_clusters = 0

    seed_nodes = sorted(data, key=lambda k: (weights[k], len(data[k])), reverse=True)

    for seed in seed_nodes:  # get highest degree node
        if seed in unvisited:
            cluster = {v for v in data[seed]}  # seed and random neighbor

            while True:
                # rank neighbors by the number of edges between the node and cluster nodes
                frontier = sorted(
                    (len(data[p] & cluster), p)
                    for p in set.union(*((data[n] - cluster) for n in cluster))
                )

                # do this until IN_vk < T_IN, SP <= 2 is met, or no frontier nodes left
                found = False
                while frontier and not found:
                    m_vk, p = frontier.pop()
                    if m_vk < t_in * len(cluster):
                        break
                    c_2neighbors = data[p] & cluster
                    c_2neighbors.update(*(data[c] & cluster for c in c_2neighbors))
                    if cluster == c_2neighbors:
                        found = True
                        break

                # otherwise, add the node to the cluster
                if found:
                    cluster.add(p)
                    continue

                break

            unvisited -= cluster
            res.append(list(cluster))

            num_clusters += 1

            if not unvisited:
                break
    return res


def __ipca_weighted(g, wlabel, t_in):
    data = defaultdict(set)  # node id => neighboring node ids
    global_edges, degrees = defaultdict(dict), defaultdict(float)
    res = []

    # read in graph
    for a, b, dt in g.edges(data=True):
        data[a].add(b)
        data[b].add(a)
        w = dt[wlabel]
        global_edges[a][b], global_edges[b][a] = w, w
        degrees[a] += w
        degrees[b] += w

    weights = defaultdict(int)
    for a, b in combinations(data, 2):
        if b in data[a]:
            shared = len(data[a] & data[b])
            weights[a] += shared
            weights[b] += shared

    unvisited = set(data)
    num_clusters = 0

    # DIFFERENT: weighted degrees
    seed_nodes = sorted(data, key=lambda k: (weights[k], degrees[k]), reverse=True)

    for seed in seed_nodes:  # get highest weight/degree node
        if seed in unvisited:
            cluster = {seed}

            while True:
                # DIFFERENT: rank neighbors by the sum of global edge weights
                # between the node and cluster nodes
                frontier = sorted(
                    (sum(global_edges[p][c] for c in data[p] & cluster), p)
                    for p in set.union(*((data[n] - cluster) for n in cluster))
                )

                # do this until IN_vK < T_IN, SP <= 2 is met, or no frontier nodes left
                found = False
                while frontier and not found:
                    m_vk, p = frontier.pop()
                    if m_vk < t_in * len(cluster):
                        break
                    c_2neighbors = data[p] & cluster
                    c_2neighbors.update(*(data[c] & cluster for c in c_2neighbors))
                    if cluster == c_2neighbors:
                        found = True
                        break

                if not found:
                    break

                cluster.add(p)  # otherwise, add the node to the cluster

            unvisited -= cluster
            if len(cluster) > 1:
                res.append(list(cluster))

                num_clusters += 1

            if unvisited:
                pass
            else:
                break
    return res
