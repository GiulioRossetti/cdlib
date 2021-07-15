# Author: True Price <jtprice@cs.unc.edu>

from collections import defaultdict


def m_code(g, weights=None, weight_threshold=0.2):
    weight_threshold = 1 - weight_threshold
    if weights is None:
        return __mcode_unweighted(g, weight_threshold)
    else:
        return __mcode_weighted(g, weights, weight_threshold)


def __mcode_unweighted(g, weight_threshold):
    res = []
    edges = defaultdict(set)  # node id => neighboring node ids

    # read in graph

    for a, b in g.edges():
        edges[a].add(b)
        edges[b].add(a)

    # Stage 1: Vertex Weighting
    weights = dict((v, 1.0) for v in edges)

    for i, v in enumerate(edges):
        neighborhood = {v} | edges[v]
        # see if larger k-cores exist
        if len(neighborhood) > 2:
            k = 1  # highest valid k-core
            while neighborhood:
                k_core = neighborhood.copy()
                invalid_nodes = True
                while invalid_nodes and neighborhood:
                    invalid_nodes = set(
                        n for n in neighborhood if len(edges[n] & neighborhood) <= k
                    )
                    neighborhood -= invalid_nodes
                k += 1  # on exit, k will be one greater than we want
            # vertex weight = k-core number * density of k-core
            weights[v] = (k - 1) * (
                sum(len(edges[n] & k_core) for n in k_core) / (2.0 * len(k_core) ** 2)
            )

        # if node has only one neighbor, we know everything we need to know

    # Stage 2: Molecular Complex Prediction
    unvisited = set(edges)
    num_clusters = 0
    for seed in sorted(weights, key=weights.get, reverse=True):
        if seed in unvisited:
            cluster, frontier = {seed}, {seed}
            w = weights[seed] * weight_threshold
            while frontier:
                cluster.update(frontier)
                unvisited -= frontier
                frontier = set(
                    n
                    for n in set.union(*(edges[n] for n in frontier)) & unvisited
                    if weights[n] > w
                )

            # haircut: only keep 2-core complexes
            invalid_nodes = True
            while invalid_nodes and cluster:
                invalid_nodes = set(n for n in cluster if len(edges[n] & cluster) < 2)
                cluster -= invalid_nodes

            if cluster:
                res.append(list(cluster))
                num_clusters += 1
    return res


def __mcode_weighted(g, wlabel, weight_threshold):
    res = []
    graph = defaultdict(set)  # node id => neighboring node ids
    edges = defaultdict(lambda: defaultdict(int))

    # read in graph
    for a, b, data in g.edges(data=True):
        graph[a].add(b)
        graph[b].add(a)
        w = data[wlabel]
        edges[a][b] = w
        edges[b][a] = w

    # Stage 1: Vertex Weighting
    weights = dict((v, sum(edges[v].values()) / len(edges[v]) ** 2) for v in graph)
    for i, v in enumerate(graph):

        neighborhood = {v} | graph[v]
        # valid k-core with the highest weight
        if len(neighborhood) > 2:
            k = 2  # already covered k = 1
            while True:
                invalid_nodes = True
                while invalid_nodes and neighborhood:
                    invalid_nodes = set(
                        n for n in neighborhood if len(graph[n] & neighborhood) < k
                    )
                    neighborhood -= invalid_nodes
                if not neighborhood:
                    break

                # vertex weight = k-core number * density of k-core
                weights[v] = max(
                    weights[v],
                    k * sum(edges[v][n] for n in neighborhood) / len(neighborhood) ** 2,
                )
                k += 1

        # if node has only one neighbor, we know everything we need to know

    # Stage 2: Molecular Complex Prediction
    unvisited = set(graph)
    num_clusters = 0
    for seed in sorted(weights, key=weights.get, reverse=True):
        if seed in unvisited:
            cluster, frontier = {seed}, {seed}
            w = weights[seed] * weight_threshold
            while frontier:
                cluster.update(frontier)
                unvisited -= frontier
                frontier = set(
                    n
                    for n in set.union(*(graph[n] for n in frontier)) & unvisited
                    if weights[n] > w
                )

            # haircut: only keep 2-core complexes
            invalid_nodes = True
            while invalid_nodes and cluster:
                invalid_nodes = set(n for n in cluster if len(graph[n] & cluster) < 2)
                cluster -= invalid_nodes

            if cluster:
                res.append(list(cluster))
                num_clusters += 1
                if not unvisited:
                    break  # quit if all nodes visited
    return res
