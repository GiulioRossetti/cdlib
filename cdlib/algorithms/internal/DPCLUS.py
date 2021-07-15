# Author: True Price <jtprice@cs.unc.edu>
from itertools import combinations
from collections import defaultdict


def dp_clus(g, weights=None, d_threshold=0.9, cp_threshold=0.5, overlap=True):
    if weights is None:
        if overlap:
            return __dpclus(g, d_threshold, cp_threshold)
        else:
            return __dpclus_crisp(g, d_threshold, cp_threshold)
    else:
        return __dpclus_w(g, d_threshold, cp_threshold, weights)


def __dpclus(g, d_threshold=0.9, cp_threshold=0.5):
    res = []
    data = defaultdict(set)  # node id => neighboring node ids

    # to match the original DPClus output, we need to keep track of the order in
    # which nodes were read from the file, in order to maintain cluster order when
    # breaking ties in our priority ranking (values are decreasing, i.e. negative)
    node_index = dict()
    b_list = []  # the original algorithm indexes column b after column a... >:-(

    # read in graph
    for a, b in g.edges():
        data[a].add(b)
        data[b].add(a)
        node_index.setdefault(a, -len(node_index))
        # node_index.setdefault(b, -len(node_index))
        b_list.append(b)

    for b in b_list:
        node_index.setdefault(b, -len(node_index))

    unvisited = set(data)
    num_clusters = 0

    while unvisited:
        # get highest degree node
        seed = max(unvisited, key=lambda k: (len(data[k] & unvisited), node_index[k]))
        frontier = data[seed] & unvisited
        if not frontier:
            break  # no connections left to analyze

        edges, weights = defaultdict(lambda: defaultdict(int)), defaultdict(int)
        for a, b in combinations(unvisited, 2):
            if b not in data[a]:
                continue
            shared = len(data[a] & data[b] & unvisited)
            edges[a][b], edges[b][a] = shared, shared
            weights[a] += shared
            weights[b] += shared

        max_w, _, node = max((w, node_index[n], n) for n, w in weights.items())
        if max_w > 0:
            seed = node
            frontier = data[seed] & unvisited

        cluster = {seed}
        cluster_degrees = {seed: 0}
        nn, ne = 1, 0  # number of nodes, edges in cluster

        while frontier:
            # find higest priority node:
            # 0. number of edges between node and cluster nodes
            # 1. sum of edge weights between node and cluster nodes
            # 2. the node's index
            # 3. the node itself
            e_nk, _, _, p = max(
                (
                    len(data[n] & cluster),
                    sum(edges[n][c] for c in cluster),
                    node_index[n],
                    n,
                )
                for n in frontier
            )

            density = 2.0 * (ne + e_nk) / (nn * (nn + 1))
            if density < d_threshold:
                break  # adding the node gives too low density; cluster is finished

            # if all nodes only have one connecting edge in cluster, use "fine-tuning"
            # this orders by the number of neighbors in the frontier, minus the
            # connectedness of the attached (cluster) node within the cluster
            cp = cp_threshold
            if e_nk == 1 and len(cluster) > 1:
                n_degree = dict()  # node => fine-tuning parameter
                for n in frontier:
                    for c in cluster:  # find adjacent cluster node
                        if n in edges[c]:
                            break
                    n_degree[n] = len(data[n] & frontier) - cluster_degrees[c]
                p = max(frontier, key=lambda k: (n_degree[k], node_index[k]))
                if n_degree[p] > 0:
                    cp /= 2.0
            if (e_nk / density / (nn + 1)) < cp:
                break  # no good node found; cluster is finished

            # otherwise, add the node to the cluster
            cluster.add(p)
            nn, ne = (nn + 1), (ne + e_nk)

            cluster_degrees[p] = e_nk
            for n in data[p] & cluster:
                cluster_degrees[n] += 1

            frontier = set.union(*((data[n] - cluster) & unvisited for n in cluster))

        # add overlapping nodes
        # frontier[2] stores our fine-tuning parameter, in this case
        frontier_nodes = set.union(*(data[c] for c in cluster)) - cluster
        frontier = sorted(
            [
                len(data[n] & cluster),
                sum(edges[n][c] for c in cluster),
                0,
                node_index[n],
                n,
            ]
            for n in frontier_nodes
        )
        # "fine-tuning"
        fine_tuning = False
        if frontier and frontier[-1][0] == 1:
            for n in frontier:
                for c in cluster:  # find adjacent cluster node
                    if n[4] in edges[c]:
                        break
                n[2] = len(data[n[4]] & frontier_nodes) - cluster_degrees[c]
            frontier.sort(key=lambda k: (k[2], k[3]))
            fine_tuning = True
        # iterate through visited neighbors and update accordingly
        while frontier:
            e_nk, _, w, _, p = frontier.pop()
            cp = cp_threshold / 2.0 if fine_tuning and w > 0 else cp_threshold

            density = 2.0 * (ne + e_nk) / (nn * (nn + 1))
            if density < d_threshold or (e_nk / density / (nn + 1)) < cp:
                continue

            # add node to the cluster
            cluster.add(p)
            nn, ne = (nn + 1), (ne + e_nk)

            cluster_degrees[p] = e_nk
            for n in data[p] & cluster:
                cluster_degrees[n] += 1

            # update E_nk for the other nodes on the frontier
            for n in frontier:
                if p in edges[n[4]]:
                    n[0] += 1

        unvisited -= cluster
        res.append(list(cluster))

        num_clusters += 1

    return res


def __dpclus_crisp(g, d_threshold, cp_threshold):
    res = []
    data = defaultdict(set)  # node id => neighboring node ids

    # to match the original DPClus output, we need to keep track of the order in
    # which nodes were read from the file, in order to maintain cluster order when
    # breaking ties in our priority ranking (values are decreasing, i.e. negative)
    node_index = dict()
    b_list = []  # the original algorithm indexes column b after column a... >:-(

    # read in graph
    for a, b in g.edges():
        data[a].add(b)
        data[b].add(a)
        node_index.setdefault(a, -len(node_index))
        # node_index.setdefault(b, -len(node_index))
        b_list.append(b)
    for b in b_list:
        node_index.setdefault(b, -len(node_index))

    unvisited = set(data)
    num_clusters = 0

    while unvisited:
        # get highest degree node
        seed = max(unvisited, key=lambda k: (len(data[k] & unvisited), node_index[k]))
        frontier = data[seed] & unvisited
        if not frontier:
            break  # no connections left to analyze

        edges, weights = defaultdict(lambda: defaultdict(int)), defaultdict(int)
        for a, b in combinations(unvisited, 2):
            if b not in data[a]:
                continue
            shared = len(data[a] & data[b] & unvisited)
            edges[a][b], edges[b][a] = shared, shared
            weights[a] += shared
            weights[b] += shared

        max_w, _, node = max((w, node_index[n], n) for n, w in weights.items())
        if max_w > 0:
            seed = node
            frontier = data[seed] & unvisited

        cluster = {seed}
        cluster_degrees = {seed: 0}
        nn, ne = 1, 0  # number of nodes, edges in cluster

        while frontier:
            # find higest priority node:
            # 0. number of edges between node and cluster nodes
            # 1. sum of edge weights between node and cluster nodes
            # 2. the node itself, as indexed by node_index
            e_nk, _, _, p = max(
                (
                    len(data[n] & cluster),
                    sum(edges[n][c] for c in cluster),
                    node_index[n],
                    n,
                )
                for n in frontier
            )

            density = 2.0 * (ne + e_nk) / (nn * (nn + 1))
            if density < d_threshold:
                break  # adding the node gives too low density; cluster is finished

            # if all nodes only have one connecting edge in cluster, use "fine-tuning"
            # this orders by the number of neighbors in the frontier, minus the
            # connectedness of the attached (cluster) node within the cluster
            cp = cp_threshold
            if e_nk == 1 and len(cluster) > 1:
                n_degree = dict()  # node => fine-tuning parameter
                for n in frontier:
                    for c in cluster:  # find adjacent cluster node
                        if n in edges[c]:
                            break
                    n_degree[n] = len(data[n] & frontier) - cluster_degrees[c]
                p = max(frontier, key=lambda k: (n_degree[k], node_index[k]))
                if n_degree[p] > 0:
                    cp /= 2.0
            if (e_nk / density / (nn + 1)) < cp:
                break  # no good node found; cluster is finished

            # otherwise, add the node to the cluster
            cluster.add(p)
            nn, ne = (nn + 1), (ne + e_nk)

            cluster_degrees[p] = e_nk
            for n in data[p] & cluster:
                cluster_degrees[n] += 1

            frontier = set.union(*((data[n] - cluster) & unvisited for n in cluster))

        unvisited -= cluster
        res.append(list(cluster))

        num_clusters += 1
    return res


def __dpclus_w(g, d_threshold, cp_threshold, weights):
    res = []
    data = defaultdict(set)  # node id => neighboring node ids
    global_edges = defaultdict(dict)

    # to match the original DPClus output, we need to keep track of the order in
    # which nodes were read from the file, in order to maintain cluster order when
    # breaking ties in our priority ranking (values are decreasing, i.e. negative)
    node_index = dict()
    b_list = []  # the original algorithm indexes column b after column a... >:-(

    # read in graph
    for a, b, dt in g.edges(data=True):

        w = dt[weights]
        data[a].add(b)
        data[b].add(a)
        global_edges[a][b] = w
        global_edges[b][a] = w
        node_index.setdefault(a, -len(node_index))
        b_list.append(b)
    for b in b_list:
        node_index.setdefault(b, -len(node_index))

    unvisited = set(data)

    while unvisited:
        # get highest degree node
        # DIFFERENT: degree of node is sum of edge weights
        seed = max(
            unvisited,
            key=lambda k: (
                sum(global_edges[n][k] for n in data[k] & unvisited),
                node_index[k],
            ),
        )
        frontier = data[seed] & unvisited
        if not frontier:
            break  # no connections left to analyze

        edges, weights = defaultdict(lambda: defaultdict(int)), defaultdict(int)
        for a, b in combinations(unvisited, 2):
            if b not in data[a]:
                continue
            e = len(data[a] & data[b] & unvisited)
            edges[a][b], edges[b][a] = e, e
            weights[a] += e
            weights[b] += e

        max_w, _, node = max((w, node_index[n], n) for n, w in weights.items())
        if max_w > 0:
            seed = node
            frontier = data[seed] & unvisited

        cluster = {seed}
        cluster_degrees = {seed: 0}
        nn, ne = 1, 0.0  # number of nodes, edges in cluster (DIFFERENT: float)

        while frontier:
            # find higest priority node:
            # 0. number of edges between node and cluster nodes
            # 1. sum of edge weights between node and cluster nodes
            # 2. the node's index
            # 3. the node itself
            # DIFFERENT: e_nk = sum of edge weights
            e_nk, _, _, p = max(
                (
                    sum(global_edges[n][c] for c in data[n] & cluster),
                    sum(edges[n][c] for c in cluster),
                    node_index[n],
                    n,
                )
                for n in frontier
            )
            # e_nk,_,_,p = max((len(data[n] & cluster),
            # sum(edges[n][c] for c in cluster), node_index[n], n) for n in frontier)

            density = 2.0 * (ne + e_nk) / (nn * (nn + 1))
            if density < d_threshold:
                break  # adding the node gives too low density; cluster is finished

            # if all nodes only have one connecting edge in cluster, use "fine-tuning"
            # this orders by the number of neighbors in the frontier, minus the
            # connectedness of the attached (cluster) node within the cluster
            cp = cp_threshold
            if e_nk == 1 and len(cluster) > 1:
                n_degree = dict()  # node => fine-tuning parameter
                for n in frontier:
                    for c in cluster:  # find adjacent cluster node
                        if n in edges[c]:
                            break
                    n_degree[n] = len(data[n] & frontier) - cluster_degrees[c]
                p = max(frontier, key=lambda k: (n_degree[k], node_index[k]))
                if n_degree[p] > 0:
                    cp /= 2.0
            if (e_nk / density / (nn + 1)) < cp:
                break  # no good node found; cluster is finished

            # otherwise, add the node to the cluster
            cluster.add(p)
            nn, ne = (nn + 1), (ne + e_nk)

            cluster_degrees[p] = e_nk
            for n in data[p] & cluster:
                cluster_degrees[n] += 1

            frontier = set.union(*((data[n] - cluster) & unvisited for n in cluster))

        if len(cluster) > 1:
            # add overlapping nodes
            # frontier[2] stores our fine-tuning parameter, in this case
            frontier_nodes = set.union(*(data[c] for c in cluster)) - cluster
            # DIFFERENT: e_nk = sum of edge weights
            frontier = sorted(
                [
                    sum(global_edges[n][c] for c in data[n] & cluster),
                    sum(edges[n][c] for c in cluster),
                    0,
                    node_index[n],
                    n,
                ]
                for n in frontier_nodes
            )
            # "fine-tuning"
            fine_tuning = False
            if frontier and frontier[-1][0] == 1:
                for n in frontier:
                    for c in cluster:  # find adjacent cluster node
                        if n[4] in edges[c]:
                            break
                    n[2] = len(data[n[4]] & frontier_nodes) - cluster_degrees[c]
                frontier.sort(key=lambda k: (k[2], k[3]))
                fine_tuning = True
            # iterate through visited neighbors and update accordingly
            while frontier:
                e_nk, _, w, _, p = frontier.pop()
                cp = cp_threshold / 2.0 if fine_tuning and w > 0 else cp_threshold

                density = 2.0 * (ne + e_nk) / (nn * (nn + 1))
                if density < d_threshold or (e_nk / density / (nn + 1)) < cp:
                    continue

                # add node to the cluster
                cluster.add(p)
                nn, ne = (nn + 1), (ne + e_nk)

                cluster_degrees[p] = e_nk
                for n in data[p] & cluster:
                    cluster_degrees[n] += 1

                # update E_nk for the other nodes on the frontier
                for n in frontier:
                    if p in edges[n[4]]:
                        n[0] += 1

        unvisited -= cluster
        if len(cluster) > 1:
            res.append(list(cluster))

    return res
