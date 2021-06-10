# -*- coding: utf-8 -*-
#
#    Copyright (C) 2018 by
#    Thomas Bonald <thomas.bonald@telecom-paristech.fr>
#    Bertrand Charpentier <bertrand.charpentier@live.fr>
#    All rights reserved.
#    BSD license.

import numpy as np
import networkx as nx


def paris(G, copy_graph=True):
    n = G.number_of_nodes()
    if copy_graph:
        F = G.copy()
    else:
        F = G

    # index nodes from 0 to n - 1
    if set(F.nodes()) != set(range(n)):
        F = nx.convert_node_labels_to_integers(F)

    # node weights
    w = {u: 0 for u in range(n)}
    wtot = 0
    for (u, v) in F.edges():
        if "weight" not in F[u][v]:
            F[u][v]["weight"] = 1
        weight = F[u][v]["weight"]
        w[u] += weight
        w[v] += weight
        wtot += weight
        if u != v:
            wtot += weight

    # cluster sizes
    s = {u: 1 for u in range(n)}

    # connected components
    cc = []

    # dendrogram as list of merges
    D = []

    # cluster index
    u = n
    while n > 0:
        # nearest-neighbor chain
        chain = [list(F.nodes())[0]]
        while len(chain) != 0:
            a = chain.pop()
            # nearest neighbor
            dmin = float("inf")
            b = -1
            for v in F.neighbors(a):
                if v != a:
                    d = w[v] * w[a] / float(F[a][v]["weight"]) / float(wtot)
                    if d < dmin:
                        b = v
                        dmin = d
                    elif d == dmin:
                        b = min(b, v)
            d = dmin
            if len(chain) != 0:
                c = chain.pop()
                if b == c:
                    # merge a,b
                    D.append([a, b, d, s[a] + s[b]])
                    # update graph
                    F.add_node(u)
                    neighbors_a = list(F.neighbors(a))
                    neighbors_b = list(F.neighbors(b))
                    for v in neighbors_a:
                        F.add_edge(u, v, weight=F[a][v]["weight"])
                    for v in neighbors_b:
                        if F.has_edge(u, v):
                            F[u][v]["weight"] += F[b][v]["weight"]
                        else:
                            F.add_edge(u, v, weight=F[b][v]["weight"])
                    F.remove_node(a)
                    F.remove_node(b)
                    n -= 1
                    # update weight and size
                    w[u] = w.pop(a) + w.pop(b)
                    s[u] = s.pop(a) + s.pop(b)
                    # change cluster index
                    u += 1
                else:
                    chain.append(c)
                    chain.append(a)
                    chain.append(b)
            elif b >= 0:
                chain.append(a)
                chain.append(b)
            else:
                # remove the connected component
                cc.append((a, s[a]))
                F.remove_node(a)
                w.pop(a)
                s.pop(a)
                n -= 1

    # add connected components to the dendrogram
    a, s = cc.pop()
    for b, t in cc:
        s += t
        D.append([a, b, float("inf"), s])
        a = u
        u += 1

    return __reorder_dendrogram(np.array(D))


def __reorder_dendrogram(D):
    n = np.shape(D)[0] + 1
    order = np.zeros((2, n - 1), float)
    order[0] = range(n - 1)
    order[1] = np.array(D)[:, 2]
    index = np.lexsort(order)
    nindex = {i: i for i in range(n)}
    nindex.update({n + index[t]: n + t for t in range(n - 1)})
    return np.array(
        [
            [nindex[int(D[t][0])], nindex[int(D[t][1])], D[t][2], D[t][3]]
            for t in range(n - 1)
        ]
    )[index, :]


# Rank clusterings at every level of the dendrogram
def __rank_clustering(D):
    logdist = np.log(D[:, 2])
    delta = logdist[1:] - logdist[:-1]
    return np.argsort(-delta[len(delta) // 2 :]) + 1 + len(delta) // 2


# Select the k-th best clustering
def paris_best_clustering(D, k=0):
    return __select_clustering(D, __rank_clustering(D)[k])


# Select the clustering after k merges
def __select_clustering(D, k):
    n = np.shape(D)[0] + 1
    k = min(k, n - 1)
    cluster = {i: [i] for i in range(n)}
    for t in range(k):
        cluster[n + t] = cluster.pop(int(D[t][0])) + cluster.pop(int(D[t][1]))
    return sorted(cluster.values(), key=len, reverse=True)
