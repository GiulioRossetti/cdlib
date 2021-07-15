# Author: True Price <jtprice@cs.unc.edu>

from math import log
from collections import defaultdict
import functools


def graphentropy(g, weight=None):
    if weight is None:
        return __graph_entropy(g)
    else:
        return __graph_entropy_w(g, weight)


# define entropy of a vertex in terms of an associated cluster
def __entropy(c, v, data):
    neighbors = data[v]
    inner = len(c & neighbors) / float(len(neighbors))  # p(inner links)
    return (
        0
        if inner == 0 or inner == 1
        else -inner * log(inner, 2) - (1 - inner) * log(1 - inner, 2)
    )


def __entropy_w(c, v, degree, data):
    deg = degree[v]
    if deg == 0:  # if node degree is zero, do nothing
        return 0

    neighbors = data[v]
    # calculate p(inner links)
    inner = sum(w for n, w in neighbors.items() if n in c) / deg
    return (
        0
        if inner <= 0.0 or inner >= 1.0
        else -inner * log(inner, 2) - (1 - inner) * log(1 - inner, 2)
    )


def __graph_entropy(g):
    data = defaultdict(set)  # node id => neighboring node ids

    # read in graph
    for a, b in g.edges():
        data[a].add(b)
        data[b].add(a)

    candidates = set(data)
    clusters = []
    while candidates:
        v = candidates.pop()  # select a random vertex
        cluster = data[v].copy()  # add neighbors to cluster
        cluster.add(v)
        entropies = dict((x, __entropy(cluster, x, data)) for x in data)

        # step 2: try removing neighbors to minimize entropy
        for n in list(cluster):
            if n == v:
                continue  # don't remove our seed, obviously
            new_c = cluster.copy()
            new_c.remove(n)
            new_e = dict((x, __entropy(new_c, x, data)) for x in data[n])
            # if removing the neighbor decreases new entropy (for the node and
            # all its neighbors), then do so
            if sum(new_e.values()) < sum(entropies[x] for x in data[n]):
                cluster = new_c
                entropies.update(new_e)

        # boundary candidates
        c = functools.reduce(lambda a, b: a | b, (data[x] for x in cluster)) - cluster
        while c:
            n = c.pop()
            new_c = cluster.copy()
            new_c.add(n)
            new_e = dict((x, __entropy(new_c, x, data)) for x in data[n])
            if sum(new_e.values()) < sum(entropies[x] for x in data[n]):
                cluster = new_c
                entropies.update(new_e)
                c &= data[n] - cluster

        # remove the elements of the cluster from our candidate set; add cluster
        candidates -= cluster
        clusters.append(list(cluster))

    return clusters


def __graph_entropy_w(g, wlabel):
    data = defaultdict(dict)  # node id => neighboring node id => edge weight
    degree = defaultdict(float)  # weighted node degrees

    # read in graph
    for a, b, dt in g.edges(data=True):
        w = dt[wlabel]
        data[a][b] = w
        data[b][a] = w
        degree[a] += w
        degree[b] += w

    candidates = set(data)
    clusters = []
    while candidates:
        v = candidates.pop()  # select a random vertex
        cluster = set(data[v])  # add neighbors to cluster
        cluster.add(v)
        entropies = dict((x, __entropy_w(cluster, x, degree, data)) for x in data)

        # step 2: try removing neighbors to minimize entropy
        for n in list(cluster):
            if n == v:
                continue  # don't remove our seed, obviously
            new_c = cluster.copy()
            new_c.remove(n)
            new_e = dict((x, __entropy_w(new_c, x, degree, data)) for x in data[n])
            # if removing the neighbor decreases new entropy (for the node and
            # all its neighbors), then do so
            if sum(new_e.values()) < sum(entropies[x] for x in data[n]):
                cluster = new_c
                entropies.update(new_e)

        # boundary candidates
        c = (
            functools.reduce(lambda a, b: a | b, (set(data[x]) for x in cluster))
            - cluster
        )
        while c:
            n = c.pop()
            new_c = cluster.copy()
            new_c.add(n)
            new_e = dict((x, __entropy_w(new_c, x, degree, data)) for x in data[n])
            if sum(new_e.values()) < sum(entropies[x] for x in data[n]):
                cluster = new_c
                entropies.update(new_e)
                c &= set(data[n]) - cluster

        # remove the elements of the cluster from our candidate set; add cluster
        candidates -= cluster

        clusters.append(list(cluster))
    return clusters
