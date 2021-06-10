import math
import random
import sys
import operator
from collections import defaultdict


def principled(g, cluster_count):

    clusters = range(cluster_count)
    nodes = list(g.nodes())
    edges = list(g.edges())

    # Do the actual clustering
    k = defaultdict(lambda: defaultdict(lambda: random.random()))
    for n in range(1, 100):
        kappa = defaultdict(lambda: 0)
        k_prime = defaultdict(lambda: defaultdict(lambda: 0))
        for z in clusters:
            for i in nodes:
                kappa[z] += k[i][z]

        for i, j in edges:
            D = 0
            for z in clusters:
                D += (k[i][z] * k[j][z]) / kappa[z]

            for z in clusters:
                q = (k[i][z] * k[j][z]) / (D * kappa[z])
                k_prime[i][z] += q
                k_prime[j][z] += q
        k = k_prime

    # Output per-node community membership
    communities = defaultdict(list)
    allocation_matrix = {}
    for i in nodes:
        expected_colors = k[i].items()
        expected_degree = sum([c[1] for c in expected_colors])
        membership = map(
            lambda x: (x[0], 100 * x[1] / expected_degree), expected_colors
        )
        allocation_matrix[i] = {cluster: weight for cluster, weight in membership}
        communities[
            max(allocation_matrix[i].items(), key=operator.itemgetter(1))[0]
        ].append(i)

    return list(communities.values()), allocation_matrix
