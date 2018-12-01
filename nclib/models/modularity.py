import community as louvain_modularity
import leidenalg
import igraph as ig
from collections import defaultdict


def louvain(g, weight='weight', resolution=1., randomize=False):

    coms = louvain_modularity.best_partition(g, weight=weight, resolution=resolution, randomize=randomize)

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in coms.items():
        coms_to_node[c].append(n)

    coms_louvain = [tuple(c) for c in coms_to_node.values()]
    return coms_louvain


def leiden(g):
    gi = ig.Graph(directed=False)
    gi.add_vertices(list(g.nodes()))
    gi.add_edges(list(g.edges()))

    part = leidenalg.find_partition(gi, leidenalg.ModularityVertexPartition)
    coms = [gi.vs[x]['name'] for x in part]
    return coms

