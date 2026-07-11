from __future__ import annotations

from typing import Iterable

import igraph as ig
import networkx as nx
import numpy as np

from hidef import hidef_finder


def _nx_to_igraph(graph: nx.Graph) -> tuple[ig.Graph, list]:
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.vs["name"] = [str(n) for n in nodes]
    edges = [(index[u], index[v]) for u, v in graph.edges()]
    if edges:
        g.add_edges(edges)
    return g, nodes


class _SequentialPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def hidef_communities(
    graph: nx.Graph,
    minres: float = 0.01,
    maxres: float = 10.0,
    sample: float = 1.0,
    jaccard: float = 0.75,
    alg: str = "leiden",
    density: float = 0.1,
    neighbors: int = 10,
    k: int = 5,
    f: float = 1.0,
    p: int = 100,
    numthreads: int = 1,
) -> list[list]:
    ig_graph, nodes = _nx_to_igraph(graph)
    original_pool = hidef_finder.mp.Pool
    hidef_finder.mp.Pool = _SequentialPool
    try:
        cluster_graph = hidef_finder.run(
            [ig_graph],
            jaccard=jaccard,
            sample=sample,
            minres=minres,
            maxres=maxres,
            alg=alg,
            density=density,
            neighbors=neighbors,
            numthreads=numthreads,
        )
        consensus = hidef_finder.consensus(cluster_graph, k=k, f=f, p=p)
    finally:
        hidef_finder.mp.Pool = original_pool
    communities = []
    for membership, _persistence in consensus:
        nodes_in_comm = [nodes[i] for i in np.where(membership)[0]]
        if nodes_in_comm:
            communities.append(nodes_in_comm)
    return communities
