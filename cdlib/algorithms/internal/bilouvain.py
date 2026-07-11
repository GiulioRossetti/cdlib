from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sknetwork.clustering import Louvain


def bilouvain_partition(
    graph: nx.Graph,
    weight: str | None = None,
    resolution: float = 1.0,
    random_state: int | None = None,
) -> list[list]:
    nodes = list(graph.nodes())
    matrix = nx.to_scipy_sparse_array(
        graph, nodelist=nodes, weight=weight, format="csr", dtype=float
    )
    matrix = csr_matrix(matrix)
    model = Louvain(resolution=resolution, modularity="dugue", random_state=random_state)
    labels = model.fit_predict(matrix)
    communities = {}
    for node, label in zip(nodes, labels):
        communities.setdefault(int(label), []).append(node)
    return [sorted(comm) for comm in communities.values() if len(comm) > 0]
