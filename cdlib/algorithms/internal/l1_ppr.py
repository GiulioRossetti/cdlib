"""L1-regularized Personalized PageRank local expansion."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy import sparse


def l1_ppr(
    A: sparse.spmatrix,
    seedset: np.ndarray,
    min_comm_size: int,
    max_comm_size: int,
    alpha: float = 0.85,
    epsilon: float = 1e-4,
) -> List[int]:
    """Run the localized push-based APPR expansion."""

    n = A.shape[0]
    if len(seedset) == 0:
        return []

    A_csr = A.tocsr().astype(float)
    degrees = np.asarray(A_csr.sum(axis=1)).ravel()

    p = np.zeros(n)
    r = np.zeros(n)
    r[seedset] = 1.0 / len(seedset)

    queue = {u for u in seedset if degrees[u] > 0 and r[u] >= epsilon * degrees[u]}

    while queue:
        u = queue.pop()
        val = r[u]
        r[u] = 0.0

        p[u] += (1.0 - alpha) * val
        push_val = alpha * val / degrees[u]

        start_idx = A_csr.indptr[u]
        end_idx = A_csr.indptr[u + 1]
        neighbors = A_csr.indices[start_idx:end_idx]
        weights = A_csr.data[start_idx:end_idx]

        for v, w in zip(neighbors, weights):
            if degrees[v] > 0:
                old_rv = r[v]
                r[v] += push_val * w
                if r[v] >= epsilon * degrees[v] and old_rv < epsilon * degrees[v]:
                    queue.add(v)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_p = np.where(degrees > 0, p / degrees, 0.0)

    all_sorted = np.argsort(normalized_p)[::-1]

    def cal_conductance(cluster):
        sub = A_csr[cluster, :][:, cluster]
        cut = float(A_csr[cluster, :].sum() - sub.sum())
        total_vol = float(A_csr.sum())
        cluster_vol = float(A_csr[cluster, :].sum())
        denom = min(cluster_vol, total_vol - cluster_vol)
        if denom <= 0:
            return 1.0
        return float(cut / denom)

    best_cond = 1.1
    best_size = min_comm_size
    max_comm = min(max_comm_size, n)

    for k in range(min_comm_size, max_comm + 1):
        candidate = all_sorted[:k]
        cond = cal_conductance(candidate)
        if cond < best_cond:
            best_cond = cond
            best_size = k

    return list(all_sorted[:best_size])
