"""Personalized PageRank sweep local expansion."""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


def ppr_sweep(
    A: sparse.spmatrix,
    seedset: np.ndarray,
    min_comm_size: int,
    max_comm_size: int,
    alpha: float = 0.85,
    tol: float = 1e-6,
) -> List[int]:
    """Run a sweep cut over a PPR score vector."""

    n = A.shape[0]
    if len(seedset) == 0:
        return []

    A_csr = A.tocsr().astype(float)
    degrees = np.asarray(A_csr.sum(axis=1)).ravel()

    s = np.zeros(n)
    s[seedset] = 1.0 / len(seedset)

    diag_deg = np.where(degrees > 0, degrees, 1.0)
    D = sparse.diags(diag_deg, format="csr")
    M = D - alpha * A_csr
    b = (1.0 - alpha) * s

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            x, _ = cg(M, b, tol=tol, maxiter=500)
        except TypeError:
            x, _ = cg(M, b, rtol=tol, atol=tol, maxiter=500)

    p = diag_deg * x

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(degrees > 0, p / degrees, 0.0)

    all_sorted = np.argsort(r)[::-1]

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
