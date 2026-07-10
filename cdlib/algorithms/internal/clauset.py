"""Clauset local modularity seed-set expansion."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy import sparse


def clauset_expansion(
    A: sparse.spmatrix,
    seedset: np.ndarray,
    min_comm_size: int,
    max_comm_size: int,
) -> List[int]:
    """Greedily expand a seed set by maximizing Clauset's local modularity."""

    if len(seedset) == 0:
        return []

    adj_dict = {}
    rows, cols = A.nonzero()
    data = A.data
    for i in range(len(rows)):
        u = int(rows[i])
        v = int(cols[i])
        w = float(data[i])
        adj_dict.setdefault(u, {})[v] = w

    def compute_r(c_set: set[int]) -> float:
        boundary = []
        for u in c_set:
            for v in adj_dict.get(u, {}):
                if v not in c_set:
                    boundary.append(u)
                    break

        b_set = set(boundary)
        b_in = 0.0
        b_out = 0.0
        seen_internal = set()

        for u in b_set:
            for v, w in adj_dict.get(u, {}).items():
                if v in c_set:
                    edge = (min(u, v), max(u, v))
                    if edge not in seen_internal:
                        seen_internal.add(edge)
                        b_in += w
                else:
                    b_out += w

        denom = b_in + b_out
        if denom == 0.0:
            return 1.0
        return b_in / denom

    c_set = set(int(s) for s in seedset)
    best_c = list(c_set)
    best_score = -1.0

    if len(c_set) >= min_comm_size:
        best_score = compute_r(c_set)

    while len(c_set) < max_comm_size:
        candidates = set()
        for u in c_set:
            for v in adj_dict.get(u, {}):
                if v not in c_set:
                    candidates.add(v)

        if not candidates:
            break

        best_cand = None
        best_cand_score = -1.0

        for cand in candidates:
            c_temp = c_set | {cand}
            score = compute_r(c_temp)
            if score > best_cand_score:
                best_cand_score = score
                best_cand = cand

        if best_cand is None:
            break

        c_set.add(best_cand)

        if len(c_set) >= min_comm_size and best_cand_score > best_score:
            best_score = best_cand_score
            best_c = list(c_set)

    return best_c
