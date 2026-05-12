"""
Pure-Python reference implementation of the Highway overlapping community
detection algorithm for CDlib.

This module is intended to live at:

    cdlib/algorithms/internal/Highway.py

The public CDlib wrapper should be added in:

    cdlib/algorithms/overlapping_partition.py

and should call:

    highway_nx(G, ...)

The implementation below keeps only algorithmic logic. It intentionally removes
the C++ command-line interface, file I/O, tracing CSV writers, manifest writers,
timers, and filesystem dependencies.

The output format follows CDlib internal implementations such as SLPA_nx.py:
a list of communities, where each community is a list of original NetworkX nodes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Dict, Hashable, Iterable, List, MutableMapping, Sequence, Tuple

import networkx as nx


Node = Hashable
MembershipRow = List[Tuple[int, float]]


@dataclass
class HighwayBuildConfig:
    """Configuration for sparse Highway backbone construction."""

    top_r: int = 3
    mod_jaccard_alpha: float = 0.70
    ensure_min1_per_node: bool = True
    symmetrize: bool = True


@dataclass
class PropConfig:
    """Configuration for anchor-membership propagation."""

    top_r: int = 3
    T: int = 10
    damping: float = 0.90
    eta_leak: float = 0.0
    tau: float = 0.85
    eps: float = 1e-12


@dataclass
class LocalRefineConfig:
    """
    Lightweight Python version of the anchor-preserving pattern refinement.

    This is intentionally conservative for CDlib: it keeps the same high-level
    idea as the C++ implementation but avoids trace files and implementation
    details that are mainly for experiments.
    """

    enable_pattern_refinement: bool = True
    confidence_self_fraction_weight: float = 0.50
    confidence_low_entropy_weight: float = 0.50
    pattern_confidence_floor: float = 0.05
    pattern_confidence_ceiling: float = 0.95
    update_strength: float = 0.45
    node_mode_power: float = 2.00
    pattern_target_mix: float = 0.65
    target_sharpen_gamma: float = 1.00
    min_abs_mass_to_keep: float = 1e-12
    renormalize: bool = True


def _safe_int(value: int, lo: int = 1) -> int:
    return max(lo, int(value))


def _clamp(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, float(value)))


def _normalize(dist: MutableMapping[int, float]) -> Dict[int, float]:
    total = sum(v for v in dist.values() if v > 0.0)
    if total <= 0.0:
        return {}
    return {k: float(v) / total for k, v in dist.items() if v > 0.0}


def _sharpen(dist: MutableMapping[int, float], gamma: float) -> Dict[int, float]:
    gamma = max(1.0, float(gamma))
    if not dist:
        return {}
    if abs(gamma - 1.0) <= 1e-12:
        return _normalize(dist)
    return _normalize({k: max(0.0, v) ** gamma for k, v in dist.items()})


def _topk_items(dist: MutableMapping[int, float], k: int) -> MembershipRow:
    k = _safe_int(k)
    items = [(cid, float(val)) for cid, val in dist.items() if cid >= 0 and val > 0.0]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:k]


def _stable_softmax(items: MembershipRow, tau: float, eps: float = 1e-12) -> MembershipRow:
    if not items:
        return []

    tau = max(float(tau), eps)
    max_value = max(v for _, v in items)

    exps = []
    total = 0.0
    for cid, value in items:
        ev = exp((value - max_value) / tau)
        exps.append((cid, ev))
        total += ev

    if total <= eps:
        mass = 1.0 / len(items)
        return [(cid, mass) for cid, _ in items]

    return [(cid, ev / total) for cid, ev in exps]


def _relabel_graph_to_contiguous(
    g: nx.Graph,
) -> Tuple[List[Node], Dict[Node, int], List[List[int]]]:
    """
    Convert a NetworkX graph with arbitrary node labels into contiguous ids.

    Returns:
        nodes: original labels by contiguous id
        node_to_id: original label -> contiguous id
        adj: sorted adjacency lists using contiguous ids
    """

    if g.is_directed():
        g = g.to_undirected()

    nodes = list(g.nodes())
    node_to_id = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    adj_sets = [set() for _ in range(n)]
    for u, v in g.edges():
        if u == v:
            continue
        if u not in node_to_id or v not in node_to_id:
            continue
        a = node_to_id[u]
        b = node_to_id[v]
        if a == b:
            continue
        adj_sets[a].add(b)
        adj_sets[b].add(a)

    adj = [sorted(neigh) for neigh in adj_sets]
    return nodes, node_to_id, adj


def _degrees(adj: Sequence[Sequence[int]]) -> List[int]:
    return [len(neigh) for neigh in adj]


def _intersection_size_sorted(a: Sequence[int], b: Sequence[int]) -> int:
    i = 0
    j = 0
    count = 0

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            count += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    return count


def _jaccard_score(adj: Sequence[Sequence[int]], deg: Sequence[int], u: int, v: int) -> float:
    du = max(0, deg[u])
    dv = max(0, deg[v])
    if du == 0 and dv == 0:
        return 0.0

    inter = _intersection_size_sorted(adj[u], adj[v])
    union = du + dv - inter
    if union <= 0:
        return 0.0

    return float(inter) / float(union)


def _build_highway_adjacency(
    full_adj: Sequence[Sequence[int]],
    cfg: HighwayBuildConfig,
) -> List[List[int]]:
    """
    Build sparse Highway backbone using the mod-jaccard hybrid score.

    For each source node v, retain the top-r neighbors according to:

        score(u, v) = alpha * s_mod(u, v) + (1 - alpha) * s_jac(u, v)

    where s_mod(u, v) = 1 - deg(u) deg(v) / (2m) for observed edges.
    """

    n = len(full_adj)
    if n == 0:
        return []

    deg = _degrees(full_adj)
    m_undirected = max(1.0, 0.5 * sum(deg))
    r = _safe_int(cfg.top_r)
    alpha = _clamp(cfg.mod_jaccard_alpha, 0.0, 1.0)

    picked: List[List[int]] = [[] for _ in range(n)]

    for v in range(n):
        neigh = list(full_adj[v])
        if not neigh:
            continue

        scored = []
        dv = float(max(0, deg[v]))
        for u in neigh:
            du = float(max(0, deg[u]))
            modularity_score = 1.0 - (du * dv) / (2.0 * m_undirected)
            jaccard_score = _jaccard_score(full_adj, deg, u, v)
            hybrid_score = alpha * modularity_score + (1.0 - alpha) * jaccard_score
            scored.append((u, hybrid_score))

        scored.sort(key=lambda x: (-x[1], x[0]))
        chosen = [u for u, _ in scored[: min(r, len(scored))]]

        if cfg.ensure_min1_per_node and not chosen and scored:
            chosen = [scored[0][0]]

        picked[v] = chosen

    if not cfg.symmetrize:
        return [sorted(set(neigh)) for neigh in picked]

    undirected_edges = set()
    for v, neigh in enumerate(picked):
        for u in neigh:
            if u < 0 or u >= n or u == v:
                continue
            a = min(u, v)
            b = max(u, v)
            undirected_edges.add((a, b))

    h_sets = [set() for _ in range(n)]
    for a, b in undirected_edges:
        h_sets[a].add(b)
        h_sets[b].add(a)

    return [sorted(neigh) for neigh in h_sets]


def _select_anchors_greedy_dedup(
    full_adj: Sequence[Sequence[int]],
    max_anchors: int,
) -> List[int]:
    """
    Greedy degree-cover anchor selection.

    Nodes are sorted by degree descending. A node becomes an anchor only if it
    has not already been covered by a previously selected anchor or one of that
    anchor's full-graph neighbors.
    """

    n = len(full_adj)
    if n <= 0:
        return []

    max_anchors = _safe_int(max_anchors)
    deg = _degrees(full_adj)
    order = list(range(n))
    order.sort(key=lambda u: (-deg[u], u))

    covered = [False] * n
    anchors: List[int] = []

    for u in order:
        if len(anchors) >= max_anchors:
            break
        if covered[u]:
            continue

        anchors.append(u)
        covered[u] = True

        for v in full_adj[u]:
            if 0 <= v < n:
                covered[v] = True

    if not anchors:
        anchors.append(order[0])

    return anchors


def _initial_state(n: int, r: int, anchors: Sequence[int]) -> List[MembershipRow]:
    r = _safe_int(r)
    state: List[MembershipRow] = [[] for _ in range(n)]

    for cid, v in enumerate(anchors):
        if 0 <= v < n:
            state[v] = [(cid, 1.0)]

    return state


def _propagate_anchor_memberships(
    highway_adj: Sequence[Sequence[int]],
    full_adj: Sequence[Sequence[int]],
    anchors: Sequence[int],
    cfg: PropConfig,
) -> List[MembershipRow]:
    """
    Neighbor-only top-r anchor-membership propagation.

    A node is updated from its neighbors' previous anchor memberships. The node's
    own previous state is not directly reinforced during the main aggregation.
    """

    n = len(highway_adj)
    r = _safe_int(cfg.top_r)
    k = len(anchors)

    state = _initial_state(n, r, anchors)
    if n <= 0 or k <= 0:
        return state

    deg_h = [max(1, d) for d in _degrees(highway_adj)]
    deg_f = [max(1, d) for d in _degrees(full_adj)]

    T = max(0, int(cfg.T))
    damping = _clamp(cfg.damping, 0.0, 1.0)
    eta_leak = max(0.0, float(cfg.eta_leak))

    for _ in range(T):
        next_state: List[MembershipRow] = [[] for _ in range(n)]

        for v in range(n):
            acc: Dict[int, float] = defaultdict(float)

            for u in highway_adj[v]:
                if u < 0 or u >= n:
                    continue
                weight = 1.0 / sqrt(float(deg_h[u] * deg_h[v]))

                for cid, prob in state[u]:
                    if cid >= 0 and prob > 0.0:
                        acc[cid] += prob * weight

            if eta_leak > 0.0:
                for u in full_adj[v]:
                    if u < 0 or u >= n:
                        continue
                    weight = eta_leak / sqrt(float(deg_f[u] * deg_f[v]))

                    for cid, prob in state[u]:
                        if cid >= 0 and prob > 0.0:
                            acc[cid] += prob * weight

            if not acc:
                next_state[v] = list(state[v])
                continue

            top = _topk_items(acc, min(r, k))
            next_state[v] = _stable_softmax(top, cfg.tau, cfg.eps)

        if damping < 1.0:
            mixed_state: List[MembershipRow] = [[] for _ in range(n)]
            for v in range(n):
                mixed: Dict[int, float] = defaultdict(float)

                for cid, prob in next_state[v]:
                    mixed[cid] += damping * prob

                for cid, prob in state[v]:
                    mixed[cid] += (1.0 - damping) * prob

                top = _topk_items(mixed, min(r, k))
                mixed_state[v] = _normalize(dict(top))
                mixed_state[v] = _topk_items(mixed_state[v], min(r, k))

            state = mixed_state
        else:
            state = next_state

    return state


def _support_key(row: MembershipRow) -> Tuple[int, ...]:
    ids = sorted({cid for cid, prob in row if cid >= 0 and prob > 0.0})
    return tuple(ids)


def _normalized_entropy(counts: MutableMapping[int, int]) -> float:
    positive = [v for v in counts.values() if v > 0]
    total = float(sum(positive))
    k = len(positive)

    if total <= 0.0 or k <= 1:
        return 0.0

    h = 0.0
    for count in positive:
        p = float(count) / total
        h -= p * log(p + 1e-12)

    return h / log(float(k))


def _same_pattern_neighbor_ratio(
    adj: Sequence[Sequence[int]],
    node_pid: Sequence[int],
    v: int,
) -> float:
    if v < 0 or v >= len(adj) or v >= len(node_pid):
        return 0.0

    pid = node_pid[v]
    if pid < 0:
        return 0.0

    total = 0
    same = 0

    for u in adj[v]:
        if u < 0 or u >= len(node_pid):
            continue
        qid = node_pid[u]
        if qid < 0:
            continue
        total += 1
        if qid == pid:
            same += 1

    if total <= 0:
        return 0.0

    return float(same) / float(total)


def _neighbor_anchor_consensus(
    state: Sequence[MembershipRow],
    adj: Sequence[Sequence[int]],
    v: int,
) -> Dict[int, float]:
    dist: Dict[int, float] = defaultdict(float)
    n = len(state)

    if v < 0 or v >= len(adj) or v >= n:
        return {}

    for u in adj[v]:
        if u < 0 or u >= n:
            continue
        for cid, prob in state[u]:
            if cid >= 0 and prob > 0.0:
                dist[cid] += prob

    return _normalize(dist)


def _refine_anchor_preserving_patterns(
    state: Sequence[MembershipRow],
    full_adj: Sequence[Sequence[int]],
    backbone_adj: Sequence[Sequence[int]],
    cfg: LocalRefineConfig,
    top_r: int,
) -> List[MembershipRow]:
    """
    First Python version of the pattern-neighborhood refinement.

    The refinement is anchor-preserving: it only updates distributions over
    anchor ids. It does not create new labels unrelated to the selected anchors.
    """

    if not cfg.enable_pattern_refinement:
        return [list(row) for row in state]

    n = len(state)
    r = _safe_int(top_r)

    if n <= 0:
        return []

    pattern_to_pid: Dict[Tuple[int, ...], int] = {}
    patterns: List[Dict[str, object]] = []
    node_pid = [-1] * n

    for v, row in enumerate(state):
        key = _support_key(row)
        if key not in pattern_to_pid:
            pattern_to_pid[key] = len(patterns)
            patterns.append(
                {
                    "key": key,
                    "nodes": [],
                    "internal_edges": 0,
                    "external_edges": 0,
                    "ext_counts": defaultdict(int),
                    "target": defaultdict(float),
                }
            )

        pid = pattern_to_pid[key]
        node_pid[v] = pid
        patterns[pid]["nodes"].append(v)

    if not patterns:
        return [list(row) for row in state]

    # Estimate pattern structure and target anchor distributions.
    for u in range(n):
        pu = node_pid[u]
        if pu < 0:
            continue

        for v in full_adj[u]:
            if v < 0 or v >= n:
                continue
            pv = node_pid[v]
            if pv < 0:
                continue

            if pu == pv:
                patterns[pu]["internal_edges"] += 1
            else:
                patterns[pu]["external_edges"] += 1
                patterns[pu]["ext_counts"][pv] += 1

        for cid, prob in state[u]:
            if cid >= 0 and prob > 0.0:
                patterns[pu]["target"][cid] += prob

    pattern_confidence = [0.0] * len(patterns)
    pattern_target: List[Dict[int, float]] = []

    for pid, pat in enumerate(patterns):
        internal_edges = int(pat["internal_edges"])
        external_edges = int(pat["external_edges"])
        total_edges = internal_edges + external_edges

        self_fraction = 0.0 if total_edges <= 0 else float(internal_edges) / total_edges
        neighbor_entropy = _normalized_entropy(pat["ext_counts"])

        w_self = max(0.0, cfg.confidence_self_fraction_weight)
        w_entropy = max(0.0, cfg.confidence_low_entropy_weight)
        w_sum = w_self + w_entropy

        if w_sum <= 1e-20:
            conf = 0.0
        else:
            conf = (
                w_self * _clamp(self_fraction, 0.0, 1.0)
                + w_entropy * _clamp(1.0 - neighbor_entropy, 0.0, 1.0)
            ) / w_sum

        conf = _clamp(conf, cfg.pattern_confidence_floor, cfg.pattern_confidence_ceiling)
        pattern_confidence[pid] = conf

        target = _normalize(pat["target"])
        target = _sharpen(target, cfg.target_sharpen_gamma)
        pattern_target.append(target)

    refined: List[MembershipRow] = [list(row) for row in state]

    for v in range(n):
        pid = node_pid[v]
        if pid < 0:
            continue

        own = _normalize(dict(state[v]))
        if not own:
            continue

        neigh = _neighbor_anchor_consensus(state, backbone_adj, v)
        target = pattern_target[pid]

        local_same = _same_pattern_neighbor_ratio(backbone_adj, node_pid, v)
        local_mode = _clamp(local_same, 0.0, 1.0) ** max(1.0, cfg.node_mode_power)

        consensus: Dict[int, float] = defaultdict(float)
        for cid, prob in target.items():
            consensus[cid] += cfg.pattern_target_mix * prob

        for cid, prob in neigh.items():
            consensus[cid] += (1.0 - cfg.pattern_target_mix) * prob

        consensus = _normalize(consensus)
        if not consensus:
            continue

        strength = _clamp(cfg.update_strength * pattern_confidence[pid] * local_mode, 0.0, 1.0)

        updated: Dict[int, float] = defaultdict(float)
        for cid, prob in own.items():
            updated[cid] += (1.0 - strength) * prob

        for cid, prob in consensus.items():
            updated[cid] += strength * prob

        items = [
            (cid, prob)
            for cid, prob in updated.items()
            if cid >= 0 and prob > cfg.min_abs_mass_to_keep
        ]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[:r]

        if cfg.renormalize:
            refined[v] = _topk_items(_normalize(dict(items)), r)
        else:
            refined[v] = items

    return refined


def _decode_state_labels(
    state: Sequence[MembershipRow],
    theta: float,
    max_memberships: int,
) -> List[List[int]]:
    """
    Decode node-level anchor labels.

    This follows the C++ main decoding behavior: keep labels with probability
    above theta; if none survive, keep the top label as a fallback.
    """

    keep_max = _safe_int(max_memberships)
    theta = float(theta)
    memberships: List[List[int]] = []

    for row in state:
        items = [(cid, prob) for cid, prob in row if cid >= 0 and prob > 0.0]
        items.sort(key=lambda x: (-x[1], x[0]))

        labels = []
        for cid, prob in items:
            if len(labels) >= keep_max:
                break
            if prob >= theta:
                labels.append(cid)

        if not labels and items:
            labels.append(items[0][0])

        memberships.append(labels)

    return memberships


def _build_communities_from_memberships(
    node_memberships: Sequence[Sequence[int]],
    nodes: Sequence[Node],
    min_community_size: int = 1,
) -> List[List[Node]]:
    """
    Convert node-level anchor memberships into CDlib-style communities.

    This function intentionally keeps duplicate, nested, and graph-scale
    communities. Highway's decoding stage treats each surviving anchor label as
    an output community. Therefore, if two anchor labels induce the same node
    set, both are preserved instead of being merged in post-processing.

    Only communities smaller than min_community_size are removed.
    """

    comm_to_nodes: Dict[int, List[Node]] = defaultdict(list)

    for vid, labels in enumerate(node_memberships):
        for cid in labels:
            comm_to_nodes[cid].append(nodes[vid])

    communities = []
    min_community_size = _safe_int(min_community_size)

    for cid in sorted(comm_to_nodes):
        comm = comm_to_nodes[cid]
        if len(comm) >= min_community_size:
            communities.append(comm)

    return communities


def highway_nx(
    G: nx.Graph,
    highway_top_r: int = 3,
    mod_jaccard_alpha: float = 0.70,
    ensure_min1_per_node: bool = True,
    symmetrize: bool = True,
    max_anchors: int | None = None,
    prop_top_r: int = 3,
    prop_T: int = 10,
    prop_damping: float = 0.90,
    prop_eta_leak: float = 0.0,
    prop_tau: float = 0.85,
    enable_pattern_refinement: bool = True,
    local_update_strength: float = 0.45,
    local_pattern_target_mix: float = 0.65,
    local_node_mode_power: float = 2.00,
    decode_theta: float = 0.30,
    max_memberships: int = 3,
    min_community_size: int = 1,
) -> List[List[Node]]:
    """
    Run the Highway overlapping community detection algorithm on a NetworkX graph.

    Parameters are named to match the current C++ implementation where possible.

    Returns:
        A list of overlapping communities, where each community is represented
        as a list of original NetworkX node labels.
    """

    if G is None:
        raise ValueError("G must be a NetworkX graph.")

    nodes, _, full_adj = _relabel_graph_to_contiguous(G)
    n = len(nodes)

    if n == 0:
        return []

    hcfg = HighwayBuildConfig(
        top_r=highway_top_r,
        mod_jaccard_alpha=mod_jaccard_alpha,
        ensure_min1_per_node=ensure_min1_per_node,
        symmetrize=symmetrize,
    )
    pcfg = PropConfig(
        top_r=prop_top_r,
        T=prop_T,
        damping=prop_damping,
        eta_leak=prop_eta_leak,
        tau=prop_tau,
    )
    rcfg = LocalRefineConfig(
        enable_pattern_refinement=enable_pattern_refinement,
        update_strength=local_update_strength,
        pattern_target_mix=local_pattern_target_mix,
        node_mode_power=local_node_mode_power,
    )

    highway_adj = _build_highway_adjacency(full_adj, hcfg)

    if max_anchors is None or max_anchors < 0:
        max_anchors = max(8, min(30, n // 5))

    anchors = _select_anchors_greedy_dedup(full_adj, max_anchors)

    state = _propagate_anchor_memberships(
        highway_adj=highway_adj,
        full_adj=full_adj,
        anchors=anchors,
        cfg=pcfg,
    )

    state_refined = _refine_anchor_preserving_patterns(
        state=state,
        full_adj=full_adj,
        backbone_adj=highway_adj,
        cfg=rcfg,
        top_r=prop_top_r,
    )

    node_memberships = _decode_state_labels(
        state=state_refined,
        theta=decode_theta,
        max_memberships=max_memberships,
    )

    return _build_communities_from_memberships(
        node_memberships=node_memberships,
        nodes=nodes,
        min_community_size=min_community_size,
    )
