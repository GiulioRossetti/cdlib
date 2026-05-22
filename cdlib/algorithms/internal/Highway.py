"""
C++ backend wrapper for the Highway overlapping community detection algorithm.

This module is intended to live at:

    cdlib/algorithms/internal/Highway.py

The compiled C++ executable is expected at:

    cdlib/algorithms/internal/highway_cpp/build/highway

The public CDlib wrapper in:

    cdlib/algorithms/overlapping_partition.py

should call:

    highway_nx(G, ...)

This module keeps the CDlib-facing API in Python, while the actual Highway
algorithm is executed by the optimized C++ backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Hashable, List, Sequence, Tuple

import networkx as nx


Node = Hashable


def _default_highway_cpp_binary() -> Path:
    """
    Return the default path of the compiled Highway C++ executable.

    Expected layout:
        cdlib/algorithms/internal/highway_cpp/build/highway
    """
    return Path(__file__).resolve().parent / "highway_cpp" / "build" / "highway"


def _write_networkx_edgelist_for_cpp(
    G: nx.Graph,
    path: Path,
) -> Tuple[List[Node], Dict[Node, int]]:
    """
    Write a NetworkX graph as a contiguous integer edge list for the C++ backend.

    Returns:
        nodes: original NetworkX node labels indexed by contiguous integer id.
        node_to_id: mapping from original NetworkX node label to contiguous id.
    """
    if G.is_directed():
        G = G.to_undirected()

    nodes = list(G.nodes())
    node_to_id = {node: i for i, node in enumerate(nodes)}

    with path.open("w", encoding="utf-8") as f:
        for u, v in G.edges():
            if u == v:
                continue
            f.write(f"{node_to_id[u]} {node_to_id[v]}\n")

    return nodes, node_to_id


def _read_cpp_communities_json(
    path: Path,
    nodes: Sequence[Node],
    min_community_size: int = 1,
) -> List[List[Node]]:
    """
    Read communities from the C++ communities.json output.

    The C++ output uses contiguous integer node ids. This function maps them
    back to original NetworkX node labels.
    """
    if not path.exists():
        return []

    min_community_size = max(1, int(min_community_size))

    with path.open("r", encoding="utf-8") as f:
        raw_communities = json.load(f)

    communities: List[List[Node]] = []

    for raw_comm in raw_communities:
        comm = []

        for vid in raw_comm:
            vid = int(vid)
            if 0 <= vid < len(nodes):
                comm.append(nodes[vid])

        if len(comm) >= min_community_size:
            communities.append(comm)

    return communities


def _remove_exact_duplicate_communities(
    communities: List[List[Node]],
    deduplicate_communities: bool = True,
) -> List[List[Node]]:
    """
    Remove exactly duplicated communities.

    Two communities are treated as duplicates if they contain the same node set,
    regardless of node order.
    """
    if not deduplicate_communities:
        return communities

    seen = set()
    deduped: List[List[Node]] = []

    for comm in communities:
        key = tuple(sorted(comm, key=lambda x: str(x)))

        if key not in seen:
            seen.add(key)
            deduped.append(comm)

    return deduped


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
    local_confidence_self_fraction_weight: float = 0.85,
    local_confidence_low_entropy_weight: float = 0.15,
    local_pattern_confidence_floor: float = 0.05,
    local_pattern_confidence_ceiling: float = 1.00,
    local_update_strength: float = 0.50,
    local_node_mode_power: float = 1.50,
    local_pattern_target_mix: float = 0.75,
    local_target_sharpen_gamma: float = 1.20,
    local_min_abs_mass_to_keep: float = 1e-8,
    local_renormalize: bool = True,
    decode_theta: float = 0.30,
    max_memberships: int = 3,
    min_community_size: int = 1,
    deduplicate_communities: bool = True,
) -> List[List[Node]]:
    """
    Run Highway on a NetworkX graph.

    This function exposes the same Python-facing API as the pure-Python Highway
    implementation. In this branch, the computation is delegated internally to
    the compiled C++ executable.

    Args:
        G:
            Input NetworkX graph.
        highway_top_r:
            Number of top highway edges retained per node.
        mod_jaccard_alpha:
            Mixture weight between modularity-style score and Jaccard score.
        ensure_min1_per_node:
            Whether to keep at least one edge per node when possible.
        symmetrize:
            Whether to symmetrize the retained highway edges.
        max_anchors:
            Optional maximum number of anchors.
        prop_top_r:
            Number of propagated memberships retained per node.
        prop_T:
            Number of propagation iterations.
        prop_damping:
            Propagation damping factor.
        prop_eta_leak:
            Leakage parameter in propagation.
        prop_tau:
            Softmax temperature for propagation.
        enable_pattern_refinement:
            Whether to enable anchor-preserving pattern refinement.
        local_confidence_self_fraction_weight:
            Weight for self-fraction confidence.
        local_confidence_low_entropy_weight:
            Weight for low-entropy confidence.
        local_pattern_confidence_floor:
            Minimum pattern confidence.
        local_pattern_confidence_ceiling:
            Maximum pattern confidence.
        local_update_strength:
            Strength of local refinement update.
        local_node_mode_power:
            Power parameter for local node mode.
        local_pattern_target_mix:
            Mixture weight for pattern target.
        local_target_sharpen_gamma:
            Sharpening parameter for local target.
        local_min_abs_mass_to_keep:
            Minimum absolute membership mass to keep.
        local_renormalize:
            Whether to renormalize local memberships.
        decode_theta:
            Decoding threshold.
        max_memberships:
            Maximum number of memberships per node.
        min_community_size:
            Minimum size of returned communities.
        deduplicate_communities:
            If True, remove exact duplicate communities before returning.

    Returns:
        A list of overlapping communities. Each community is represented as a
        list of original NetworkX node labels.
    """
    if G is None:
        raise ValueError("G must be a NetworkX graph.")

    if G.is_directed():
        G = G.to_undirected()

    nodes = list(G.nodes())

    if len(nodes) == 0:
        return []

    if G.number_of_edges() == 0:
        communities = [[node] for node in nodes]
        return _remove_exact_duplicate_communities(
            communities,
            deduplicate_communities=deduplicate_communities,
        )

    binary = _default_highway_cpp_binary()

    if not binary.exists():
        raise FileNotFoundError(
            f"Highway C++ binary not found: {binary}. "
            "Please compile the C++ backend first."
        )

    if not os.access(binary, os.X_OK):
        raise PermissionError(
            f"Highway C++ binary is not executable: {binary}. "
            f"Run: chmod +x {binary}"
        )

    with tempfile.TemporaryDirectory(prefix="highway_cpp_") as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        input_path = tmpdir / "graph.edgelist"
        communities_json_path = tmpdir / "communities.json"

        nodes, _ = _write_networkx_edgelist_for_cpp(G, input_path)

        cmd = [
            str(binary),
            "--input",
            str(input_path),
            "--highway_top_r",
            str(highway_top_r),
            "--ensure_min1",
            "1" if ensure_min1_per_node else "0",
            "--symmetrize",
            "1" if symmetrize else "0",
            "--mod_jaccard_alpha",
            str(mod_jaccard_alpha),
            "--prop_top_r",
            str(prop_top_r),
            "--prop_T",
            str(prop_T),
            "--prop_damping",
            str(prop_damping),
            "--prop_eta_leak",
            str(prop_eta_leak),
            "--prop_tau",
            str(prop_tau),
            "--local_enable_pattern_refinement",
            "1" if enable_pattern_refinement else "0",
            "--local_confidence_self_fraction_weight",
            str(local_confidence_self_fraction_weight),
            "--local_confidence_low_entropy_weight",
            str(local_confidence_low_entropy_weight),
            "--local_pattern_confidence_floor",
            str(local_pattern_confidence_floor),
            "--local_pattern_confidence_ceiling",
            str(local_pattern_confidence_ceiling),
            "--local_update_strength",
            str(local_update_strength),
            "--local_node_mode_power",
            str(local_node_mode_power),
            "--local_pattern_target_mix",
            str(local_pattern_target_mix),
            "--local_target_sharpen_gamma",
            str(local_target_sharpen_gamma),
            "--local_min_abs_mass_to_keep",
            str(local_min_abs_mass_to_keep),
            "--local_renormalize",
            "1" if local_renormalize else "0",
            "--decode_theta",
            str(decode_theta),
            "--max_memberships",
            str(max_memberships),
        ]

        if max_anchors is not None:
            cmd.extend(["--max_anchors", str(max_anchors)])

        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(tmpdir),
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Highway C++ backend failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Working directory: {tmpdir}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        communities = _read_cpp_communities_json(
            communities_json_path,
            nodes,
            min_community_size=min_community_size,
        )

        if not communities:
            produced_files = "\n".join(
                str(p) for p in tmpdir.rglob("*") if p.is_file()
            )

            raise RuntimeError(
                "Highway C++ backend finished, but no communities were parsed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Working directory: {tmpdir}\n"
                f"Expected file: {communities_json_path}\n"
                f"Produced files:\n{produced_files}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        return _remove_exact_duplicate_communities(
            communities,
            deduplicate_communities=deduplicate_communities,
        )