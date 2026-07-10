from __future__ import annotations

import enum
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import scipy.cluster.hierarchy as sch

from cdlib import NodeClustering
from cdlib.evaluation import modularity_overlap
from cdlib.utils import convert_graph_formats


class GHACLinkageMethod(enum.Enum):
    SINGLE = 1
    COMPLETE = 2
    AVERAGE = 3


def _graph_to_integer_graph(g_original: object) -> nx.Graph:
    graph = convert_graph_formats(g_original, nx.Graph)
    return nx.convert_node_labels_to_integers(graph, label_attribute="old_id")


def _clique_bases(graph: nx.Graph, min_base_size: int) -> list[tuple[int, ...]]:
    bases = [tuple(sorted(clique)) for clique in nx.find_cliques(graph) if len(clique) >= min_base_size]
    if not bases:
        bases = [(node,) for node in graph.nodes()]
    bases.sort(key=lambda base: (len(base), base))
    return bases


def _shortest_path_matrix(graph: nx.Graph, weight: Optional[str] = None) -> np.ndarray:
    nodes = list(graph.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    dist = np.full((len(nodes), len(nodes)), np.inf, dtype=float)
    np.fill_diagonal(dist, 0.0)
    if weight is None:
        lengths = nx.all_pairs_shortest_path_length(graph)
    else:
        lengths = nx.all_pairs_dijkstra_path_length(graph, weight=weight)
    for source, mapping in lengths:
        for target, d in mapping.items():
            dist[index[source], index[target]] = float(d)
    finite = dist[np.isfinite(dist)]
    if finite.size:
        dist[~np.isfinite(dist)] = float(finite.max() + 1.0)
    return dist


class GraphAgglomerativeClusteringClosedTrail:
    def __init__(
        self,
        graph: nx.Graph,
        ct_linkage_method: GHACLinkageMethod,
        ct_distance_matrix: np.ndarray,
        bases: list[tuple[int, ...]],
        weight_attribute: Optional[str] = None,
    ):
        self.graph = graph
        self.m = nx.number_of_edges(self.graph)
        self.degrees = dict(nx.degree(self.graph))
        self.ct_linkage_method = ct_linkage_method
        self.ct_distance_matrix = ct_distance_matrix
        self.bases = bases
        self.weight_attribute = weight_attribute
        self.wt = None
        if weight_attribute is not None and self.graph.number_of_edges() > 0:
            self.wt = sum(w for _, _, w in self.graph.edges(data=weight_attribute))
        self.reset()
        self.linkage_matrix = None

    def reset(self):
        self.clusters_map_of_sets = {}
        self.clusters_map_of_edges_sets = {}
        for i, base in enumerate(self.bases):
            self.clusters_map_of_sets[i] = set(base)
            graph_overlap = nx.subgraph(self.graph, base)
            self.clusters_map_of_edges_sets[i] = {
                (min(u, v), max(u, v)) for u, v in nx.edges(graph_overlap)
            }

    def calculate_ct_method_between_clusters(self, cluster1, cluster2, edges_list1, edges_list2):
        intersect = cluster1 & cluster2
        submatrix_indices = np.ix_(list(cluster1 - intersect), list(cluster2 - intersect))
        submatrix = self.ct_distance_matrix[submatrix_indices]
        if submatrix.size == 0:
            return 0.0
        if self.ct_linkage_method == GHACLinkageMethod.SINGLE:
            d = float(np.min(submatrix))
        elif self.ct_linkage_method == GHACLinkageMethod.COMPLETE:
            d = float(np.max(submatrix))
        else:
            d = float(np.average(submatrix))

        if len(intersect) > 0:
            if len(intersect) == 1:
                graph_overlap = nx.subgraph(self.graph, list(intersect))
            else:
                graph_overlap = nx.edge_subgraph(self.graph, edges_list1 & edges_list2).copy()
                graph_overlap.add_nodes_from(intersect)
            cliques_in_overlap = list(nx.find_cliques(graph_overlap))
            max_clique_size = len(max(cliques_in_overlap, key=len)) if cliques_in_overlap else 0
            denominator = 1 + max_clique_size
            if self.weight_attribute is not None and self.wt:
                cliques_in_overlap = [
                    clique for clique in cliques_in_overlap if len(clique) == max_clique_size
                ]
                weighted_cliques_list = [
                    sum(
                        w / self.wt
                        for w in nx.get_edge_attributes(
                            nx.subgraph(graph_overlap, clique), name=self.weight_attribute
                        ).values()
                    )
                    for clique in cliques_in_overlap
                ]
                max_overlap_weight = max(weighted_cliques_list) if weighted_cliques_list else 0.0
                denominator += max_overlap_weight
            d /= denominator
        return d

    def calculate_pairwise_distance_matrix(self):
        bases_count = len(self.bases)
        clusters_distance_matrix = np.zeros((bases_count, bases_count))
        for i in range(bases_count):
            for j in range(i + 1, bases_count):
                d = self.calculate_ct_method_between_clusters(
                    self.clusters_map_of_sets[i],
                    self.clusters_map_of_sets[j],
                    self.clusters_map_of_edges_sets[i],
                    self.clusters_map_of_edges_sets[j],
                )
                clusters_distance_matrix[i, j] = d
                clusters_distance_matrix[j, i] = d
        return clusters_distance_matrix

    def run(self):
        bases_count = len(self.bases)
        if bases_count <= 1:
            self.linkage_matrix = np.zeros((0, 4))
            return self.linkage_matrix

        distance_matrix = self.calculate_pairwise_distance_matrix()
        linkage_matrix = np.empty((bases_count - 1, 4))
        linkage_clusters_reuse_translation = list(range(bases_count))
        np.fill_diagonal(distance_matrix, np.inf)
        i = 0
        while i < bases_count - 1:
            tmp = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
            m1, m2 = tmp
            if not np.isfinite(distance_matrix[m1, m2]):
                break
            linkage_matrix[i, 0] = linkage_clusters_reuse_translation[m1]
            linkage_matrix[i, 1] = linkage_clusters_reuse_translation[m2]
            linkage_matrix[i, 2] = distance_matrix[m1, m2]
            linkage_clusters_reuse_translation[m1] = bases_count + i

            self.clusters_map_of_sets[m1] = self.clusters_map_of_sets[m1] | self.clusters_map_of_sets[m2]
            self.clusters_map_of_sets[m2] = None
            self.clusters_map_of_edges_sets[m1] = (
                self.clusters_map_of_edges_sets[m1] | self.clusters_map_of_edges_sets[m2]
            )
            self.clusters_map_of_edges_sets[m2] = None
            linkage_matrix[i, 3] = len(self.clusters_map_of_sets[m1])

            for idx in range(bases_count):
                if self.clusters_map_of_sets[idx] is not None and idx != m1:
                    d = self.calculate_ct_method_between_clusters(
                        self.clusters_map_of_sets[m1],
                        self.clusters_map_of_sets[idx],
                        self.clusters_map_of_edges_sets[m1],
                        self.clusters_map_of_edges_sets[idx],
                    )
                    distance_matrix[m1, idx] = d
                    distance_matrix[idx, m1] = d

            distance_matrix[m2, :] = np.inf
            distance_matrix[:, m2] = np.inf
            i += 1

        self.linkage_matrix = linkage_matrix[:i]
        return self.linkage_matrix


def _bases_to_communities(
    bases: list[tuple[int, ...]],
    labels: Iterable[int],
    graph: nx.Graph,
) -> list[list[int]]:
    grouped_bases: dict[int, list[tuple[int, ...]]] = {}
    for base, label in zip(bases, labels):
        grouped_bases.setdefault(int(label), []).append(base)

    communities = []
    for base_group in grouped_bases.values():
        nodes = set()
        for base in base_group:
            nodes.update(base)
        if len(nodes) > 1:
            communities.append(sorted(nodes))

    if not communities:
        communities = [[node] for node in graph.nodes()]
    else:
        covered = {node for community in communities for node in community}
        for node in graph.nodes():
            if node not in covered:
                communities.append([node])
    return communities


def wghac(
    g_original: object,
    min_base_size: int = 2,
    linkage_method: str = "single",
    ct_distance_matrix: Optional[np.ndarray] = None,
    weight: Optional[str] = None,
) -> NodeClustering:
    graph = _graph_to_integer_graph(g_original)
    if graph.number_of_nodes() == 0:
        return NodeClustering([], g_original, "wGHAC", overlap=True)

    if ct_distance_matrix is None:
        ct_distance_matrix = _shortest_path_matrix(graph, weight=weight)
    else:
        ct_distance_matrix = np.asarray(ct_distance_matrix, dtype=float)

    bases = _clique_bases(graph, min_base_size=min_base_size)
    if len(bases) == 1:
        communities = [list(bases[0])]
        communities = [[graph.nodes[n]["old_id"] for n in comm] for comm in communities]
        return NodeClustering(
            communities,
            g_original,
            "wGHAC",
            method_parameters={
                "min_base_size": min_base_size,
                "linkage_method": linkage_method,
                "weight": weight,
            },
            overlap=True,
        )

    linkage_enum = GHACLinkageMethod[linkage_method.upper()]
    ghac = GraphAgglomerativeClusteringClosedTrail(
        graph,
        linkage_enum,
        ct_distance_matrix,
        bases,
        weight_attribute=weight,
    )
    linkage_matrix = ghac.run()

    best_score = None
    best_communities = None
    if linkage_matrix.size == 0:
        labels = list(range(len(bases)))
        best_communities = _bases_to_communities(bases, labels, graph)
    else:
        for level in range(1, linkage_matrix.shape[0] + 1):
            labels = sch.fcluster(linkage_matrix, t=level, criterion="distance")
            communities = _bases_to_communities(bases, labels, graph)
            communities = [
                [graph.nodes[node]["old_id"] for node in community]
                for community in communities
                if len(community) > 0
            ]
            nc = NodeClustering(
                communities,
                g_original,
                "wGHAC",
                method_parameters={
                    "min_base_size": min_base_size,
                    "linkage_method": linkage_method,
                    "weight": weight,
                },
                overlap=True,
            )
            score = modularity_overlap(g_original, nc, weight).score
            if best_score is None or score > best_score:
                best_score = score
                best_communities = communities

    return NodeClustering(
        best_communities,
        g_original,
        "wGHAC",
        method_parameters={
            "min_base_size": min_base_size,
            "linkage_method": linkage_method,
            "weight": weight,
        },
        overlap=True,
    )
