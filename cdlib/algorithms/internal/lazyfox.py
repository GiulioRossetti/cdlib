from __future__ import annotations

from collections import defaultdict
from enum import Enum

import networkx as nx
import numpy as np


def _intersection_size(list1, list2):
    return len(set(list1).intersection(set(list2)))


class ChangeType(Enum):
    STAY = 1
    COPY = 2
    LEAVE = 3
    TRANSFER = 4
    NONE = 5


class Change:
    def __init__(self, node_id=-1, source_community=-1, target_community=-1):
        self.node_id = node_id
        self.source_community = source_community
        self.target_community = target_community

    def get_type(self):
        if self.source_community == -1 and self.target_community == -1:
            return ChangeType.STAY
        if self.source_community == -1 and self.target_community != -1:
            return ChangeType.COPY
        if self.source_community != -1 and self.target_community == -1:
            return ChangeType.LEAVE
        if self.source_community != -1 and self.target_community != -1:
            return ChangeType.TRANSFER
        return ChangeType.NONE


class ChangeCounter:
    def __init__(self):
        self.stays = 0
        self.copies = 0
        self.leaves = 0
        self.transfers = 0

    def add(self, change):
        change_type = change.get_type()
        if change_type == ChangeType.STAY:
            self.stays += 1
        elif change_type == ChangeType.COPY:
            self.copies += 1
        elif change_type == ChangeType.LEAVE:
            self.leaves += 1
        elif change_type == ChangeType.TRANSFER:
            self.transfers += 1


class Fox:
    def __init__(self, graph: nx.Graph, threshold=0.01):
        if graph.number_of_nodes() == 0:
            self.G = graph.copy()
            self.original_labels = {}
            self.threshold = threshold
            self.number_nodes = 0
            self.adjacency_list = {}
            self.hashmap_nc = defaultdict(list)
            self.hashmap_cwcc = {}
            self.hashmap_clookup = defaultdict(dict)
            self.number_neighbors = []
            self.cc_per_node = []
            self.node_ids = []
            self.iteration_count = 1
            self.global_wcc = 0.0
            self.wcc_diff = 0.0
            self.cc = 0.0
            return

        self.threshold = threshold
        self.iteration_count = 1
        self.cc = 0.0
        self.global_wcc = 0.0
        self.wcc_diff = 0.0

        self.G = nx.convert_node_labels_to_integers(graph.copy(), label_attribute="old_id")
        self.original_labels = {
            node: data["old_id"] for node, data in self.G.nodes(data=True)
        }
        self.adjacency_list = {
            node: list(sorted(self.G.neighbors(node))) for node in self.G.nodes
        }
        self.number_nodes = self.G.number_of_nodes()
        self.hashmap_nc = defaultdict(list)
        self.hashmap_cwcc = {}
        self.hashmap_clookup = defaultdict(dict)
        self.number_neighbors = []
        self.cc_per_node = []

        self.count_neighbors_per_node()
        self.calculate_cc_per_node()
        self.order_nodes()
        self.cc = self.calculate_global_cc()
        self.initialize_cluster()
        self.initialize_cluster_maps()
        self.global_wcc = sum(self.hashmap_cwcc.values())
        self.remove_all_single_node_communities()

    def initialize_cluster(self):
        max_community_id = -1
        for node_id in self.node_ids:
            if self.hashmap_nc[node_id]:
                continue
            max_community_id += 1
            self.hashmap_nc[node_id].append(max_community_id)
            for neighbor_id in self.adjacency_list[node_id]:
                if not self.hashmap_nc[neighbor_id]:
                    self.hashmap_nc[neighbor_id].append(max_community_id)
        self.maximum_community_id = max_community_id

    def initialize_cluster_maps(self):
        for node_id in self.node_ids:
            community = self.hashmap_nc[node_id][0]
            cmi = self.hashmap_clookup.get(community, {})
            cmi[node_id] = 0
            self.hashmap_clookup[community] = cmi

        for community_id in range(self.maximum_community_id + 1):
            cmi = self.hashmap_clookup.get(community_id, {})
            nodes_in_community = list(cmi.keys())
            for node_id in cmi:
                cmi[node_id] = float(
                    _intersection_size(nodes_in_community, self.adjacency_list[node_id])
                )
            self.hashmap_clookup[community_id] = cmi

        for community_id in range(self.maximum_community_id + 1):
            community = self.hashmap_clookup.get(community_id, {})
            self.hashmap_cwcc[community_id] = self.calculate_wcc_dach_community(community)

    def calculate_cc_per_node(self):
        cc = nx.clustering(self.G)
        self.cc_per_node = [cc[i] for i in range(self.number_nodes)]

    def calculate_global_cc(self):
        if self.number_nodes == 0:
            return 0.0
        return sum(self.cc_per_node) / float(self.number_nodes)

    def count_neighbors_per_node(self):
        self.number_neighbors = [len(neighbors) for neighbors in self.adjacency_list.values()]

    def order_nodes(self):
        self.node_ids = list(range(self.number_nodes))
        self.node_ids.sort(key=lambda a: self.number_neighbors[a], reverse=True)
        self.node_ids.sort(key=lambda a: self.cc_per_node[a], reverse=True)

    def calculate_wcc_dach(self, node_id, community):
        if len(community) <= 1:
            return 0.0
        node_degree = self.number_neighbors[node_id]
        if node_degree <= 1:
            return 0.0

        node_degree_to_community = community[node_id]
        node_degree_to_graph_without_community = float(node_degree) - node_degree_to_community
        edges_in_community = sum(community.values()) / 2.0
        possible_edges_in_community = float(len(community) * (len(community) - 1) / 2.0)
        if possible_edges_in_community == 0:
            return 0.0
        community_density = edges_in_community / possible_edges_in_community

        expected_triangles_with_community = (
            node_degree_to_community * (node_degree_to_community - 1) / 2.0
        ) * community_density
        expected_triangles_with_graph = (node_degree * (node_degree - 1) / 2.0) * self.cc
        if expected_triangles_with_graph == 0:
            return 0.0
        denom = len(community) - 1 + node_degree_to_graph_without_community
        if denom == 0:
            return 0.0
        return (
            (expected_triangles_with_community / expected_triangles_with_graph)
            * node_degree
            / denom
        )

    def calculate_wcc_dach_community(self, community):
        return sum(self.calculate_wcc_dach(node_id, community) for node_id in community)

    def remove_single_node_community(self, community_id):
        if len(self.hashmap_clookup.get(community_id, {})) > 1:
            raise ValueError("Community contains more than 1 node!")
        last_node_id = next(iter(self.hashmap_clookup[community_id]))
        del self.hashmap_clookup[community_id]
        del self.hashmap_cwcc[community_id]
        if last_node_id in self.hashmap_nc:
            self.hashmap_nc[last_node_id] = [
                community for community in self.hashmap_nc[last_node_id] if community != community_id
            ]

    def remove_all_single_node_communities(self):
        for community_id in range(self.maximum_community_id + 1):
            if (
                community_id in self.hashmap_clookup
                and len(self.hashmap_clookup[community_id]) <= 1
            ):
                self.remove_single_node_community(community_id)

    def get_altered_community_leave(self, node_id, community_id):
        altered_community = self.hashmap_clookup[community_id].copy()
        for neighbor in self.adjacency_list[node_id]:
            if community_id in self.hashmap_nc[neighbor]:
                altered_community[neighbor] = altered_community.get(neighbor, 0) - 1
        del altered_community[node_id]
        return altered_community

    def get_altered_community_join(self, node_id, community_id):
        altered_community = self.hashmap_clookup[community_id].copy()
        altered_community[node_id] = 0
        for neighbor in self.adjacency_list[node_id]:
            if community_id in self.hashmap_nc[neighbor]:
                altered_community[neighbor] = altered_community.get(neighbor, 0) + 1
                altered_community[node_id] = altered_community.get(node_id, 0) + 1
        return altered_community

    def calculate_delta_l(self, node_id, community_id):
        altered_community = self.get_altered_community_leave(node_id, community_id)
        return self.calculate_wcc_dach_community(altered_community) - self.hashmap_cwcc[community_id]

    def calculate_delta_j(self, node_id, community_id):
        altered_community = self.get_altered_community_join(node_id, community_id)
        return self.calculate_wcc_dach_community(altered_community) - self.hashmap_cwcc[community_id]

    def community_to_leave(self, node_id):
        best_move_delta_l = 0
        best_cid = -1
        for community_id in self.hashmap_nc[node_id]:
            delta_l = self.calculate_delta_l(node_id, community_id)
            if delta_l > best_move_delta_l:
                best_move_delta_l = delta_l
                best_cid = community_id
        return best_cid

    def community_to_join(self, node_id):
        best_move_delta_j = 0
        best_cid = -1
        relevant_communities = set()
        for neighbor in self.adjacency_list[node_id]:
            for community_id in self.hashmap_nc[neighbor]:
                relevant_communities.add(community_id)
        for community_id in self.hashmap_nc[node_id]:
            relevant_communities.discard(community_id)
        for community_id in relevant_communities:
            delta_j = self.calculate_delta_j(node_id, community_id)
            if delta_j > best_move_delta_j:
                best_move_delta_j = delta_j
                best_cid = community_id
        return best_cid

    def decide(self, node_id):
        change = Change(node_id, -1, -1)
        change.source_community = self.community_to_leave(node_id)
        change.target_community = self.community_to_join(node_id)
        return change

    def apply(self, change):
        if change.get_type() in (ChangeType.TRANSFER, ChangeType.COPY):
            altered_community = self.get_altered_community_join(
                change.node_id, change.target_community
            )
            new_wcc = self.calculate_wcc_dach_community(altered_community)
            self.wcc_diff += new_wcc - self.hashmap_cwcc[change.target_community]
            self.hashmap_clookup[change.target_community] = altered_community
            self.hashmap_cwcc[change.target_community] = new_wcc
            self.hashmap_nc[change.node_id].append(change.target_community)

        if change.get_type() in (ChangeType.TRANSFER, ChangeType.LEAVE):
            altered_community = self.get_altered_community_leave(
                change.node_id, change.source_community
            )
            new_wcc = self.calculate_wcc_dach_community(altered_community)
            self.wcc_diff += new_wcc - self.hashmap_cwcc[change.source_community]
            self.hashmap_clookup[change.source_community] = altered_community
            self.hashmap_cwcc[change.source_community] = new_wcc
            self.hashmap_nc[change.node_id].remove(change.source_community)

    def process_nodes(self, change_counter):
        changes = []
        for node_id in self.node_ids:
            change = self.decide(node_id)
            change_counter.add(change)
            if change.get_type() != ChangeType.STAY:
                changes.append(change)
        return changes

    def run(self):
        while True:
            change_counter = ChangeCounter()
            self.wcc_diff = 0.0
            changes = self.process_nodes(change_counter)
            for change in changes:
                self.apply(change)
            self.remove_all_single_node_communities()
            relative_change = 0.0 if self.global_wcc == 0 else self.wcc_diff / self.global_wcc
            self.global_wcc += self.wcc_diff
            done = (change_counter.stays == self.number_nodes) or (
                relative_change < self.threshold
            )
            if done:
                break
            self.iteration_count += 1

    def communities(self):
        result = []
        for community_id in sorted(self.hashmap_clookup):
            nodes = [
                self.original_labels[node_id]
                for node_id in sorted(self.hashmap_clookup[community_id].keys())
            ]
            if len(nodes) > 0:
                result.append(nodes)
        return result
