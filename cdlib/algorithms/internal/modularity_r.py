import networkx as nx

import time
from random import random


class ModularityRCommunityDiscovery(object):
    minimum_improvement = 0.000001

    def __init__(self, graph):
        self.graph = graph
        self.starting_node = None
        self.community = []
        self.boundary = set()
        self.shell = set()
        self.remove_self_loops()

    def reset(self):
        self.community.clear()
        self.boundary.clear()
        self.shell.clear()

    def remove_self_loops(self):
        for node in self.graph.nodes():
            if self.graph.has_edge(node, node):
                self.graph.remove_edge(node, node)

    def set_start_node(self, start_node):
        if start_node in self.graph.nodes():
            self.starting_node = start_node
            self.community.append(start_node)
            self.boundary.add(start_node)
            self.shell = set(self.graph.neighbors(start_node))
        else:
            print("Invalid starting node! Try with another one.")
            exit(-1)

    def update_sets_when_node_joins(self, node, change_boundary=False):
        self.community.append(node)
        if change_boundary:
            self.update_boundary_when_node_joins(node)
        self.update_shell_when_node_joins(node)

    def update_shell_when_node_joins(self, new_node):
        self.shell.update(self.graph.neighbors(new_node))
        for node in self.community:
            self.shell.discard(node)

    def update_boundary_when_node_joins(self, new_node):
        should_be_boundary = False
        for neighbor in self.graph.neighbors(new_node):
            if (neighbor in self.community) is False:
                should_be_boundary = True
                break
        if should_be_boundary:
            self.boundary.add(new_node)

    def find_best_next_node(self, improvements):
        best_candidate = None
        best_improvement = -float("inf")
        for candidate, improvement in sorted(
            improvements.items(), key=lambda x: random()
        ):
            if improvement > best_improvement:
                best_candidate = candidate
                best_improvement = improvement
        return best_candidate

    def community_search(self, start_node):
        self.set_start_node(start_node)
        modularity_r = 0.0
        T = self.graph.degree[start_node]

        while (
            len(self.community) < self.graph.number_of_nodes() and len(self.shell) > 0
        ):
            delta_r = (
                {}
            )  # key: candidate nodes from the shell set, value: total improved strength after a node joins.
            delta_T = (
                {}
            )  # key: candidate nodes from the shell set, value: delta T (based on notations of the paper).
            for node in self.shell:
                delta_r[node], delta_T[node] = self.compute_modularity(
                    (modularity_r, T), node
                )

            new_node = self.find_best_next_node(delta_r)
            if delta_r[new_node] < ModularityRCommunityDiscovery.minimum_improvement:
                break

            modularity_r += delta_r[new_node]
            T += delta_T[new_node]
            self.update_sets_when_node_joins(new_node, change_boundary=True)

        return sorted(
            self.community
        )  # sort is only for a better representation, can be ignored to boost performance.

    def compute_modularity(self, auxiliary_info, candidate_node):
        R, T = auxiliary_info
        x, y, z = 0, 0, 0
        for neighbor in self.graph.neighbors(candidate_node):
            if neighbor in self.boundary:
                x += 1
            else:
                y += 1

        for neighbor in [
            node
            for node in self.graph.neighbors(candidate_node)
            if node in self.boundary
        ]:
            if self.should_leave_boundary(neighbor, candidate_node):
                for node in self.graph.neighbors(neighbor):
                    if (node in self.community) and ((node in self.boundary) is False):
                        z += 1
        return float(x - R * y - z * (1 - R)) / float(T - z + y), -z + y

    def should_leave_boundary(self, possibly_leaving_node, neighbor_node):
        neighbors = set(self.graph.neighbors(possibly_leaving_node))
        neighbors.discard(neighbor_node)
        for neighbor in neighbors:
            if (neighbor in self.community) is False:
                return False
        return True
