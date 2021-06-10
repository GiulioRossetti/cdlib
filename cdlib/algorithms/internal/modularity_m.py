import networkx as nx

import time
from random import random, shuffle


class ModularityMCommunityDiscovery:
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

    def update_sets_when_node_leaves(self, node, change_boundary=False):
        self.community.remove(node)
        if change_boundary:
            self.update_boundary_when_node_leaves(node)
        self.update_shell_when_node_leaves(node)

    def update_boundary_when_node_leaves(self, old_node):
        if old_node in self.boundary:
            self.boundary.remove(old_node)
            for node in self.graph.neighbors(old_node):
                if node in self.community:
                    self.boundary.add(node)

    def update_shell_when_node_leaves(self, old_node):
        possibles_leaving_nodes = [
            node for node in self.graph.neighbors(old_node) if node in self.shell
        ]
        for node in possibles_leaving_nodes:
            should_leave_shell = True
            for neighbor in self.graph.neighbors(node):
                if neighbor in self.community:
                    should_leave_shell = False
                    break
            if should_leave_shell:
                self.shell.remove(node)
        self.shell.add(old_node)

    def community_search(self, start_node, with_amend=False):
        self.set_start_node(start_node)
        sorted_shell = list(self.shell)

        modularity = 0.0
        while (
            len(self.community) < self.graph.number_of_nodes() and len(self.shell) > 1
        ):
            Q_list = []
            sorted_shell.sort(key=self.graph.degree)
            for candidate_node in sorted_shell:
                new_modularity = self.compute_modularity("addition", candidate_node)
                if new_modularity > modularity:
                    modularity = new_modularity
                    self.update_sets_when_node_joins(candidate_node)
                    sorted_shell.remove(candidate_node)
                    Q_list.append(candidate_node)

            while True:
                Q_delete = []
                for candidate_node in sorted(self.community, key=lambda x: random()):
                    new_modularity = self.compute_modularity("deletion", candidate_node)
                    if new_modularity > modularity:
                        modularity = new_modularity
                        self.update_sets_when_node_leaves(candidate_node)
                        Q_delete.append(candidate_node)

                        if candidate_node in Q_list:
                            Q_list.remove(candidate_node)

                if len(Q_delete) == 0:
                    break

            for node in sorted(Q_list, key=lambda x: random()):
                neighbors = list(self.graph.neighbors(node))
                shuffle(neighbors)
                for neighbor in neighbors:
                    if (neighbor in self.community) is False:
                        self.shell.add(neighbor)
                        if (neighbor in sorted_shell) is False:
                            sorted_shell.append(neighbor)

            if len(Q_list) == 0:
                break

        if self.starting_node in self.community:
            return sorted(self.community)
        return []

    def compute_modularity(self, auxiliary_info, candidate_node):
        mode = auxiliary_info
        ind_s, outd_s = 0, 0

        community = list(self.community)
        if mode == "addition":
            community.append(candidate_node)
        elif mode == "deletion":
            community.remove(candidate_node)

        for node in community:
            for neighbor in self.graph.neighbors(node):
                if neighbor in community:
                    ind_s += 1
                else:
                    outd_s += 1

        return float(ind_s) / float(outd_s)

    def should_leave_boundary(self, possibly_leaving_node, neighbor_node):
        neighbors = set(self.graph.neighbors(possibly_leaving_node))
        neighbors.discard(neighbor_node)
        for neighbor in neighbors:
            if (neighbor in self.community) is False:
                return False
        return True
