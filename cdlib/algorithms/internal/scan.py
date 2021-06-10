import random
import math

"""
Xu, X., Yuruk, N., Feng, Z., & Schweiger, T. A. (2007, August). 
Scan: a structural clustering algorithm for networks. 
In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 824-833). .
"""


def cal_similarity(G, node_i, node_j):
    s1 = set(G.neighbors(node_i))
    s1.add(node_i)
    s2 = set(G.neighbors(node_j))
    s2.add(node_j)
    return len(s1 & s2) / math.sqrt(len(s1) * len(s2))


class SCAN_nx:
    def __init__(self, g, epsilon=0.5, mu=3):
        self.g = g
        self.epsilon = epsilon
        self.mu = mu

    def get_epsilon_neighbor(self, node):
        return [
            neighbor
            for neighbor in self.g.neighbors(node)
            if cal_similarity(self.g, node, neighbor) >= self.epsilon
        ]

    def is_core(self, node):
        return len(self.get_epsilon_neighbor(node)) >= self.mu

    def get_hubs_outliers(self, communities):
        other_nodes = set(self.g.nodes.keys())
        node_community = {}
        for i, c in enumerate(communities):
            for node in c:
                other_nodes.discard(node)
                node_community[node] = i
        hubs = []
        outliers = []
        for node in other_nodes:
            neighbors = self.g.neighbors(node)
            neighbor_community = set()
            for neighbor in neighbors:
                if neighbor in node_community:
                    neighbor_community.add(node_community[neighbor])
            if len(neighbor_community) > 1:
                hubs.append(node)
            else:
                outliers.append(node)
        return hubs, outliers

    def execute(self):
        # random scan nodes
        visit_sequence = list(self.g.nodes.keys())
        random.shuffle(visit_sequence)
        communities = []
        for node_name in visit_sequence:
            node = self.g.nodes[node_name]
            if node.get("classified"):
                continue
            if self.is_core(node_name):  # a new algorithms
                community = [node_name]
                communities.append(community)
                node["type"] = "core"
                node["classified"] = True
                queue = self.get_epsilon_neighbor(node_name)

                while len(queue) != 0:
                    temp = queue.pop(0)
                    if not self.g.nodes[temp].get("classified"):
                        self.g.nodes[temp]["classified"] = True
                        community.append(temp)
                    if not self.is_core(temp):
                        continue
                    R = self.get_epsilon_neighbor(temp)
                    for r in R:
                        node_r = self.g.nodes[r]
                        is_classified = node_r.get("classified")
                        if is_classified:
                            continue
                        node_r["classified"] = True
                        community.append(r)
                        if node_r.get("type") != "non-member":
                            queue.append(r)
                        else:
                            node["type"] = "non-member"
        return list(communities)
