import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from queue import Queue
from collections import defaultdict

# from dgl.data import LegacyTUDataset


class EBGC:
    def __init__(self):
        self.g = None
        self.adj = None
        self.node_num = None
        self.node_state = None
        self.node_degree = None
        self.cluster_node = None
        pass

    def node_neighbor_idxs(self, node_idx):
        neigh_list = np.array([n for n in self.g.neighbors(node_idx)])
        # neigh_list = np.array(list(dict.keys(dict(self.g[node_idx]))))
        return neigh_list

    def node_entropy(self, node_idx, cluster_idx):
        neigh_list = self.node_neighbor_idxs(node_idx)
        cluster_list = self.cluster_node[cluster_idx]
        n = np.sum(cluster_list[neigh_list])
        p_inner = n / len(neigh_list)
        p_outer = 1 - p_inner
        if p_inner == 0 or p_inner == 1:
            e_v = 0
        else:
            e_v = -p_inner * np.log2(p_inner) - p_outer * np.log2(p_outer)

        return e_v

    def graph_entropy(self, cluster_idx):
        e_g = 0
        for idx in range(self.node_num):
            e_g += self.node_entropy(idx, cluster_idx)

        return e_g

    def add_node_to_cluster(self, node_idx, cluster_idx):
        self.cluster_node[cluster_idx][node_idx] = 1
        self.node_state[node_idx] = 1

    def rm_node_to_cluster(self, node_idx, cluster_idx):
        self.cluster_node[cluster_idx][node_idx] = 0
        self.node_state[node_idx] = 0

    def cluster_add_neighbors(self, cluster_idx, seed_neighbors):
        cluster_list = self.cluster_node[cluster_idx]
        seed_neighbors = seed_neighbors[np.where(self.node_state[seed_neighbors] == 0)]
        cluster_list[seed_neighbors] = 1
        self.cluster_node[cluster_idx] = cluster_list
        self.node_state[seed_neighbors] = 1
        return seed_neighbors

    def cluster_removal_neighbors(self, cluster_idx, seed_neighbors):

        checked_seed_neighbors = []
        min_e_g = self.graph_entropy(cluster_idx)

        for neigh_idx in seed_neighbors:
            # remove neigh_idx node
            self.rm_node_to_cluster(neigh_idx, cluster_idx)
            e_g = self.graph_entropy(cluster_idx)
            if e_g < min_e_g:
                min_e_g = e_g
            else:
                # add neigh_idx node
                self.add_node_to_cluster(neigh_idx, cluster_idx)
                checked_seed_neighbors.append(neigh_idx)
        return checked_seed_neighbors

    def select_seed_ver(self):
        # Select a seed: Here we follow the decrease order of nodes' degree
        # t = np.argsort(self.node_degree)
        # deg = self.node_degree
        for idx in np.argsort(self.node_degree)[::-1]:
            if self.node_state[idx] == 0:
                # add seed into the cluster (create a cluster)
                seed_idx = idx
                cluster_list = np.zeros(self.node_num)
                self.cluster_node.append(cluster_list)
                cluster_idx = len(self.cluster_node) - 1
                self.add_node_to_cluster(seed_idx, cluster_idx)
                break

        seed_neighbors = self.node_neighbor_idxs(seed_idx)
        seed_neighbors = self.cluster_add_neighbors(cluster_idx, seed_neighbors)

        return cluster_idx, seed_neighbors

    def add_boundary_ver(self, check_list, cluster_idx):
        node_que = Queue()
        [node_que.put(x) for x in check_list]
        min_e_g = self.graph_entropy(cluster_idx)

        while not node_que.empty():
            node_idx = node_que.get()

            # find neighbor of node_idx
            neigh_idx = self.node_neighbor_idxs(node_idx)

            # check neighbor
            # the state of neighbor is 0
            neigh_idx = neigh_idx[np.where(self.node_state[neigh_idx] == 0)]
            # add the neighbor into the cluster which lowers the graph entropy
            for idx in neigh_idx:
                self.add_node_to_cluster(idx, cluster_idx)
                e_g = self.graph_entropy(cluster_idx)
                if e_g < min_e_g:
                    min_e_g = e_g
                    node_que.put(idx)
                else:
                    self.rm_node_to_cluster(idx, cluster_idx)

    def converge_check(self):
        return np.any(self.node_state == 0)

    def fit(self, g):
        self.g = g
        degree = np.array(g.degree)
        degree = degree[np.argsort(degree[:, 0])]
        self.node_degree = [x[1] for x in degree]
        self.node_num = len(self.node_degree)
        self.node_state = np.zeros(self.node_num)
        self.cluster_node = []

        while self.converge_check():

            # Step 1
            cluster_idx, seed_neighbors = self.select_seed_ver()

            # Step 2
            checked_seed_neighbors = self.cluster_removal_neighbors(
                cluster_idx, seed_neighbors
            )

            # Step 3
            self.add_boundary_ver(checked_seed_neighbors, cluster_idx)

        return np.array(self.cluster_node).T


def draw_graph(g, node_labels):
    # pos = nx.spring_layout(g)
    # nx.draw_spring(g, with_labels=False)
    # nx.draw_networkx_edge_labels(g, pos, font_size=14, alpha=0.5, rotate=True)

    k = 1 / (np.max(node_labels))
    val_map = {x: x * k for x in node_labels}
    values = [val_map.get(node_labels[node]) for node in g.nodes()]

    nx.draw(
        g,
        cmap=plt.get_cmap("viridis"),
        node_color=values,
        with_labels=True,
        font_color="white",
    )

    plt.axis("off")
    plt.show()
