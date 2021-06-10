import collections
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN


class WalkSCAN(object):
    def __init__(self, nb_steps=2, eps=0.1, min_samples=3):
        self.nb_steps = nb_steps
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_ = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def load(self, graph, init_vector):
        self.graph = graph.copy()
        self.init_vector = init_vector.copy()

    def embed_nodes(self):
        p = {0: self.init_vector.copy()}
        for t in range(self.nb_steps):
            p[t + 1] = collections.defaultdict(int)
            for v in p[t]:
                for (_, w, e_data) in self.graph.edges(v, data=True):
                    if "weight" in e_data:
                        self.weighted_ = True
                        p[t + 1][w] += (
                            float(e_data["weight"])
                            / float(self.graph.degree(v, weight="weight"))
                            * p[t][v]
                        )
                    else:
                        self.weighted_ = False
                        p[t + 1][w] += 1.0 / float(self.graph.degree(v)) * p[t][v]
        self.embedded_value_ = dict()
        self.embedded_nodes_ = list()
        for v in p[self.nb_steps]:
            self.embedded_nodes_.append(v)
            self.embedded_value_[v] = np.array(
                [p[t + 1][v] for t in range(self.nb_steps)]
            )
        self.nb_embedded_nodes_ = len(self.embedded_nodes_)

    def find_cores(self):
        if self.nb_embedded_nodes_ > 0:
            P = np.zeros((self.nb_embedded_nodes_, self.nb_steps))
            for (i, node) in enumerate(self.embedded_nodes_):
                P[i, :] = self.embedded_value_[node]
            self.dbscan_.fit(P)
            self.cores_ = collections.defaultdict(set)
            self.outliers_ = set()
            for (i, node) in enumerate(self.embedded_nodes_):
                label = self.dbscan_.labels_[i]
                if label >= 0:
                    self.cores_[label].add(node)
                else:
                    self.outliers_.add(node)
        else:
            self.cores_ = {}
            self.outliers_ = set()

    def compute_core_average_value(self):
        self.core_average_value_ = dict()
        for (core_id, core) in self.cores_.items():
            self.core_average_value_[core_id] = np.zeros(self.nb_steps)
            for node in core:
                for t in range(self.nb_steps):
                    self.core_average_value_[core_id][t] += self.embedded_value_[node][
                        t
                    ] / float(len(core))

    def sort_cores(self):
        self.sorted_core_ids_ = list(self.cores_.keys())
        self.sorted_core_ids_.sort(
            key=lambda i: list(self.core_average_value_[i]), reverse=True
        )
        self.sorted_cores_ = [self.cores_[i] for i in self.sorted_core_ids_]

    def aggregate_outliers(self):
        self.communities_ = list()
        for core in self.sorted_cores_:
            community = core.copy()
            for node in core:
                community |= set(nx.neighbors(self.graph, node)) & self.outliers_
            self.communities_.append(community)

    def detect_communities(self, graph, init_vector):
        self.load(graph, init_vector)
        self.embed_nodes()
        self.find_cores()
        self.compute_core_average_value()
        self.sort_cores()
        self.aggregate_outliers()
