import unittest
from cdlib.algorithms import louvain, eva
import networkx as nx
import numpy as np
import random
from cdlib import evaluation


class FitnessFunctionsTests(unittest.TestCase):
    def test_link_modularity(self):

        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = evaluation.link_modularity(g, communities)
        self.assertLessEqual(mod.score, 1)
        self.assertGreaterEqual(mod.score, 0)

    def test_modularity(self):
        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = evaluation.newman_girvan_modularity(g, communities)
        self.assertLessEqual(mod.score, 1)
        self.assertGreaterEqual(mod.score, -0.5)

        mod = evaluation.erdos_renyi_modularity(g, communities)
        self.assertLessEqual(mod.score, 1)
        self.assertGreaterEqual(mod.score, -0.5)

        mod = evaluation.modularity_density(g, communities)
        self.assertIsInstance(mod.score, float)

        mod = evaluation.z_modularity(g, communities)
        self.assertLessEqual(mod.score, np.sqrt(g.number_of_nodes()))
        self.assertGreaterEqual(mod.score, -0.5)

    def test_surprise(self):

        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = evaluation.surprise(g, communities)
        self.assertLessEqual(mod.score, g.number_of_edges())
        self.assertGreaterEqual(mod.score, 0)

    def test_significance(self):
        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = evaluation.significance(g, communities)
        self.assertGreaterEqual(mod.score, 0)

    def test_pquality_indexes(self):
        g = nx.karate_club_graph()

        communities = louvain(g)

        indexes = [
            evaluation.normalized_cut,
            evaluation.internal_edge_density,
            evaluation.average_internal_degree,
            evaluation.fraction_over_median_degree,
            evaluation.expansion,
            evaluation.cut_ratio,
            evaluation.edges_inside,
            evaluation.conductance,
            evaluation.max_odf,
            evaluation.avg_odf,
            evaluation.flake_odf,
            evaluation.triangle_participation_ratio,
            evaluation.size,
            evaluation.avg_embeddedness,
            evaluation.scaled_density,
            evaluation.avg_distance,
            evaluation.hub_dominance,
            evaluation.avg_transitivity,
            evaluation.modularity_overlap,
        ]

        for idx in indexes:
            res = idx(g, communities)
            self.assertIsInstance(res, evaluation.FitnessResult)

        for idx in indexes:
            try:
                res = idx(g, communities, summary=False)
                self.assertIsInstance(res, list)
            except:
                pass

        g.add_node(100)
        a = communities.communities[0]
        a.append(100)
        communities.communities[0] = a

        for idx in indexes:
            res = idx(g, communities)
            self.assertIsInstance(res, evaluation.FitnessResult)

        for idx in indexes:
            try:
                res = idx(g, communities, summary=False)
                self.assertIsInstance(res, list)
            except:
                pass

    def test_purity(self):

        l1 = ["one", "two", "three", "four"]
        l2 = ["A", "B", "C"]
        g_attr = nx.barabasi_albert_graph(100, 5)
        labels = dict()

        for node in g_attr.nodes():
            labels[node] = {"l1": random.choice(l1), "l2": random.choice(l2)}

        coms = eva(g_attr, labels, alpha=0.8)

        pur = evaluation.purity(coms)

        self.assertGreaterEqual(pur.score, 0)
        self.assertLessEqual(pur.score, 1)
