import unittest
from nclib.algorithms import louvain
import networkx as nx
import numpy as np
from nclib import evaluation


class FitnessFunctionsTests(unittest.TestCase):

    def test_link_modularity(self):

        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = evaluation.link_modularity(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, 0)

    def test_modularity(self):
        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = evaluation.newman_girvan_modularity(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, -0.5)

        mod = evaluation.erdos_renyi_modularity(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, -0.5)

        mod = evaluation.modularity_density(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, -0.5)

        mod = evaluation.z_modularity(g, communities)
        self.assertLessEqual(mod, np.sqrt(g.number_of_nodes()))
        self.assertGreaterEqual(mod, -0.5)

    def test_surprise(self):

        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = evaluation.surprise(g, communities)
        self.assertLessEqual(mod, g.number_of_edges())
        self.assertGreaterEqual(mod, 0)

    def test_significance(self):
        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = evaluation.significance(g, communities)
        self.assertGreaterEqual(mod, 0)

    def test_pquality_indexes(self):
        g = nx.karate_club_graph()
        communities = louvain(g)

        indexes = [evaluation.normalized_cut, evaluation.internal_edge_density, evaluation.average_internal_degree,
                   evaluation.fraction_over_median_degree, evaluation.expansion, evaluation.cut_ratio,
                   evaluation.edges_inside, evaluation.conductance, evaluation.max_odf, evaluation.avg_odf,
                   evaluation.flake_odf, evaluation.triangle_participation_ratio]

        for idx in indexes:
            res = idx(g, communities)
            self.assertIsInstance(res, evaluation.FitnessResult)


if __name__ == '__main__':
    unittest.main()
