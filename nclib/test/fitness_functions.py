import unittest
from nclib.models.modularity import louvain
from nclib.evaluation.fitness_functions import *


class FitnessFunctionsTests(unittest.TestCase):

    def test_link_modularity(self):

        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = link_modularity(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, 0)

    def test_modularity(self):
        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = newman_girvan_modularity(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, -0.5)

        mod = erdos_renyi_modularity(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, -0.5)

        mod = modularity_density(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, -0.5)

        mod = z_modularity(g, communities)
        self.assertLessEqual(mod, np.sqrt(g.number_of_nodes()))
        self.assertGreaterEqual(mod, -0.5)

    def test_surprise(self):

        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = surprise(g, communities)
        self.assertLessEqual(mod, g.number_of_edges())
        self.assertGreaterEqual(mod, 0)

    def test_significance(self):
        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = significance(g, communities)
        self.assertGreaterEqual(mod, 0)

    def test_pquality_indexes(self):
        g = nx.karate_club_graph()
        communities = louvain(g)

        indexes = [normalized_cut, internal_edge_density, average_internal_degree, fraction_over_median_degree,
                   expansion, cut_ratio, edges_inside, conductance, max_odf, avg_odf, flake_odf,
                   triangle_participation_ratio]

        for idx in indexes:
            res = quality_indexes(g, communities, idx)
            self.assertIsInstance(res, Result)


if __name__ == '__main__':
    unittest.main()
