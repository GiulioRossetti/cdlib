import unittest
import networkx as nx
from nclib.models.modularity import louvain
from nclib.evaluation.partition_quality import *


class PartitionQualityTests(unittest.TestCase):

    def test_link_modularity(self):

        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = link_modularity(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, 0)

    def test_modularity(self):
        g = nx.karate_club_graph()
        communities = louvain(g)

        mod = modularity(g, communities)
        self.assertLessEqual(mod, 1)
        self.assertGreaterEqual(mod, 0)

    def test_quality_indexes(self):
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
