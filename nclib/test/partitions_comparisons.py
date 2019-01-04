import unittest
import networkx as nx
from nclib.models.modularity import louvain, leiden
from nclib.evaluation.partitions_comparison import *


class PartitionsComparisonsTests(unittest.TestCase):

    def test_nmi(self):

        g = nx.karate_club_graph()
        louvain_communities = louvain(g)
        leiden_communities = leiden(g)

        score = normalized_mutual_information(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_onmi(self):

        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = overlapping_normalized_mutual_information(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_omega(self):

        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = omega(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_f1(self):

        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = f1(louvain_communities, leiden_communities)

        self.assertIsInstance(score, Result)

    def test_nf1(self):

        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = nf1(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_adjusted_rand(self):
        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = adjusted_rand_score(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_adjusted_mutual(self):
        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = adjusted_mutual_information(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_variation_of_information(self):
        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = variation_of_information(louvain_communities, leiden_communities)

        self.assertLessEqual(score, np.log(g.number_of_nodes()))
        self.assertGreaterEqual(score, 0)


if __name__ == '__main__':
    unittest.main()
