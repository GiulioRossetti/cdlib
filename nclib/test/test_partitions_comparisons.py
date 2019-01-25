import unittest
import networkx as nx
import numpy as np
from nclib.algorithms import louvain, leiden
from nclib import evaluation


class PartitionsComparisonsTests(unittest.TestCase):

    def test_nmi(self):

        g = nx.karate_club_graph()
        louvain_communities = louvain(g)
        leiden_communities = leiden(g)

        score = evaluation.normalized_mutual_information(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_onmi(self):

        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = evaluation.overlapping_normalized_mutual_information(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_omega(self):

        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = evaluation.omega(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_f1(self):

        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = evaluation.f1(louvain_communities, leiden_communities)

        self.assertIsInstance(score, evaluation.MatchingResult)

    def test_nf1(self):

        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = evaluation.nf1(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_adjusted_rand(self):
        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = evaluation.adjusted_rand_index(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_adjusted_mutual(self):
        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = evaluation.adjusted_mutual_information(louvain_communities, leiden_communities)

        self.assertLessEqual(score, 1)
        self.assertGreaterEqual(score, 0)

    def test_variation_of_information(self):
        g = nx.karate_club_graph()
        leiden_communities = leiden(g)
        louvain_communities = louvain(g)

        score = evaluation.variation_of_information(louvain_communities, leiden_communities)

        self.assertLessEqual(score, np.log(g.number_of_nodes()))
        self.assertGreaterEqual(score, 0)
