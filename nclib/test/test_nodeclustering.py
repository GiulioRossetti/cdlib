import unittest
import networkx as nx
from nclib import algorithms
from nclib import NodeClustering
from nclib import evaluation


class NodeClusteringTests(unittest.TestCase):

    def test_to_json(self):
        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        self.assertIsInstance(coms, NodeClustering)
        js = coms.to_json()
        self.assertIsInstance(js, str)
        self.assertEqual(coms.node_coverage, 1.0)

    def test_fitness_scores(self):
        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)

        self.assertIsInstance(coms.link_modularity(), float)
        self.assertIsInstance(coms.normalized_cut(), evaluation.FitnessResult)
        self.assertIsInstance(coms.internal_edge_density(), evaluation.FitnessResult)
        self.assertIsInstance(coms.average_internal_degree(), evaluation.FitnessResult)
        self.assertIsInstance(coms.fraction_over_median_degree(), evaluation.FitnessResult)
        self.assertIsInstance(coms.expansion(), evaluation.FitnessResult)
        self.assertIsInstance(coms.cut_ratio(), evaluation.FitnessResult)
        self.assertIsInstance(coms.edges_inside(), evaluation.FitnessResult)
        self.assertIsInstance(coms.conductance(), evaluation.FitnessResult)
        self.assertIsInstance(coms.max_odf(), evaluation.FitnessResult)
        self.assertIsInstance(coms.avg_odf(), evaluation.FitnessResult)
        self.assertIsInstance(coms.flake_odf(), evaluation.FitnessResult)
        self.assertIsInstance(coms.triangle_participation_ratio(), evaluation.FitnessResult)
        self.assertIsInstance(coms.newman_girvan_modularity(), float)
        self.assertIsInstance(coms.erdos_renyi_modularity(), float)
        self.assertIsInstance(coms.modularity_density(), float)
        self.assertIsInstance(coms.z_modularity(), float)
        self.assertIsInstance(coms.surprise(), float)
        self.assertIsInstance(coms.significance(), float)

    def test_node_map(self):
        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        node_com_map = coms.to_node_community_map()
        self.assertIsInstance(node_com_map, dict)

    def test_comparison(self):
        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.leiden(g)

        self.assertIsInstance(coms.normalized_mutual_information(coms2), float)
        self.assertIsInstance(coms.overlapping_normalized_mutual_information(coms2), float)
        self.assertIsInstance(coms.omega(coms2), float)
        self.assertIsInstance(coms.f1(coms2), evaluation.MatchingResult)
        self.assertIsInstance(coms.nf1(coms2), float)
        self.assertIsInstance(coms.adjusted_mutual_information(coms2), float)
        self.assertIsInstance(coms.adjusted_rand_index(coms2), float)
        self.assertIsInstance(coms.variation_of_information(coms2), float)
