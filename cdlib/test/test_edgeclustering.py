import unittest
import networkx as nx
from cdlib import algorithms
from cdlib import EdgeClustering


class EdgeClusteringTests(unittest.TestCase):
    def test_to_json(self):
        g = nx.karate_club_graph()
        coms = algorithms.hierarchical_link_community(g)
        self.assertIsInstance(coms, EdgeClustering)
        js = coms.to_json()
        self.assertIsInstance(js, str)

        coms = algorithms.hierarchical_link_community_w(g)
        self.assertIsInstance(coms, EdgeClustering)
        js = coms.to_json()
        self.assertIsInstance(js, str)

        coms = algorithms.hierarchical_link_community_full(g)
        self.assertIsInstance(coms, EdgeClustering)
        js = coms.to_json()
        self.assertIsInstance(js, str)

    def test_node_map(self):
        g = nx.karate_club_graph()
        coms = algorithms.hierarchical_link_community(g)
        edge_com_map = coms.to_edge_community_map()
        self.assertIsInstance(edge_com_map, dict)
