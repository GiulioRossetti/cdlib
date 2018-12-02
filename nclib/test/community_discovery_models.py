import unittest
from nclib.models.node_centric import demon, angel, ego_networks, node_perception
from nclib.models.modularity import louvain, leiden, greedy_modularity
from nclib.models.map_equation import infomap
from nclib.models.label_propagation import label_propagation, async_fluid
import networkx as nx
import os


class NodeCentricTests(unittest.TestCase):

    def test_ego(self):
        g = nx.karate_club_graph()
        coms = ego_networks(g)
        self.assertEqual(len(coms), g.number_of_nodes())

    def test_demon(self):
        g = nx.karate_club_graph()
        coms = demon(g, epsilon=0.25)
        self.assertEqual(len(coms), 2)

    def test_nodeperception(self):
        g = nx.karate_club_graph()
        coms = node_perception(g, threshold=0.25, overlap_threshold=0.25)
        self.assertEqual(len(coms), 42)

    def test_angel(self):
        g = nx.karate_club_graph()
        coms = angel(g, threshold=0.25)
        self.assertEqual(len(coms), 1)

    def test_louvain(self):
        g = nx.karate_club_graph()
        coms = louvain(g)
        self.assertEqual(len(coms), 4)

    def test_leiden(self):
        g = nx.karate_club_graph()
        coms = leiden(g)
        self.assertEqual(len(coms), 4)

    def test_greedy_modularity(self):
        g = nx.karate_club_graph()
        coms = greedy_modularity(g)
        self.assertEqual(len(coms), 3)

    def test_infomap(self):
        g = nx.karate_club_graph()
        coms = infomap(g)
        self.assertEqual(len(coms), 3)
        os.remove(".tree")

    def test_lp(self):
        g = nx.karate_club_graph()
        coms = label_propagation(g)
        self.assertEqual(len(coms), 3)

    def test_fluid(self):
        g = nx.karate_club_graph()
        coms = async_fluid(g, 3)
        self.assertEqual(len(coms), 3)


if __name__ == '__main__':
    unittest.main()
