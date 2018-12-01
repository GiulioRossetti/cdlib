import unittest
from nclib.models.node_centric import demon, angel
from nclib.models.modularity import louvain, leiden
from nclib.models.map_equation import infomap
import networkx as nx
import os


class NodeCentricTests(unittest.TestCase):

    def test_demon(self):
        g = nx.karate_club_graph()
        coms = demon(g, epsilon=0.25)
        self.assertEqual(len(coms), 2)

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

    def test_infomap(self):
        g = nx.karate_club_graph()
        coms = infomap(g)
        self.assertEqual(len(coms), 3)
        os.remove(".tree")


if __name__ == '__main__':
    unittest.main()
