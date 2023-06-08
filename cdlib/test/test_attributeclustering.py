import unittest
from cdlib import algorithms
import networkx as nx
import random


class AttrCommunityDiscoveryTests(unittest.TestCase):
    def test_eva(self):

        l1 = ["one", "two", "three", "four"]
        l2 = ["A", "B", "C"]
        g = nx.barabasi_albert_graph(100, 5)

        labels = dict()

        for node in g.nodes():
            labels[node] = {"l1": random.choice(l1), "l2": random.choice(l2)}

        coms = algorithms.eva(g, labels, alpha=0.5)
        self.assertEqual(type(coms.communities), list)
        if len(coms.communities) > 0:
            self.assertEqual(type(coms.communities[0]), list)
            self.assertEqual(type(coms.communities[0][0]), int)

    def test_ilouvain(self):

        l1 = [0.1, 0.4, 0.5]
        l2 = [34, 3, 112]
        g = nx.barabasi_albert_graph(100, 5)
        labels = dict()

        for node in g.nodes():
            labels[node] = {"l1": random.choice(l1), "l2": random.choice(l2)}

        coms = algorithms.ilouvain(g, labels)

        self.assertEqual(type(coms.communities), list)
