import unittest

import cdlib
from cdlib import algorithms
from cdlib import benchmark
import networkx as nx


class STBenchTest(unittest.TestCase):
    def test_LFR(self):
        n = 250
        tau1 = 3
        tau2 = 1.5
        mu = 0.1
        G, coms = benchmark.LFR(n, tau1, tau2, mu, average_degree=5, min_community=20)
        self.assertIsInstance(G, nx.Graph)
        self.assertIsInstance(coms, cdlib.NodeClustering)
        nodes_coms = sum([len(c) for c in coms.communities])
        self.assertEqual(nodes_coms, G.number_of_nodes())

    def test_xmark(self):
        N = 2000
        gamma = 3
        beta = 2
        m_cat = ["auto", "auto"]
        theta = 0.3
        mu = 0.5
        avg_k = 10
        min_com = 20

        g, coms = benchmark.XMark(
            n=N,
            gamma=gamma,
            beta=beta,
            mu=mu,
            m_cat=m_cat,
            theta=theta,
            avg_k=avg_k,
            min_com=min_com,
            type_attr="categorical",
        )

        set1 = nx.get_node_attributes(g, "label_0")
        set2 = nx.get_node_attributes(g, "label_1")
        self.assertIsInstance(g, nx.Graph)
        self.assertIsInstance(coms, cdlib.NodeClustering)
        self.assertEqual(len(set(set1.values())), len(coms.communities))
        self.assertEqual(len(set(set2.values())), len(coms.communities))

    def test_grp(self):

        g, coms = benchmark.GRP(100, 10, 10, 0.25, 0.1)
        self.assertIsInstance(g, nx.Graph)
        self.assertIsInstance(coms, cdlib.NodeClustering)

    def test_planted_partitions(self):

        g, coms = benchmark.PP(4, 3, 0.5, 0.1, seed=42)
        self.assertIsInstance(g, nx.Graph)
        self.assertIsInstance(coms, cdlib.NodeClustering)

    def test_RPG(self):
        g, coms = benchmark.RPG([10, 10, 10], 0.25, 0.01)
        self.assertIsInstance(g, nx.Graph)
        self.assertIsInstance(coms, cdlib.NodeClustering)

    def test_SBM(self):
        sizes = [75, 75, 300]
        probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
        g, coms = benchmark.SBM(sizes, probs, seed=0)
        self.assertIsInstance(g, nx.Graph)
        self.assertIsInstance(coms, cdlib.NodeClustering)
